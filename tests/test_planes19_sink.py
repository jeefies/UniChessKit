"""自对弈 sink：记录字段、z 符号、截断标记、崩溃恢复、续跑跳过。

覆盖的坑：
- ``visits`` 索引是行棋方视角（与监督分片、planes19 一致），黑方行棋必须镜像，
  否则训练时策略头和推理端对不上；
- 终局 z 要按行棋方视角翻转：1-0 时白方 +1、黑方 -1；
- 截断局（run_selfplay 的 termination == "truncated"）必须打 flag 且 z 记 0——
  否则网络会学着一个没有真实结局的胜负标签；
- 进程被杀在写一半时：sink 用旁边的 .games.jsonl 元数据把分片截回整数条，
  不留半条记录（``open_shard`` 会校验）；
- 没有访问分布的决策（开局书 / 残局表 / 网络直出）不落记录——它们没有搜索树可蒸馏。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chess

from Kit.api.types import MoveDecision
from Kit.planes19 import records as R
from Kit.planes19.encoding import (index_to_move, move_to_index, orient_move, unorient_move)
from Kit.planes19.sink import SelfPlayShardSink, game_records


def visit_pairs(board):
    """给 board 的每个合法着法一个不同的访问数，形状与 SearchPlayer.info["visits"] 相同：
    [(uci 字符串, 访问数), ...]（行棋方视角的着法）。"""
    return [(m.uci(), i + 1) for i, m in enumerate(board.legal_moves)]


def decision(move, visits=None, q=0.3):
    """visits 用行棋方视角的 [move, n] 列表（SearchPlayer.info["visits"] 的形状）。"""
    info = {"source": "search", "q": q}
    if visits:
        info["visits"] = visits
    return MoveDecision(move, "search", info)


def visited(board, move, weights=None):
    """board 上 move 的最高访问数，其余着法递减——够验证排序与截断。"""
    moves = list(board.legal_moves)
    if move not in moves:
        raise AssertionError(f"{move} 不在 {board.fen()} 的合法着法里")
    out = []
    for i, m in enumerate(moves):
        out.append([m, (100 - i * 7) if m == move else 40 - i])
    return out


class TestGameRecords(unittest.TestCase):
    def _decisions(self, ucis):
        """按逐 ply 的真实局面给访问分布（与 SearchPlayer 一致）。"""
        board = chess.Board()
        out = []
        for u in ucis:
            out.append(decision(chess.Move.from_uci(u), visit_pairs(board)))
            board.push(chess.Move.from_uci(u))
        return out

    def test_z_is_side_relative(self):
        ucis = ["e2e4", "e7e5", "g1f3"]
        rec = {"game": 3, "result": "1-0", "termination": "checkmate", "plies": 3,
               "book_plies": 0, "moves": ucis}
        recs = game_records(rec, self._decisions(ucis))
        self.assertEqual(len(recs), 3)
        self.assertEqual([int(r["z"]) for r in recs], [1, -1, 1])
        self.assertEqual([int(r["ply"]) for r in recs], [0, 1, 2])
        self.assertEqual(int(recs[0]["game"]), 3)
        self.assertEqual(int(recs[0]["flags"]), 0)
        # 黑方行棋的那条：整份访问分布都要镜像落盘（visit_pairs 的访问数按枚举序递增，
        # selfplay_record 取访问最多的 16 条 → 即枚举序的后 16 条）
        board1 = chess.Board()
        board1.push(chess.Move.from_uci("e2e4"))
        pairs1 = visit_pairs(board1)
        top = sorted(pairs1, key=lambda t: (-t[1], move_to_index(orient_move(
            chess.Move.from_uci(t[0]), chess.BLACK))))[:16]
        want = set(move_to_index(orient_move(chess.Move.from_uci(t[0]), chess.BLACK))
                   for t in top)
        got = set(int(v) for v in recs[1]["visit_move"] if v)
        self.assertEqual(got, want)
        self.assertIn(len(got), (16, min(16, len(pairs1))))

    def test_black_to_move_indices_are_mirrored(self):
        """黑方行棋时落盘索引必须镜像成「行棋方视角」。

        orient_move 本身的代数性质由 test_planes19 的黄金数据覆盖，这里只证明 sink 真的按
        当时行棋方做了映射：给一条**布局上下不对称**的局面，黑白索引必然不同。
        """
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1")
        # 角色的第一手就是黑方走的：用记录里的首着之前没有前置着法来保证 turn == BLACK
        board0 = chess.Board()
        board0.push_san("e4")
        got_uci = "c7c5"          # 黑方 c7→c5；镜像后应为白方 c2c4
        pairs = [(got_uci, 5)]
        rec = {"game": 1, "result": "1/2-1/2", "termination": "adjudicated", "plies": 2,
               "book_plies": 1, "moves": ["e2e4", got_uci]}
        recs = game_records(rec, [decision(chess.Move.from_uci("e2e4"), [("e2e4", 3)]),
                                  decision(chess.Move.from_uci(got_uci), pairs)])
        want = move_to_index(orient_move(chess.Move.from_uci(got_uci), chess.BLACK))
        naive = move_to_index(orient_move(chess.Move.from_uci(got_uci), chess.WHITE))
        self.assertNotEqual(want, naive)
        # 第二条记录（黑方走的那手）必须用黑方视角落盘
        self.assertEqual(int(recs[1]["visit_move"][0]), want)
        self.assertEqual(unorient_move(index_to_move(int(recs[1]["visit_move"][0])),
                                       chess.BLACK).uci(), got_uci)
        self.assertEqual(int(recs[1]["z"]), 0)

    def test_truncated_flags(self):
        board = chess.Board()
        rec = {"game": 5, "result": "1/2-1/2", "termination": "truncated", "plies": 1,
               "book_plies": 0, "moves": ["e2e4"]}
        recs = game_records(rec, [decision(chess.Move.from_uci("e2e4"), visit_pairs(board))])
        self.assertEqual(int(recs[0]["flags"]), R.FLAG_TRUNCATED)
        self.assertEqual(int(recs[0]["z"]), 0)

    def test_decisions_without_visits_skipped(self):
        board = chess.Board()
        rec = {"game": 1, "result": "1-0", "termination": "resign", "plies": 3, "book_plies": 2,
               "moves": ["e2e4", "e7e5", "g1f3"]}
        recs = game_records(rec, [decision(chess.Move.from_uci("e2e4")),   # 开局书，无访问分布
                                  decision(chess.Move.from_uci("e7e5"), visit_pairs(board)),
                                  decision(chess.Move.from_uci("g1f3"))])
        self.assertEqual(len(recs), 1)
        self.assertEqual(int(recs[0]["ply"]), 1)


class TestSink(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="p19sink_")
        self.path = Path(self.d) / "sp_0000.sp.bin"
        self.sink = SelfPlayShardSink(self.path)

    def _game(self, gid, n=3):
        board = chess.Board()
        ucis = ["e2e4", "e7e5", "g1f3"][:n]
        decisions = [decision(chess.Move.from_uci(u), visit_pairs(board))
                     for u in ucis]
        return {"game": gid, "result": "1-0", "termination": "checkmate", "plies": n,
                "book_plies": 0, "moves": ucis}, decisions

    def test_writes_and_done_games(self):
        rec, decs = self._game(0)
        self.sink.on_game_end(rec, chess.Board(), decs)
        self.assertEqual(self.sink.done_games(), {0})
        self.assertTrue(self.path.exists())
        # 大小应是 160 的整数倍
        self.assertEqual(self.path.stat().st_size % R.SELFPLAY_DTYPE.itemsize, 0)
        want = len(game_records(rec, decs))
        self.assertEqual(self.sink.records, want)
        meta = json.loads(self.path.with_name("sp_0000.games.jsonl").read_text())
        self.assertEqual(meta["game"], 0)
        self.assertEqual(meta["records"], want)

    def test_repair_truncates_partial_shard(self):
        """尾部写了一半：重新打开时按元数据截回整数条。"""
        rec, decs = self._game(0)
        self.sink.on_game_end(rec, chess.Board(), decs)
        n = self.path.stat().st_size
        with open(self.path, "ab") as f:
            f.write(b"\x01" * 50)              # 模拟写了一半
        self.assertNotEqual(self.path.stat().st_size, n)
        sink2 = SelfPlayShardSink(self.path)
        self.assertEqual(self.path.stat().st_size, n)
        self.assertEqual(sink2.done_games(), {0})
        self.assertEqual(len(R.open_shard(self.path)), n // R.SELFPLAY_DTYPE.itemsize)

    def test_repair_truncates_when_meta_lost(self):
        """元数据也丢了：截到 0（宁可少一局，不留半条）。"""
        rec, decs = self._game(0)
        self.sink.on_game_end(rec, chess.Board(), decs)
        self.path.with_name("sp_0000.games.jsonl").unlink()
        sink2 = SelfPlayShardSink(self.path)
        self.assertEqual(self.path.stat().st_size, 0)
        self.assertEqual(sink2.done_games(), set())
        sink2.on_game_end(rec, chess.Board(), decs)
        self.assertEqual(sink2.done_games(), {0})

    def test_bad_suffix(self):
        with self.assertRaises(ValueError):
            SelfPlayShardSink(Path(self.d) / "x.bin")


if __name__ == "__main__":
    unittest.main()
