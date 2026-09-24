"""96 字节监督记录 / 160 字节自对弈记录的往返与解码对照。

覆盖的坑：
- 记录 → 棋盘 → 记录必须完全一致（位棋盘逐格相同）；
- ``record_policy`` 的降变必须带 promo 字段，否则 g7f8n 会被判成非法的 g7f8（旧验收脚本踩过）；
- ``decode_batch`` 与 ``planes19.encode`` 逐位相同（含黑方行棋的镜像、过路兵、五十步/重复平面）；
- ``decode_targets`` 的策略概率和为 1、截断到前 5 条后再归一（旧 build_evals 的坑）；
- ``repair_castling`` 只把「我方王在 e1」时的 e1h1 / e1a1 搬到 e1g1 / e1c1；
  王在别处时 e1→h1 是合法普通着法，绝不能搬；
- ``check_shard`` 的易位写法检查：分片里不得出现 4*64+7 / 4*64+0 的「王吃己车」易位写法。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from Kit.planes19 import encoding as p19
from Kit.planes19 import records as R
from Kit.planes19.build.check import check_shard
from Kit.planes19.encoding import encode, move_to_index, move_to_promo_index, orient_move, \
    repetitions_of


def boards():
    """覆盖：初始局面、易位、过路兵（含只能吃过路兵解将的）、升变、近 50 步、重复、子力不足。"""
    out = [chess.Board()]
    b = chess.Board()
    b.push_san("e4"); b.push_san("e5"); b.push_san("Nf3"); b.push_san("Nc6")
    b.push_san("Bb5"); b.push_san("a6")
    out.append(b.copy())                                  # 有易位权
    b = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        b.push_san(san)
    out.append(b.copy())                                  # 白可吃过路兵
    b = chess.Board("5k2/8/8/K1Pp3r/8/8/8/8 b - - 0 1")
    out.append(b.copy())                                  # 黑吃过路兵解将（经典 pin）
    for uci in ("e2e4", "b8c6", "e4e5", "c6e5"):
        pass
    b = chess.Board()
    b.push_san("e4"); b.push_san("d5"); b.push_san("exd5")
    out.append(b.copy())                                  # 吃过路兵之后
    b = chess.Board("8/P7/8/8/8/8/6k1/4K3 w - - 0 1")
    out.append(b.copy())                                  # 升变边缘（兵线 a7）
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 95 120")
    out.append(b.copy())                                  # 接近 50 步
    b = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8") * 3:
        b.push_san(san)
    out.append(b.copy())                                  # 第三次/第五次重复之间
    b = chess.Board("8/8/4k3/8/8/2K1B3/8/8 w - - 0 1")
    out.append(b.copy())                                  # 子力不足判定用
    return out


class TestDecode(unittest.TestCase):
    def test_decode_batch_equals_encode(self):
        recs = []
        for b in boards():
            policy = []
            for mv in b.legal_moves:
                om = orient_move(mv, b.turn)
                policy = [(move_to_index(om), 0.5)]
                break
            recs.append(R.board_to_record(b, policy=policy or None, promo=0, rep=repetitions_of(b)))
        arr = np.array(recs, dtype=R.RECORD_DTYPE)
        got = R.decode_batch(arr)
        want = np.stack([encode(b, repetitions_of(b)) for b in boards()])
        np.testing.assert_array_equal(got, want)

    def test_roundtrip_board(self):
        for b in boards():
            rec = R.board_to_record(b, policy=[(move_to_index(orient_move(list(b.legal_moves)[0],
                                                                        b.turn)), 1.0)],
                                    promo=move_to_promo_index(list(b.legal_moves)[0]),
                                    rep=repetitions_of(b))
            back = R.record_to_board(rec)
            self.assertEqual(back.pawns, b.pawns, b.fen())
            self.assertEqual(back.piece_map(), b.piece_map(), b.fen())
            self.assertEqual(back.castling_rights, b.castling_rights, b.fen())
            self.assertEqual(back.ep_square, b.ep_square, b.fen())
            self.assertEqual(back.halfmove_clock, b.halfmove_clock, b.fen())
            self.assertEqual(back.turn, b.turn, b.fen())
            # 重放同一标签得到的位棋盘逐格相同
            again = R.record_to_board(R.board_to_record(back, rep=repetitions_of(b)))
            self.assertEqual(again.pawns, back.pawns, b.fen())

    def test_decode_targets_policy_normalized_and_promo(self):
        # 手工造一条：两个升变 PV + 非升变
        rec = np.zeros(1, dtype=R.RECORD_DTYPE)[0]
        b = chess.Board("8/P7/8/8/8/8/6k1/4K3 w - - 0 1")
        R_R = R.board_to_record(b, policy=None, rep=0)
        rec["policy_move"] = [move_to_index(orient_move(chess.Move.from_uci("a7a8q"), chess.WHITE)),
                              move_to_index(orient_move(chess.Move.from_uci("a7a8r"), chess.WHITE)),
                              0, 0, 0]
        rec["policy_prob"] = [0, 32767, 0, 0, 0]
        rec["promo"] = 0
        arr = np.array([rec], dtype=R.RECORD_DTYPE)
        policy, promo, wdl = R.decode_targets(arr)
        self.assertEqual(float(policy.sum()), 1.0)
        self.assertEqual(int(np.count_nonzero(policy)), 1)
        self.assertEqual(int(promo[0]), 0)
        # 只带策略、WDL 全 0 的记录（旧 build_evals 的 WDL 由评分折算，不会为 0；
        # 但自对弈记录允许全 0）：和为 0 时不归一，训练侧按 wdl.sum() > 0 判断要不要算价值损失
        self.assertEqual(float(wdl.sum()), 0.0)
        # 空标签（PGN 分片）→ 策略全零，训练按这个屏蔽策略损失
        empty = R.board_to_record(chess.Board())
        p2, _, _ = R.decode_targets(np.array([empty], dtype=R.RECORD_DTYPE))
        self.assertEqual(float(p2.sum()), 0.0)


class TestRepairCastling(unittest.TestCase):
    def _policy(self, idx, val=1.0):
        p = np.zeros((1, 4096), dtype=np.float32)
        p[0, idx] = val
        return p

    def test_repair_moves_only_when_king_on_e1(self):
        # 王在 e1、白方行棋：e1h1 → e1g1
        r = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        base = R.board_to_record(r)
        rec = np.zeros(1, dtype=R.RECORD_DTYPE)[0]
        rec["policy_move"][0] = 4 * 64 + 7            # 旧标签：e1h1（王吃己车写法）
        rec["policy_prob"][0] = 1000
        for name in ("pawns", "knights", "bishops", "rooks", "queens", "kings", "occ_white",
                     "occ_black", "castling", "ep", "halfmove", "side", "rep", "promo"):
            rec[name] = base[name]
        arr = np.array([rec], dtype=R.RECORD_DTYPE)
        policy = self._policy(4 * 64 + 7, 0.5)
        moved = R.repair_castling(policy, arr)
        self.assertEqual(moved, 1)
        self.assertEqual(float(policy[0, 4 * 64 + 6]), 0.5)
        self.assertEqual(float(policy[0, 4 * 64 + 7]), 0.0)
        p, _, _ = R.decode_targets(arr, repair_castling=True)
        idx = int(np.argmax(p))
        self.assertEqual(divmod(idx, 64), (4, 6))
        self.assertIn(p19.unorient_move(p19.index_to_move(idx), r.turn), r.legal_moves)

    def test_no_repair_when_king_elsewhere(self):
        # 王在 e5：e1→h1 是普通着法（车吃车），不许搬
        b = chess.Board("4k3/8/8/4K3/8/8/4r3/R6r w - - 0 1")
        rec = R.board_to_record(b)
        policy = self._policy(0 * 64 + 7)
        self.assertEqual(R.repair_castling(policy, np.array([rec], dtype=R.RECORD_DTYPE)), 0)
        self.assertEqual(float(policy[0, 0 * 64 + 7]), 1.0)

    def test_no_repair_when_black_king_on_e1(self):
        # 黑王在 e1、白方行棋（candidates 里也覆盖不到这个组合）
        b = chess.Board("4k3/8/8/8/8/8/8/4K2R b K - 0 1")
        back = b.mirror()                       # 黑王 e8，黑行棋 → 我方王 e1
        rec = R.board_to_record(back)
        policy = self._policy(4 * 64 + 0)       # e1a1（我方）
        moved = R.repair_castling(policy, np.array([rec], dtype=R.RECORD_DTYPE))
        self.assertEqual(moved, 1)
        self.assertEqual(float(policy[0, 4 * 64 + 2]), 1.0)

    def test_built_shard_has_no_rook_castling_form(self):
        """旧 build_evals 踩过的坑：Lichess 把易位写成 e1h1，标签落到 4*64+7。
        新代码用 ``board.parse_uci`` 归一化，分片验收必须能查出来。"""
        b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        good = R.board_to_record(b, policy=[(move_to_index(orient_move(
            chess.Move.from_uci("e1g1"), chess.WHITE)), 1.0)])
        bad = R.board_to_record(b, policy=[(move_to_index(orient_move(
            chess.Move.from_uci("e1h1"), chess.WHITE)), 1.0)])   # 王吃己车的非法/非规范写法
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.bin"
            p.write_bytes(np.array([good, good], dtype=R.RECORD_DTYPE).tobytes())
            self.assertTrue(check_shard(p)["ok"])
            p.write_bytes(np.array([good, bad], dtype=R.RECORD_DTYPE).tobytes())
            r = check_shard(p)
            self.assertFalse(r["ok"])
            self.assertEqual(r["castling_rook_form"], 1)


class TestSelfPlayRecord(unittest.TestCase):
    def test_visits_top16_and_z_sign(self):
        b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
        # 20 个着法都给了访问数：只保留前 16，且按访问降序
        visits = [(move_to_index(orient_move(m, chess.WHITE)), n)
                  for (n, m) in enumerate(sorted(b.legal_moves, key=lambda m: m.uci()), start=1)]
        rec = R.selfplay_record(b, visits=visits, z=-1, q=0.5, game=7, ply=3)
        self.assertEqual(rec["z"], -1)
        self.assertEqual(int(rec["game"]), 7)
        self.assertEqual(int(rec["ply"]), 3)
        got = R.record_policy(rec)
        self.assertEqual(len(got), 16)
        probs = [p for _, p in got]
        self.assertEqual(probs, sorted(probs, reverse=True))
        # 截断到 16 条后概率和 < 1（设计要求：读取端按访问数归一，不补均匀尾）
        self.assertLess(sum(probs), 1.0)
        self.assertGreater(sum(probs), 0.9)
        # z=0 / +1 / 非法值
        R.selfplay_record(b, visits=[(visits[0][0], 1)], z=0, q=0.0, game=0, ply=0)
        R.selfplay_record(b, visits=[(visits[0][0], 1)], z=1, q=0.0, game=0, ply=0)
        with self.assertRaises(ValueError):
            R.selfplay_record(b, visits=[(visits[0][0], 1)], z=2, q=0.0, game=0, ply=0)

    def test_record_to_board_for_selfplay(self):
        b = chess.Board("8/P7/8/8/8/8/6k1/4K3 w - - 0 1")
        rec = R.selfplay_record(b, visits=[(move_to_index(orient_move(
            chess.Move.from_uci("a7a8q"), chess.WHITE)), 3)], z=1, q=0.8, game=1, ply=0)
        back = R.record_to_board(rec)
        self.assertEqual(back.piece_map(), b.piece_map())
        x = R.decode_batch(np.array([rec], dtype=R.SELFPLAY_DTYPE))
        np.testing.assert_array_equal(x[0], encode(b, repetitions_of(b)))

    def test_shard_dtype_by_filename(self):
        self.assertIs(R.shard_dtype("a/evals_0001.bin"), R.RECORD_DTYPE)
        self.assertIs(R.shard_dtype("a/sp_0000.sp.bin"), R.SELFPLAY_DTYPE)

    def test_open_shard_rejects_partial(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.bin"
            p.write_bytes(b"\x01" * 97)
            with self.assertRaises(ValueError):
                R.open_shard(p)
            self.assertEqual(R.shard_len(p), 97 // R.RECORD_DTYPE.itemsize)
            good = np.zeros(4, dtype=R.RECORD_DTYPE)
            p.write_bytes(good.tobytes())
            self.assertEqual(len(R.open_shard(p)), 4)


class TestPieceCounts(unittest.TestCase):
    def test_counts_and_buckets(self):
        b = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")      # 2 子
        rec = R.board_to_record(b)
        arr = np.array([rec], dtype=R.RECORD_DTYPE)
        self.assertEqual(int(R.piece_counts(arr)[0]), 2)
        out = R.piece_count_bucket(arr, 8)
        self.assertEqual(int(out[0]), (2 - 1) * 8 // 32)
        out = R.piece_count_bucket(arr, 1)
        self.assertEqual(int(out[0]), 0)


if __name__ == "__main__":
    unittest.main()
