"""自对弈管线（``pipelines.selfplay``）测试：开局分配、种子、book 强制、裁决、Sink 契约、攒批。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from Kit.api import EvalRequest, MoveDecision, PlayerError, immediate
from Kit.pipelines.selfplay import (SelfPlayConfig, game_seed_sequence,
                                    load_book_lines, plan_selfplay, run_selfplay)


class _CountEval:
    model_key = "count"

    def __init__(self):
        self.batches = []

    def evaluate(self, payloads):
        self.batches.append(len(payloads))
        return [None] * len(payloads)


class _BookRandomPlayer:
    """走 book，之后按每局随机数流随机走；每步做一次假前向，用来检验跨局攒批。"""

    def __init__(self, ev, obey_book=True):
        self.name = "fake"
        self.ev = ev
        self.obey_book = obey_book
        self.observed = 0
        self.closed = False

    def new_game(self, start):
        self.start = start
        self.rng = np.random.default_rng(game_seed_sequence(start.seed, start.index))
        return immediate(None)

    def choose(self, board, budget):
        yield EvalRequest(self.ev, [board.fen()])
        ply = len(board.move_stack)
        moves = list(board.legal_moves)
        pick = moves[int(self.rng.integers(len(moves)))]
        if self.obey_book and ply < len(self.start.book):
            return MoveDecision(chess.Move.from_uci(self.start.book[ply]), source="book",
                                info={"ply": ply})
        return MoveDecision(pick, source="search", info={"ply": ply})

    def observe(self, board, move):
        self.observed += 1
        return immediate(None)

    def close(self):
        self.closed = True


class _IndexDerivedPlayer(_BookRandomPlayer):
    """按 ``(seed, index)`` 派生随机数流的 Player——真实引擎的口径。

    ``SsmSelfPlayer`` 用 ``default_rng(SeedSequence(seed, spawn_key=(index,)))``；
    ``SearchPlayer.new_game`` 在 ``index != 0`` 时同样这么派生。管线只负责把全局
    seed 原样传下去并给出 ``GameStart.index``，派生是 Player 自己的事。
    """

    def new_game(self, start):
        self.start = start
        self.rng = np.random.default_rng(
            np.random.SeedSequence(int(start.seed), spawn_key=(int(start.index),)))
        return immediate(None)


class _Sink:
    def __init__(self):
        self.games = []

    def on_game_end(self, record, board, decisions):
        self.games.append((record, board.copy(), list(decisions)))


def _write(lines):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


class TestPlanning(unittest.TestCase):
    def test_seed_sequence_matches_spawn(self):
        for seed in (0, 42, 123456789):
            kids = np.random.SeedSequence(seed).spawn(7)
            for i in (0, 3, 6):
                a = np.random.default_rng(kids[i]).random(8)
                b = np.random.default_rng(game_seed_sequence(seed, i)).random(8)
                self.assertTrue(np.array_equal(a, b))

    def test_book_lines_truncate_keep_duplicates_drop_illegal(self):
        path = _write(["e4 e5 Nf3 Nc6 Bb5 a6 Ba4", "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6",
                       "d4 Qxd7", "", "c4 # 英国式", "d2d4 d7d5"])
        try:
            lines, dropped = load_book_lines(path, 6)
        finally:
            os.unlink(path)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], lines[1])                  # 裁切后重复，仍各占一个序号
        self.assertEqual(len(lines[0]), 6)
        self.assertEqual(lines[2], ("c2c4",))
        self.assertEqual(lines[3], ("d2d4", "d7d5"))

    def test_book_lines_drop_null_move(self):
        """"0000"：解析期就要丢掉，不能带进对局。

        2026-09-27 修：python-chess 的 parse_san 把 "0000" 当 null move 收下，
        ``load_book_lines`` 原来会留下 ``("e2e4", "e7e5", "0000")`` 这种"合法"行——
        selfplay 跑到这一 ply 才 PlayerError 整批中止（算力全废），arena 则 push 成
        空着悄悄翻转行棋方。现在 parse_line 直接拒绝，这里记 dropped。
        """
        path = _write(["e4 e5 Nf3", "e4 e5 0000", "0000", "d2d4 d7d5"])
        try:
            lines, dropped = load_book_lines(path, 6)
        finally:
            os.unlink(path)
        self.assertEqual(dropped, 2)
        self.assertEqual(lines, [("e2e4", "e7e5", "g1f3"), ("d2d4", "d7d5")])

    def test_plan_round_robin(self):
        cfg = SelfPlayConfig(games=7)
        lines = [("e2e4",), ("d2d4",), ("c2c4",)]
        tasks = plan_selfplay(cfg, lines)
        self.assertEqual([t.book_id for t in tasks], [0, 1, 2, 0, 1, 2, 0])
        self.assertEqual(tasks[4].book, ("d2d4",))
        self.assertTrue(all(t.book == () and t.book_id is None for t in plan_selfplay(cfg)))

    def test_plan_first_game_offsets_global_index(self):
        lines = [("e2e4",), ("d2d4",), ("c2c4",)]
        whole = plan_selfplay(SelfPlayConfig(games=7), lines)
        parts = (plan_selfplay(SelfPlayConfig(games=4), lines)
                 + plan_selfplay(SelfPlayConfig(games=3, first_game=4), lines))
        self.assertEqual(parts, whole)                        # 拆分与整跑同一批任务
        with self.assertRaises(ValueError):
            SelfPlayConfig(games=1, first_game=-1)


class TestRunSelfPlay(unittest.TestCase):
    def _run(self, concurrency, games=6, max_plies=20, openings=None, obey=True):
        ev = _CountEval()
        players = []

        def make():
            p = _BookRandomPlayer(ev, obey)
            players.append(p)
            return p

        sink = _Sink()
        cfg = SelfPlayConfig(games=games, seed=5, max_plies=max_plies, concurrency=concurrency,
                             openings=openings, book_plies=4)
        summary = run_selfplay(cfg, make, sink)
        return summary, sink, players, ev

    def test_records_and_book(self):
        path = _write(["e4 e5 Nf3 Nc6 Bb5", "d4 d5"])
        try:
            summary, sink, players, ev = self._run(1, openings=path)
        finally:
            os.unlink(path)
        self.assertEqual(summary["games"], 6)
        self.assertEqual(len(sink.games), 6)
        self.assertEqual([r["game"] for r, _, _ in sink.games], list(range(6)))   # 并发 1 = 局序
        for (rec, board, decisions), p in zip(sink.games, players):
            book = ("e2e4", "e7e5", "g1f3", "b8c6") if rec["game"] % 2 == 0 else ("d2d4", "d7d5")
            self.assertEqual(tuple(rec["moves"][:len(book)]), book)
            self.assertEqual(rec["book_plies"], len(book))
            self.assertEqual(rec["book_id"], rec["game"] % 2)
            self.assertEqual(len(decisions), rec["plies"])
            self.assertEqual([d.info["ply"] for d in decisions], list(range(rec["plies"])))
            self.assertEqual([m.uci() for m in board.move_stack], rec["moves"])
            self.assertLessEqual(rec["plies"], 20)
            self.assertTrue(p.start.both_sides)
            self.assertEqual(p.observed, rec["plies"])           # 单 Player：每步只 observe 一次
            self.assertTrue(p.closed)
        self.assertEqual(summary["book_plies"], 3 * 4 + 3 * 2)

    def test_truncation_counts_whole_game(self):
        summary, sink, _, _ = self._run(1, games=2, max_plies=5)
        for rec, _, _ in sink.games:
            self.assertEqual(rec["plies"], 5)
            self.assertEqual(rec["termination"], "truncated")
        self.assertEqual(summary["termination"], {"truncated": 2})

    def test_concurrency_batches_and_same_games(self):
        _, s1, _, ev1 = self._run(1)
        _, s4, _, ev4 = self._run(4)
        self.assertEqual(max(ev1.batches), 1)
        self.assertEqual(max(ev4.batches), 4)
        by_game = lambda s: {r["game"]: r["moves"] for r, _, _ in s.games}   # noqa: E731
        self.assertEqual(by_game(s1), by_game(s4))       # 随机数只取决于 (seed, 局序号)

    def test_split_run_equals_whole_run(self):
        path = _write(["e4 e5 Nf3 Nc6 Bb5", "d4 d5", "c4"])
        try:
            _, whole, _, _ = self._run(3, games=7, openings=path)
            ev = _CountEval()
            parts = _Sink()
            for first, n in ((0, 3), (3, 4)):
                cfg = SelfPlayConfig(games=n, seed=5, max_plies=20, concurrency=2,
                                     openings=path, book_plies=4, first_game=first)
                run_selfplay(cfg, lambda: _BookRandomPlayer(ev), parts)
        finally:
            os.unlink(path)
        key = lambda s: sorted((r["game"], r["book_id"], tuple(r["moves"])) for r, _, _ in s.games)  # noqa: E731
        self.assertEqual(key(parts), key(whole))

    def test_book_violation_raises(self):
        path = _write(["e4 e5"])
        try:
            with self.assertRaises(PlayerError):
                self._run(2, openings=path, obey=False)
        finally:
            os.unlink(path)

    def test_empty_openings_file_rejected(self):
        path = _write(["Qxd7"])
        try:
            with self.assertRaises(ValueError):
                self._run(1, openings=path)
        finally:
            os.unlink(path)


class TestSelfPlayShardSink(unittest.TestCase):
    """run_selfplay → SelfPlayShardSink 全链路口径（fake 引擎，不落正式目录）。

    钉住 2026-09-27 核对过的三条口径：
    - **每局记录数 == plies − book_plies**：开局书着法原样走、不搜索（无访问分布），
      sink 跳过——文档里"arena 记录数 = plies − 6 个开局 ply"说的就是这件事；
    - 分片大小始终是 160 字节的整数倍，元数据与分片逐局对得上；
    - 重开 sink（续跑）能跳过已写的局，同一批任务不会重复落盘。
    """

    def _run(self, d, path, **kw):
        from Kit.planes19 import records as R
        from Kit.planes19.sink import SelfPlayShardSink
        from Kit.players import SearchPlayer
        from Kit.planes19 import Planes19Expander
        from Kit.search import PUCTConfig
        from Kit.testing.fakes import FakePlanesEvaluator

        ev = FakePlanesEvaluator("fake:sink")

        def make():
            return SearchPlayer("probe", Planes19Expander(ev), simulations=8,
                                puct=PUCTConfig(batch_size=8))

        sink = SelfPlayShardSink(Path(d) / path)
        cfg = SelfPlayConfig(games=4, seed=7, max_plies=24, concurrency=2, **kw)
        summary = run_selfplay(cfg, make, sink)
        return summary, sink, R.SELFPLAY_DTYPE.itemsize

    def test_records_exclude_book_plies_and_resume_skips(self):
        book = _write(["e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", "d2d4 d7d5"])
        try:
            with tempfile.TemporaryDirectory() as d:
                summary, sink, isz = self._run(d, "sp_0000.sp.bin", openings=book, book_plies=6)
                shard = Path(d) / "sp_0000.sp.bin"
                meta = [json.loads(l) for l in
                        (Path(d) / "sp_0000.games.jsonl").read_text(encoding="utf-8")
                        .splitlines()]
                self.assertEqual(summary["games"], 4)
                self.assertEqual(len(meta), 4)
                self.assertEqual(shard.stat().st_size, sink.records * isz)
                self.assertEqual(sink.records, sum(m["records"] for m in meta))
                self.assertEqual(sink.done_games(), {0, 1, 2, 3})
                for m in meta:
                    task_book = 6 if m["game"] % 2 == 0 else 2
                    self.assertLessEqual(m["plies"], 24)
                    self.assertEqual(m["records"], m["plies"] - task_book,
                                     f"第 {m['game']} 局：记录数必须是 plies − 开局 ply 数")
                # 记录里的 ply 不含开局 ply（开局着法没有访问分布）
                from Kit.planes19 import records as R
                mm = R.open_shard(shard)
                rows = [(int(r["game"]), int(r["ply"])) for r in mm]
                del mm                        # 先关掉内存映射：Windows 上临时目录才删得掉
                by_game = {}
                for gid, ply in rows:
                    by_game.setdefault(gid, []).append(ply)
                for m in meta:
                    plies = by_game[m["game"]]
                    self.assertEqual(len(plies), m["records"])
                    self.assertEqual(plies, list(range(plies[0], plies[0] + len(plies))))
                    self.assertGreaterEqual(plies[0], 6 if m["game"] % 2 == 0 else 2)
                # 续跑：同样的任务全被跳过，分片不再长
                summary2 = run_selfplay(SelfPlayConfig(games=4, seed=7, max_plies=24,
                                                        concurrency=2, openings=book,
                                                        book_plies=6),
                                        self._make_player(), sink,
                                        skip_games=sink.done_games())
                self.assertEqual(summary2["games"], 0)
                self.assertEqual(shard.stat().st_size, sink.records * isz)
        finally:
            os.unlink(book)

    @staticmethod
    def _make_player():
        from Kit.planes19 import Planes19Expander
        from Kit.players import SearchPlayer
        from Kit.search import PUCTConfig
        from Kit.testing.fakes import FakePlanesEvaluator

        ev = FakePlanesEvaluator("fake:sink")
        return lambda: SearchPlayer("probe", Planes19Expander(ev), simulations=8,
                                    puct=PUCTConfig(batch_size=8))


class TestPerGameSeed(unittest.TestCase):
    """每局的随机数流必须按局号派生（G1：全局 seed 会让同一批对局变成同一盘棋）。

    分工：管线把全局 seed 原样交给 Player 并给出 ``GameStart.index``；由 Player 按
    (seed, index) 派生自己的流。不能反过来在管线里派生整数种子——S 的开局 π′ 缓存键
    是 ``(seed, book_id, ply)``，故意不含局序号，同一条开局要在所有局里共享搜索结果。
    """

    def _run(self, games=6, seed=5, max_plies=20, concurrency=1):
        ev = _CountEval()
        sink = _Sink()
        cfg = SelfPlayConfig(games=games, seed=seed, max_plies=max_plies,
                            concurrency=concurrency)
        run_selfplay(cfg, lambda: _IndexDerivedPlayer(ev), sink)
        return sink

    def test_pipeline_passes_global_seed_and_game_index(self):
        seen = self._run(games=5).games
        moves = [tuple(r["moves"]) for r, _, _ in seen]
        self.assertEqual(len(moves), 5)
        self.assertEqual(len(set(moves)), 5)

    def test_seed_and_index_reach_the_player(self):
        starts = []
        ev = _CountEval()

        class _Recorder(_IndexDerivedPlayer):
            def new_game(self, start):
                starts.append((start.seed, start.index))
                return super().new_game(start)

        sink = _Sink()
        run_selfplay(SelfPlayConfig(games=4, seed=9, max_plies=12, concurrency=1),
                     lambda: _Recorder(ev), sink)
        seeds = {s for s, _ in starts}
        idx = {i for _, i in starts}
        self.assertEqual(seeds, {9}, "seed 必须原样传下去（S 的 π′ 缓存键依赖它）")
        self.assertEqual(idx, {0, 1, 2, 3}, "每局必须给出不同的 GameStart.index")

    def test_same_seed_reproduces_games(self):
        a = [tuple(r["moves"]) for r, _, _ in self._run().games]
        b = [tuple(r["moves"]) for r, _, _ in self._run().games]
        self.assertEqual(a, b)                        # 同配置可复现

    def test_split_by_game_index_matches_whole(self):
        whole = self._run(games=5).games
        by_game = {r["game"]: tuple(r["moves"]) for r, _, _ in whole}
        parts = []
        for first, n in ((0, 2), (2, 3)):
            ev = _CountEval()
            sink = _Sink()
            run_selfplay(SelfPlayConfig(games=n, seed=5, max_plies=20, concurrency=1,
                                       first_game=first),
                         lambda: _IndexDerivedPlayer(ev), sink)
            parts += sorted((r["game"], tuple(r["moves"])) for r, _, _ in sink.games)
        self.assertEqual(parts, sorted(by_game.items()))
        self.assertEqual(len(parts), 5)


if __name__ == "__main__":
    unittest.main()
