"""自对弈管线（``pipelines.selfplay``）测试：开局分配、种子、book 强制、裁决、Sink 契约、攒批。"""
from __future__ import annotations

import os
import tempfile
import unittest

import chess
import numpy as np

from unichess_kit.api import EvalRequest, MoveDecision, PlayerError, immediate
from unichess_kit.pipelines.selfplay import (SelfPlayConfig, game_seed_sequence, load_book_lines,
                                             plan_selfplay, run_selfplay)


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


if __name__ == "__main__":
    unittest.main()
