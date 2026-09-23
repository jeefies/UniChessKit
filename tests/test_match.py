import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chess

from unichess_kit.api import PlayerError, immediate
from unichess_kit.api.types import MoveDecision
from unichess_kit.pipelines.match import (MatchConfig, SprtConfig, plan_games, run_match,
                                          summarize)
from unichess_kit.players import RandomPlayer
from unichess_kit.registry import EngineSpec
from unichess_kit.testing.fakes import (make_failing_player_factory, make_fake_player_factory,
                                        make_random_player_factory)

KIT = "unichess_kit.testing.fakes"


def strip(records):
    return [{k: v for k, v in r.items() if k != "elapsed_s"} for r in records]


def read_games(path):
    lines = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l]
    return lines[0], sorted((r for r in lines[1:] if r["type"] == "game"), key=lambda r: r["game"])


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestPlan(unittest.TestCase):
    def test_pairs_share_opening_and_swap_colours(self):
        from unichess_kit.pipelines.match import load_book
        cfg = MatchConfig(pairs=5, seed=2)
        tasks = plan_games(cfg, load_book("bundled"))
        self.assertEqual([t.game for t in tasks], list(range(10)))
        for p in range(5):
            a, b = tasks[2 * p], tasks[2 * p + 1]
            self.assertEqual((a.pair, b.pair), (p, p))
            self.assertEqual(a.opening, b.opening)
            self.assertEqual((a.a_is_white, b.a_is_white), (True, False))
        self.assertEqual(len({tasks[2 * p].opening for p in range(5)}), 5)
        self.assertEqual(len({t.seed_a for t in tasks} | {t.seed_b for t in tasks}), 20)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            MatchConfig(pairs=0)
        with self.assertRaises(ValueError):
            MatchConfig.from_dict({"pairs": 1, "bogus": 1})
        cfg = MatchConfig.from_dict({"pairs": 2, "sprt": {"elo1": 20}})
        self.assertEqual(cfg.sprt.elo1, 20)


class TestMatch(Base):
    def run_fake(self, out=None, pairs=3, **kw):
        cfg = MatchConfig(pairs=pairs, max_plies=60, concurrency=4, **kw)
        return run_match(cfg, make_a=make_fake_player_factory("a", simulations=16),
                         make_b=make_fake_player_factory("b", salt="x", simulations=8),
                         out_path=out, names={"A": "a", "B": "b"})

    def test_scoring_by_model_and_records(self):
        out = self.dir / "r.jsonl"
        s = self.run_fake(out)
        header, games = read_games(out)
        self.assertEqual(header["type"], "header")
        self.assertEqual(len(games), 6)
        self.assertEqual(s["a_wins"] + s["b_wins"] + s["draws"], 6)
        for g in games:
            board = chess.Board()
            for u in g["opening"] + g["moves"]:
                board.push_uci(u)                  # 棋谱可重放
            self.assertEqual(g["plies"], len(board.move_stack))
            white_score = {"1-0": 1.0, "0-1": 0.0}.get(g["result"], 0.5)
            self.assertEqual(g["a_score"], white_score if g["white"] == "A" else 1 - white_score)
            self.assertLessEqual(g["plies"], 60)
            if g["termination"] == "truncated":
                self.assertEqual(g["plies"], 60)
        self.assertTrue(Path(str(out) + ".summary.json").exists())
        self.assertEqual(s["games_planned"], 6)

    def test_reproducible(self):
        a = self.dir / "a.jsonl"
        b = self.dir / "b.jsonl"
        sa, sb = self.run_fake(a), self.run_fake(b)
        self.assertEqual(strip(read_games(a)[1]), strip(read_games(b)[1]))
        self.assertEqual(sa["batch"], sb["batch"])       # 批的组成也一致

    def test_concurrency_does_not_change_games(self):
        """跨局攒批只改变批的拼法，不改变任何一局（fake 评估器逐局面独立）。"""
        a = self.dir / "a.jsonl"
        b = self.dir / "b.jsonl"
        run_match(MatchConfig(pairs=2, max_plies=40, concurrency=1),
                  make_a=make_fake_player_factory("a"), make_b=make_random_player_factory(),
                  out_path=a, names={"A": "a", "B": "r"})
        run_match(MatchConfig(pairs=2, max_plies=40, concurrency=4),
                  make_a=make_fake_player_factory("a"), make_b=make_random_player_factory(),
                  out_path=b, names={"A": "a", "B": "r"})
        self.assertEqual(strip(read_games(a)[1]), strip(read_games(b)[1]))

    def test_resume_after_crash(self):
        full = self.dir / "full.jsonl"
        self.run_fake(full)
        part = self.dir / "part.jsonl"
        lines = full.read_text(encoding="utf-8").splitlines(keepends=True)
        part.write_text("".join(lines[:4]) + lines[4][:25], encoding="utf-8")   # 3 局 + 半行
        s = self.run_fake(part)
        self.assertEqual(strip(read_games(part)[1]), strip(read_games(full)[1]))
        self.assertEqual(s["games"], 6)
        again = self.run_fake(part)                   # 已全部完成：不再下棋
        self.assertEqual(again["games"], 6)
        self.assertEqual(again["batch"], {})

    def test_resume_rejects_other_config(self):
        out = self.dir / "r.jsonl"
        self.run_fake(out, pairs=1)
        with self.assertRaisesRegex(ValueError, "配置哈希"):
            self.run_fake(out, pairs=2)

    def test_resume_rejects_corrupt_middle_line(self):
        out = self.dir / "r.jsonl"
        self.run_fake(out, pairs=1)
        lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
        out.write_text(lines[0] + "{garbage\n" + "".join(lines[1:]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "损坏"):
            self.run_fake(out, pairs=1)

    def test_distinct_games_without_openings(self):
        """无开局 + 确定性双方：每对第二盘都重复第一对 → duplicate_rate 必须报出来。"""
        s = run_match(MatchConfig(pairs=3, max_plies=30, openings=None),
                      make_a=make_fake_player_factory("a", simulations=0),
                      make_b=make_fake_player_factory("b", salt="x", simulations=0),
                      names={"A": "a", "B": "b"})
        self.assertEqual(s["distinct_games"], 2)
        self.assertAlmostEqual(s["duplicate_rate"], 1 - 2 / 6)

    def test_error_stops_whole_batch(self):
        with self.assertRaisesRegex(RuntimeError, "故意失败"):
            run_match(MatchConfig(pairs=4, max_plies=60),
                      make_a=make_failing_player_factory(2), make_b=make_random_player_factory(),
                      out_path=self.dir / "e.jsonl")

    def test_illegal_move_raises(self):
        class Cheater(RandomPlayer):
            def choose(self, board, budget):
                return immediate(MoveDecision(chess.Move.from_uci("e2e5"), "cheat"))
        with self.assertRaises(PlayerError):
            run_match(MatchConfig(pairs=1, openings=None), make_a=Cheater,
                      make_b=make_random_player_factory())

    def test_players_closed(self):
        closed = []

        class Tracking(RandomPlayer):
            def close(self):
                closed.append(1)
        run_match(MatchConfig(pairs=2, max_plies=10), make_a=Tracking,
                  make_b=make_random_player_factory())
        self.assertEqual(len(closed), 4)

    def test_sprt_early_stop(self):
        s = run_match(MatchConfig(pairs=40, max_plies=80, concurrency=2,
                                  sprt=SprtConfig(elo0=0, elo1=200, min_pairs=4)),
                      make_a=make_fake_player_factory("a", simulations=24),
                      make_b=make_random_player_factory(), names={"A": "a", "B": "r"})
        self.assertIn(s["sprt"]["verdict"], ("H0", "H1"))
        self.assertTrue(s["stopped_by_sprt"])
        self.assertLess(s["games"], 80)

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            run_match(MatchConfig(pairs=1), make_a=RandomPlayer)
        with self.assertRaises(ValueError):
            run_match(MatchConfig(pairs=1, workers=2), make_a=RandomPlayer, make_b=RandomPlayer)


class TestMatchWorkers(Base):
    def spec(self, factory, **kwargs):
        return EngineSpec(factory=f"{KIT}:{factory}", kwargs=kwargs)

    def test_workers_same_games_as_single_process(self):
        a = self.dir / "a.jsonl"
        b = self.dir / "b.jsonl"
        common = dict(spec_a=self.spec("make_fake_player_factory", name="a", simulations=8),
                      spec_b=self.spec("make_random_player_factory"))
        run_match(MatchConfig(pairs=3, max_plies=30), out_path=a, **common)
        s = run_match(MatchConfig(pairs=3, max_plies=30, workers=2), out_path=b, **common)
        # 配置不同（workers），哈希不同，但每一局应完全相同
        self.assertEqual(strip(read_games(a)[1]), strip(read_games(b)[1]))
        self.assertEqual(s["games"], 6)
        self.assertGreater(s["batch"]["forwards"], 0)

    def test_worker_error_stops(self):
        from unichess_kit.runtime import WorkerError
        with self.assertRaisesRegex(WorkerError, "故意失败"):
            run_match(MatchConfig(pairs=2, max_plies=30, workers=2),
                      spec_a=self.spec("make_failing_player_factory", fail_after=1),
                      spec_b=self.spec("make_random_player_factory"))


class TestCli(Base):
    def test_cli(self):
        from unichess_kit.pipelines.match import main
        conf = {"a": {"factory": f"{KIT}:make_fake_player_factory", "kwargs": {"name": "a"},
                      "label": "fake-a"},
                "b": {"factory": f"{KIT}:make_random_player_factory", "label": "rand"},
                "match": {"pairs": 1, "max_plies": 20}}
        cp = self.dir / "c.json"
        cp.write_text(json.dumps(conf), encoding="utf-8")
        self.assertEqual(main([str(cp), "--out", str(self.dir / "o.jsonl"), "--quiet"]), 0)
        header, games = read_games(self.dir / "o.jsonl")
        self.assertEqual(header["names"], {"A": "fake-a", "B": "rand"})
        self.assertEqual(len(games), 2)
        self.assertIn("kit_version", header["provenance"])


class TestSummary(unittest.TestCase):
    def test_pentanomial_only_complete_pairs(self):
        recs = [
            {"game": 0, "pair": 0, "a_score": 1.0, "white": "A", "termination": "checkmate",
             "opening": [], "moves": ["e2e4"], "plies": 1, "sources": {"A": {}, "B": {}}},
            {"game": 1, "pair": 0, "a_score": 0.5, "white": "B", "termination": "threefold",
             "opening": [], "moves": ["d2d4"], "plies": 1, "sources": {"A": {}, "B": {}}},
            {"game": 2, "pair": 1, "a_score": 0.0, "white": "A", "termination": "truncated",
             "opening": [], "moves": ["c2c4"], "plies": 1, "sources": {"A": {}, "B": {}}},
        ]
        s = summarize(recs, MatchConfig(pairs=2), {"A": "a", "B": "b"})
        self.assertEqual(s["pentanomial"], [0, 0, 0, 1, 0])
        self.assertEqual((s["a_wins"], s["b_wins"], s["draws"]), (1, 1, 1))
        self.assertEqual(s["by_color"]["a_as_white"], {"games": 2, "score": 1.0})
        self.assertAlmostEqual(s["truncated_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
