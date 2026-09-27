import json
import shutil
import tempfile
import unittest
from pathlib import Path

import chess

from Kit.api import PlayerError, immediate
from Kit.api.types import MoveDecision
from Kit.pipelines.match import (MatchConfig, SprtConfig, plan_games, run_match,
                                          summarize)
from Kit.players import RandomPlayer
from Kit.registry import EngineSpec
from Kit.testing.fakes import (make_failing_player_factory, make_fake_player_factory,
                                        make_random_player_factory)

KIT = "Kit.testing.fakes"


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
        from Kit.pipelines.match import load_book
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

    def test_max_plies_after_opening(self):
        from Kit.pipelines.match import load_book
        whole = MatchConfig(pairs=3, max_plies=10)
        after = MatchConfig(pairs=3, max_plies=10, max_plies_after_opening=True)
        for t in plan_games(after, load_book("bundled")):
            self.assertEqual(whole.referee(t).max_plies, 10)
            self.assertEqual(after.referee(t).max_plies, 10 + len(t.opening))
        # 默认值不进 to_dict：既有结果文件的配置哈希保持不变
        self.assertNotIn("max_plies_after_opening", whole.to_dict())
        self.assertTrue(after.to_dict()["max_plies_after_opening"])
        self.assertEqual(MatchConfig.from_dict(after.to_dict()), after)


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

    def test_truncation_counts_only_after_opening(self):
        s = self.run_fake(pairs=2, max_plies_after_opening=True)
        self.assertGreater(s["games"], 0)
        out = self.dir / "t.jsonl"
        cfg = MatchConfig(pairs=2, max_plies=3, concurrency=2, max_plies_after_opening=True)
        run_match(cfg, make_a=make_random_player_factory("a"), make_b=make_random_player_factory("b"),
                  out_path=out)
        _, games = read_games(out)
        for r in games:
            self.assertEqual(r["termination"], "truncated")
            self.assertEqual(len(r["moves"]), 3)
            self.assertEqual(r["plies"], len(r["opening"]) + 3)

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

    def test_sprt_verdict_is_frozen_at_boundary(self):
        """越界即结论：停止后又落盘的在途对局不得把 verdict 抹成 None。

        2026-09-27 loop_p4_v2 gen 0 实测：``workers=2`` 时第 85 局（40 对）llr=+3.164
        越过 H1 上界 2.890 触发停止，剩下几局在途对局补到 101 局后 llr 回落到 +1.471。
        旧代码拿**最终记录集**重算 verdict，得到 None——一次本该换代的判定变成"没换代"。
        workers=1 时没有在途对局，所以这个 bug 一直没暴露。

        序列（每对给 a 方总分）：8 对 A 全胜 -> 86 对一胜一负。前者把 llr 推过 H1
        上界触发停止，后者把最终 llr 拉回 -0.082（界内），即"该停且该判 H1，
        但重算已无判决"。
        """
        from Kit.pipelines.match import _Run, _sprt_decided
        cfg = MatchConfig(pairs=400, max_plies=80, sprt=SprtConfig(elo0=0, elo1=60,
                                                                  min_pairs=8))
        pairs = [2.0] * 8 + [1.0] * 86
        run = _Run(cfg, {"A": "a", "B": "b"}, None, None)
        i = 0
        for p, total in enumerate(pairs):
            for g, sc in enumerate((total / 2.0, total / 2.0)):
                run.add({"game": i, "pair": p, "a_score": sc,
                         "termination": "checkmate", "plies": 40,
                         "white": "A" if g == 0 else "B", "opening": [f"o{i}"],
                         "moves": [f"m{i}"], "sources": {"A": {"search": 1, "tablebase": 0},
                                                         "B": {"search": 1, "tablebase": 0}}})
                i += 1
        # 停止已触发，且判决冻结为 H1
        self.assertTrue(run.stopped)
        self.assertEqual(run.stopped_verdict, "H1")
        # 而拿最终记录集重算已经退回界内（这正是旧代码出错的地方）
        decided, verdict = _sprt_decided(run.records, cfg)
        self.assertFalse(decided)
        self.assertIsNone(verdict)
        llr = summarize(run.records, cfg, {})["sprt"]["llr"]
        self.assertTrue(-2.251 < llr < 2.890, f"最终 llr={llr} 应落在界内")

    def test_sprt_stopped_always_has_a_verdict(self):
        """不变式：只要停了，就必须带着一个非 None 的判决（否则 promotion 无从判断）。"""
        from Kit.pipelines.match import _Run, _sprt_decided
        cfg = MatchConfig(pairs=60, max_plies=80, sprt=SprtConfig(elo0=0, elo1=60,
                                                                 min_pairs=4))
        run = _Run(cfg, {"A": "a", "B": "b"}, None, None)
        for i in range(30):
            run.add({"game": i, "pair": i // 2, "a_score": 1.0,
                     "termination": "checkmate", "plies": 40,
                     "white": "A" if i % 2 == 0 else "B", "opening": [f"o{i}"],
                     "moves": [f"m{i}"], "sources": {"A": {"search": 1, "tablebase": 0},
                                                     "B": {"search": 1, "tablebase": 0}}})
        self.assertTrue(run.stopped)
        self.assertIn(run.stopped_verdict, ("H0", "H1"))

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
        from Kit.runtime import WorkerError
        with self.assertRaisesRegex(WorkerError, "故意失败"):
            run_match(MatchConfig(pairs=2, max_plies=30, workers=2),
                      spec_a=self.spec("make_failing_player_factory", fail_after=1),
                      spec_b=self.spec("make_random_player_factory"))


class TestCli(Base):
    def test_cli(self):
        from Kit.pipelines.match import main
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
