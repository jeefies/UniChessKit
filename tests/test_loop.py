"""换代循环（``pipelines.loop``）测试：占位符、状态机、闸门、窗口累积分片。

只测 **Loop 自身不跑子进程** 的逻辑——三阶段各自以子进程运行（显存随进程释放），
那部分由小规模冒烟覆盖；这里钉死的是「哪一代替换什么、什么时候换代、按哪几分片训练」：

- ``_subst`` 的三种替换（整体等于占位符 ⇒ 可为列表；子串替换；递归进 dict/list）；
- ``mapping`` 的 ``{weights}`` / ``{candidate}`` / ``{gen}`` / ``{gen_dir}`` /
  ``{selfplay_dir}`` / ``{selfplay_files}``（window 跨代累积分片，含生成前不存在目录的情形）；
- 配置校验：未知字段、gate 取值、``gate=sprt`` 必须有 ``match.sprt``；
- ``promote``：``score`` 门槛与 ``sprt`` 判决只认 H1；
- 状态落盘：``load_state`` 的默认值与写入后的读回（中断续跑的落点）。
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from Kit.pipelines.loop import Loop, _subst


def _conf(tmp: Path, **kw) -> dict:
    conf = {
        "out": str(tmp / "loop"),
        "generations": 3,
        "initial": "/w/champion.pt",
        "games": 10,
        "window": 2,
        "engine": {"factory": "No.such:factory",
                   "kwargs": {"checkpoint": "{weights}", "simulations": 800}},
        "selfplay": {"games": 0, "first_game": 0, "concurrency": 4},
        "sink": {"factory": "No.such:Sink", "kwargs": {"path": "{selfplay_dir}/s.sp.bin"}},
        "train": {"task": {"factory": "No.such:make_task", "kwargs": {"base": "{weights}"}},
                  "steps": 5, "device": "cpu",
                  "optimizer": {"lr": 5e-06, "weight_decay": 0.0001, "betas": [0.9, 0.999]}},
        "export": "final.pt",
        "arena": {"match": {"pairs": 4, "simulations": 2400,
                            "sprt": {"elo0": 0.0, "elo1": 40.0, "alpha": 0.05, "beta": 0.1}},
                  "gate": {"kind": "sprt"}},
    }
    conf.update(kw)
    return conf


def _write_spard(path: Path, name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    p = path / name
    p.write_bytes(b"\0")
    return str(p)


class TestSubst(unittest.TestCase):
    def test_whole_value_and_substring(self):
        m = {"{a}": [1, 2], "{b}": "/x/y", "{c}": 7}
        self.assertEqual(_subst("{a}", m), [1, 2])              # 整体相等 ⇒ 原类型返回
        self.assertEqual(_subst("/x/y/{c}", m), "/x/y/7")        # 子串替换
        self.assertEqual(_subst(["{a}", {"k": "{b}"}], m), [[1, 2], {"k": "/x/y"}])
        self.assertEqual(_subst({"n": 3, "s": "no-marker"}, m), {"n": 3, "s": "no-marker"})
        self.assertEqual(_subst(42, m), 42)


class TestLoopConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            Loop(_conf(self.tmp, bogus=1), self.tmp)

    def test_gate_kind_and_sprt_requirement(self):
        with self.assertRaises(ValueError):
            Loop(_conf(self.tmp, arena={"gate": {"kind": "nope"}}), self.tmp)
        bad = _conf(self.tmp)
        bad["arena"] = {"match": {"pairs": 4}, "gate": {"kind": "sprt"}}   # 缺 match.sprt
        with self.assertRaises(ValueError):
            Loop(bad, self.tmp)
        ok = _conf(self.tmp)
        ok["arena"] = {"match": {"pairs": 4}, "gate": {"kind": "score", "min_score": 0.55}}
        Loop(ok, self.tmp)                                     # score 闸门不需要 sprt


class TestLoopMapping(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))
        self.loop = Loop(_conf(self.tmp, window=2), self.tmp)

    def test_placeholders(self):
        m = self.loop.mapping(3, "/w/champ.pt")
        self.assertEqual(m["{weights}"], "/w/champ.pt")
        self.assertEqual(m["{gen}"], 3)
        self.assertEqual(m["{gen_dir}"], str(self.tmp / "loop" / "gen_0003"))
        self.assertEqual(m["{selfplay_dir}"], str(self.tmp / "loop" / "gen_0003" / "selfplay"))
        self.assertEqual(m["{candidate}"],
                         str(self.tmp / "loop" / "gen_0003" / "train" / "final.pt"))
        self.assertEqual(m["{selfplay_files}"], [])            # 还没生成过自对弈

    def test_window_accumulates_recent_generations(self):
        base = self.tmp / "loop"
        _write_spard(base / "gen_0000" / "selfplay", "a.sp.bin")
        _write_spard(base / "gen_0001" / "selfplay", "b.sp.bin")
        _write_spard(base / "gen_0002" / "selfplay", "c.sp.bin")
        # window=2 = 本代 + 上一代：第 2 代带 gen_0001/0002，gen_0000 已滑出窗口
        files = self.loop.mapping(2, "/w")["{selfplay_files}"]
        self.assertEqual([Path(f).name for f in files], ["b.sp.bin", "c.sp.bin"])
        self.assertEqual([Path(f).parent.parent.name for f in files],
                         ["gen_0001", "gen_0002"])
        self.assertEqual(self.loop.mapping(0, "/w")["{selfplay_files}"],
                         [str(base / "gen_0000" / "selfplay" / "a.sp.bin")])

    def test_mapping_survives_missing_dirs(self):
        self.assertEqual(self.loop.mapping(7, "/w")["{selfplay_files}"], [])


class TestPromote(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_score_gate(self):
        loop = Loop(_conf(self.tmp, arena={"match": {"pairs": 4},
                                           "gate": {"kind": "score", "min_score": 0.55}}), self.tmp)
        self.assertTrue(loop.promote({"score_a": 0.6}))
        self.assertFalse(loop.promote({"score_a": 0.5}))
        self.assertFalse(loop.promote({"score_a": None}))       # 没有分数 ⇒ 不换代

    def test_sprt_gate_only_accepts_h1(self):
        loop = Loop(_conf(self.tmp), self.tmp)
        self.assertTrue(loop.promote({"sprt": {"verdict": "H1"}}))
        for verdict in ("H0", "inconclusive"):
            self.assertFalse(loop.promote({"sprt": {"verdict": verdict}}))
        self.assertFalse(loop.promote({"sprt": None}))           # 没配 sprt ⇒ 不换代


def _two_variant_conf(tmp: Path) -> dict:
    """只含两个已备好产物的变体，用于验证续跑跳过。"""
    conf = _conf(tmp)
    train = conf["train"]
    train["variants"] = [{"label": "a", "optimizer": {"lr": 1e-05}},
                         {"label": "b", "optimizer": {"lr": 2e-05}}]
    train["screen"] = {"pairs": 4, "seed": 5, "max_plies": 60, "concurrency": 2}
    return conf


class TestLoopState(unittest.TestCase):
    def test_default_and_roundtrip(self):
        tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))
        loop = Loop(_conf(tmp), tmp)
        st = loop.load_state()
        self.assertEqual((st["generation"], st["phase"], st["champion"]),
                         (0, "selfplay", "/w/champion.pt"))
        self.assertEqual(st["history"], [])
        loop.out.mkdir(parents=True, exist_ok=True)
        (loop.out / "loop_state.json").write_text(
            json.dumps({"generation": 2, "phase": "arena", "champion": "/w/gen1.pt",
                        "history": [{"generation": 0, "promoted": True}]}), encoding="utf-8")
        st = loop.load_state()
        self.assertEqual((st["generation"], st["phase"], st["champion"]),
                         (2, "arena", "/w/gen1.pt"))
        self.assertEqual(len(st["history"]), 1)


def _variant_conf(tmp: Path, **kw) -> dict:
    """带枚举搜索的 loop 配置（3 个变体）。"""
    conf = _conf(tmp)
    train = conf["train"]
    train["variants"] = [{"label": "lr1e-5", "optimizer": {"lr": 1e-05}, "steps": 800},
                         {"label": "lr5e-6", "optimizer": {"lr": 5e-06}},
                         {"label": "lr1e-6", "optimizer": {"lr": 1e-06}, "steps": 1500}]
    train["screen"] = {"pairs": 4, "seed": 5, "max_plies": 60, "concurrency": 2}
    conf.update(kw)
    return conf


class TestVariantSearchConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_variants_require_labels_and_screen(self):
        with self.assertRaises(ValueError):
            Loop(_variant_conf(self.tmp, train={"task": {"factory": "x:y"}, "steps": 1,
                                                "variants": [{"optimizer": {"lr": 1}}]}),
                self.tmp)
        dup = _variant_conf(self.tmp)
        dup["train"]["variants"][1]["label"] = "lr1e-5"
        with self.assertRaises(ValueError):
            Loop(dup, self.tmp)
        no_screen = _variant_conf(self.tmp)
        no_screen["train"].pop("screen")
        with self.assertRaises(ValueError):
            Loop(no_screen, self.tmp)

    def test_merge_is_recursive_and_leaves_base_intact(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        m = loop.mapping(0, "/w/champ.pt")
        base = copy.deepcopy(loop.conf["train"])
        got = loop._variant_train_conf(loop.variants[0], m)
        self.assertEqual(got["steps"], 800)                       # variant 覆盖
        self.assertEqual(got["optimizer"]["lr"], 1e-05)           # 递归合并
        self.assertEqual(got["optimizer"]["weight_decay"], 0.0001)  # 未覆盖的保留
        self.assertEqual(got["optimizer"]["betas"], [0.9, 0.999])   # 无关键不动
        self.assertEqual(got["out"], str(loop.gen_dir(0) / "train_lr1e-5"))
        self.assertNotIn("label", got)
        self.assertNotIn("variants", got)
        self.assertNotIn("screen", got)
        self.assertEqual(loop.conf["train"], base)                # 原配置未被改动

    def test_variant_placeholders_substituted(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        m = loop.mapping(2, "/w/champ.pt")
        got = loop._variant_train_conf(loop.variants[1], m)
        self.assertEqual(got["task"]["kwargs"]["base"], "{weights}".replace("{weights}",
                                                                            "/w/champ.pt"))
        self.assertEqual(got["steps"], 5)                         # 模板默认步数
        self.assertEqual(got["optimizer"]["lr"], 5e-06)

    def test_screen_conf_is_candidate_vs_champion(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        m = loop.mapping(1, "/w/champ.pt")
        c = loop._screen_conf(m, "lr5e-6")
        cand = str(loop.gen_dir(1) / "train_lr5e-6" / "final.pt")
        self.assertEqual(c["a"]["kwargs"]["checkpoint"], cand)
        self.assertEqual(c["a"]["label"], "lr5e-6")
        self.assertEqual(c["b"]["kwargs"]["checkpoint"], "/w/champ.pt")
        self.assertEqual(c["b"]["label"], "champion")
        self.assertEqual(c["match"]["pairs"], 4)
        self.assertEqual(c["match"]["seed"], 5)

    def test_search_picks_highest_score(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = loop.gen_dir(0)
        gd.mkdir(parents=True, exist_ok=True)
        (gd / "search.json").write_text(json.dumps({
            "lr1e-5": {"label": "lr1e-5", "score_a": 0.52, "elo": 5.0, "games": 8},
            "lr5e-6": {"label": "lr5e-6", "score_a": 0.62, "elo": 88.0, "games": 8},
            "lr1e-6": {"label": "lr1e-6", "score_a": 0.55, "elo": 30.0, "games": 8},
            "_selected": "lr1e-6"}), encoding="utf-8")
        self.assertEqual(loop._search_all(0, loop.mapping(0, "/w/c.pt")), "lr5e-6")
        self.assertEqual(loop._search_results(0)["_selected"], "lr5e-6")

    def test_search_tie_breaks_by_elo_then_games(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = loop.gen_dir(0)
        gd.mkdir(parents=True, exist_ok=True)
        # 三个变体都要有结果，否则 _search_all 会去真跑缺的那个
        (gd / "search.json").write_text(json.dumps({
            "lr1e-5": {"label": "lr1e-5", "score_a": 0.60, "elo": 10.0, "games": 8},
            "lr5e-6": {"label": "lr5e-6", "score_a": 0.60, "elo": 40.0, "games": 8},
            "lr1e-6": {"label": "lr1e-6", "score_a": 0.60, "elo": 40.0, "games": 12}}),
            encoding="utf-8")
        self.assertEqual(loop._search_all(0, loop.mapping(0, "/w/c.pt")), "lr1e-6")

    def test_search_resume_skips_finished_variants(self):
        """训练导出 + 筛选赛汇总都在的变体必须跳过（不重跑 GPU）。"""
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = loop.gen_dir(0)
        gd.mkdir(parents=True, exist_ok=True)
        for label in ("lr1e-5", "lr5e-6"):
            (gd / f"train_{label}").mkdir(parents=True, exist_ok=True)
            (gd / f"train_{label}" / "final.pt").write_bytes(b"\0")
            (gd / f"screen_{label}.jsonl.summary.json").write_text(
                json.dumps({"games": 8, "score_a": 0.5 + 0.01 * len(label),
                            "elo": len(label)}), encoding="utf-8")
        # 第三个变体缺产物：真跑会失败，所以这里只验证前两个被读回而不是重跑
        res = loop._search_results(0)
        self.assertEqual(res, {})                     # 还没写 search.json
        loop._search_all.__self__  # noqa: B018  (仅确认方法在)
        # 手工模拟：只给前两个 variant 的配置，验证读取逻辑
        two = Loop(_variant_conf(self.tmp, train=None) if False else
                   _two_variant_conf(self.tmp), self.tmp)
        two_gd = two.gen_dir(0)
        for label in ("a", "b"):
            (two_gd / f"train_{label}").mkdir(parents=True, exist_ok=True)
            (two_gd / f"train_{label}" / "final.pt").write_bytes(b"\0")
            (two_gd / f"screen_{label}.jsonl.summary.json").write_text(
                json.dumps({"games": 8, "score_a": 0.6 if label == "b" else 0.4,
                            "elo": 3.0}), encoding="utf-8")
        self.assertEqual(two._search_all(0, two.mapping(0, "/w/c.pt")), "b")

    def test_search_results_missing_is_empty(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        self.assertEqual(loop._search_results(0), {})

    def test_no_variants_uses_single_train(self):
        loop = Loop(_conf(self.tmp), self.tmp)
        self.assertEqual(loop.variants, [])


class TestEnumerateGenerations(unittest.TestCase):
    """只枚举前 N 代：uses_search 的判定、锁定配置的构造与读取。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_default_is_always_enumerate(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        self.assertIsNone(loop.enumerate_generations)
        self.assertTrue(all(loop.uses_search(g) for g in range(6)))

    def test_first_n_only(self):
        loop = Loop(_variant_conf(self.tmp, enumerate_generations=3), self.tmp)
        self.assertEqual([loop.uses_search(g) for g in range(5)],
                         [True, True, True, False, False])
        loop0 = Loop(_variant_conf(self.tmp, enumerate_generations=0), self.tmp)
        self.assertFalse(loop0.uses_search(0))
        with self.assertRaises(ValueError):
            Loop(_variant_conf(self.tmp, enumerate_generations=-1), self.tmp)

    def test_fixed_conf_uses_locked_overrides(self):
        loop = Loop(_variant_conf(self.tmp, enumerate_generations=3), self.tmp)
        m = loop.mapping(4, "/w/champ.pt")
        got = loop._fixed_train_conf(m, {"optimizer": {"lr": 1e-06}, "steps": 600})
        self.assertEqual(got["steps"], 600)                       # 锁定配置生效
        self.assertEqual(got["optimizer"]["lr"], 1e-06)
        self.assertEqual(got["optimizer"]["weight_decay"], 0.0001)  # 未覆盖的仍在
        self.assertEqual(got["out"], str(loop.gen_dir(4) / "train"))  # 不是 train_<label>
        self.assertNotIn("variants", got)
        self.assertNotIn("screen", got)
        self.assertEqual(got["task"]["kwargs"]["base"], "/w/champ.pt")

    def test_selected_variant_config_roundtrip(self):
        loop = Loop(_variant_conf(self.tmp, enumerate_generations=3), self.tmp)
        gd = loop.gen_dir(0)
        gd.mkdir(parents=True, exist_ok=True)
        (gd / "search.json").write_text(json.dumps({
            "lr1e-6_s600": {"label": "lr1e-6_s600",
                            "config": {"optimizer": {"lr": 1e-06}, "steps": 600},
                            "score_a": 0.4},
            "lr1e-5_s600": {"label": "lr1e-5_s600",
                            "config": {"optimizer": {"lr": 1e-05}, "steps": 600},
                            "score_a": 0.62},
            "lr1e-5_s1200": {"label": "lr1e-5_s1200",
                             "config": {"optimizer": {"lr": 1e-05}, "steps": 1200},
                             "score_a": 0.55},
            "_selected": "lr1e-5_s600"}), encoding="utf-8")
        self.assertEqual(loop._selected_variant_config(0),
                         {"optimizer": {"lr": 1e-05}, "steps": 600})
        self.assertEqual(loop._selected_variant_config(1), {})     # 没跑过搜索的代

    def test_base_train_strips_search_only_keys(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        base = loop._base_train()
        self.assertNotIn("variants", base)
        self.assertNotIn("screen", base)
        self.assertEqual(base["steps"], 5)                        # 模板本身没被改
        self.assertIn("variants", loop.conf["train"])


if __name__ == "__main__":
    unittest.main()
