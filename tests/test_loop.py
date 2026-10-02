"""换代循环（``pipelines.loop``）测试：占位符、状态机、闸门、窗口累积分片。

只测 **Loop 自身不跑子进程** 的逻辑——三阶段各自以子进程运行（显存随进程释放），
那部分由小规模冒烟覆盖；这里钉死的是「哪一代替换什么、什么时候换代、按哪几分片训练」：

- ``_subst`` 的三种替换（整体等于占位符 ⇒ 可为列表；子串替换；递归进 dict/list）；
- ``mapping`` 的 ``{weights}`` / ``{candidate}`` / ``{gen}`` / ``{gen_dir}`` /
  ``{selfplay_dir}`` / ``{selfplay_files}``（window 跨代累积分片，含生成前不存在目录的情形）；
- 配置校验：未知字段、gate 取值、``gate=sprt`` 必须有 ``match.sprt``；
- ``promote``：``score`` 门槛与 ``sprt`` 判决只认 H1；
- 状态落盘：``load_state`` 的默认值与写入后的读回（中断续跑的落点）；
- 配置体检：拼错的占位符、``train.screen`` 与 ``arena.match`` 开局库不一致、
  锁定配置来自已不在网格里的变体、``search.json`` 混着旧网格的 label。
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

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


class _Stop(Exception):
    """用来中断 ``run()`` 的哨兵。"""


class TestArenaResumeCandidate(unittest.TestCase):
    """中断后从 arena 阶段续跑：候选权重必须跟着 search 代的胜者走。

    2026-09-27 实测踩到：在 arena 阶段重启 loop，``mapping`` 给的 ``{candidate}`` 是
    ``gen_XXXX/train/final.pt``（``phase_train`` 的产物路径），而 search 代的胜者在
    ``gen_XXXX/train_<label>/`` 下，于是 ``FileNotFoundError``。只有正常往下走的
    search 分支会覆盖 ``{candidate}``，续跑路径没人覆盖。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_candidate_path_follows_search_winner(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        self.assertEqual(loop.candidate_path(0, "lr2e-5_s1200"),
                         str(loop.gen_dir(0) / "train_lr2e-5_s1200" / "final.pt"))

    def test_candidate_path_without_variant_is_train_dir(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        self.assertEqual(loop.candidate_path(0, None),
                         str(loop.gen_dir(0) / "train" / "final.pt"))
        m = loop.mapping(0, "/w/c.pt")
        self.assertEqual(m["{candidate}"], str(loop.gen_dir(0) / "train" / "final.pt"))

    def test_candidate_path_after_enumerate_exhausted(self):
        """``enumerate_generations`` 用尽的代不再 search，胜者路径让位给 ``train/``。"""
        loop = Loop(_variant_conf(self.tmp, enumerate_generations=1), self.tmp)
        self.assertFalse(loop.uses_search(1))
        self.assertEqual(loop.candidate_path(1, "lr2e-5_s1200"),
                         str(loop.gen_dir(1) / "train" / "final.pt"))

    def test_resume_in_arena_uses_search_winner(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = loop.gen_dir(0)
        gd.mkdir(parents=True, exist_ok=True)
        (gd / "train_lr2e-5_s1200").mkdir()
        (gd / "train_lr2e-5_s1200" / "final.pt").write_bytes(b"\0")
        (gd / "search.json").write_text(json.dumps({
            "lr2e-5_s1200": {"label": "lr2e-5_s1200", "config": {"steps": 1200},
                             "score_a": 0.673, "elo": 125.5, "games": 385},
            "_selected": "lr2e-5_s1200"}), encoding="utf-8")
        loop.state_path.write_text(json.dumps({
            "generation": 0, "phase": "arena", "champion": "/w/c.pt",
            "history": [], "variant": "lr2e-5_s1200"}), encoding="utf-8")
        seen = {}

        def fake_arena(self, g, m):
            seen["g"], seen["cand"] = g, m["{candidate}"]
            raise _Stop

        with mock.patch.object(Loop, "phase_arena", fake_arena):
            with self.assertRaises(_Stop):
                loop.run()
        self.assertEqual(seen["g"], 0)
        self.assertEqual(seen["cand"], str(gd / "train_lr2e-5_s1200" / "final.pt"))
        self.assertTrue((gd / "train_lr2e-5_s1200" / "final.pt").exists())

    def test_resume_in_arena_falls_back_to_search_json(self):
        """状态文件没写 ``variant`` 时退回 ``search.json`` 的 ``_selected``。

        ``98b2798`` 只认 ``st["variant"]``；老版本（或手改过）的状态文件没有这个字段，
        续跑又跌回 ``gen_XXXX/train/final.pt``。胜者的权威记录在 ``search.json`` 里，
        状态文件只是它的缓存。
        """
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = loop.gen_dir(0)
        (gd / "train_lr5e-6").mkdir(parents=True, exist_ok=True)
        (gd / "train_lr5e-6" / "final.pt").write_bytes(b"\0")
        (gd / "search.json").write_text(json.dumps({
            "lr1e-5": {"label": "lr1e-5", "config": {"steps": 800}, "score_a": 0.5},
            "lr5e-6": {"label": "lr5e-6", "config": {"steps": 1200}, "score_a": 0.67},
            "lr1e-6": {"label": "lr1e-6", "config": {"steps": 1500}, "score_a": 0.6},
            "_selected": "lr5e-6"}), encoding="utf-8")
        loop.state_path.write_text(json.dumps({
            "generation": 0, "phase": "arena", "champion": "/w/c.pt", "history": []}),
            encoding="utf-8")
        seen = {}

        def fake_arena(self, g, m):
            seen["cand"] = m["{candidate}"]
            raise _Stop

        with mock.patch.object(Loop, "phase_arena", fake_arena):
            with self.assertRaises(_Stop):
                loop.run()
        self.assertEqual(seen["cand"], str(gd / "train_lr5e-6" / "final.pt"))


class TestLockedTrainConfSchedule(unittest.TestCase):
    """枚举用尽后按锁定配置训练：``schedule`` 与变体训练一样必须**整体替换**。

    2026-09-27 loop_p4_v2：新网格的基座是 ``{"kind":"onecycle","pct_start":0.25}``，
    含一个代内对照 ``ctrl_const_2e-5_wd4``（``{"schedule": {"kind": "constant"}}``）。
    它一旦在 gen 0/1/2 里胜出，``train_variant`` 就带着这份 schedule 进 gen 3+；
    锁定那一侧只会普通合并，合出 ``{"kind": "constant", "pct_start": 0.25}``，
    子进程 ``Kit train`` 直接以「调度 constant 有未知参数 ['pct_start']」退出
    （那时候这一代的 9.6h 自对弈已经花掉了）。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def _onecycle_conf(self) -> dict:
        conf = _variant_conf(self.tmp)
        conf["train"]["schedule"] = {"kind": "onecycle", "pct_start": 0.25}
        conf["train"]["variants"] = [
            {"label": "ctrl_const", "optimizer": {"lr": 2e-05, "weight_decay": 1e-04},
             "schedule": {"kind": "constant"}},
            {"label": "1c_5e-5", "optimizer": {"lr": 5e-05, "weight_decay": 1e-05}},
        ]
        return conf

    def test_locked_schedule_is_replaced_not_merged(self):
        loop = Loop(self._onecycle_conf(), self.tmp)
        m = loop.mapping(3, "/w/champ.pt")
        got = loop._fixed_train_conf(m, {"optimizer": {"lr": 2e-05, "weight_decay": 1e-04},
                                        "schedule": {"kind": "constant"}})
        self.assertEqual(got["schedule"], {"kind": "constant"})     # 不带 pct_start
        self.assertEqual(got["optimizer"]["lr"], 2e-05)              # 锁定的 lr
        self.assertEqual(got["optimizer"]["weight_decay"], 1e-04)    # 锁定的 wd
        self.assertEqual(got["optimizer"]["betas"], [0.9, 0.999])    # 基座里的保留
        self.assertEqual(got["task"]["kwargs"]["base"], "/w/champ.pt")
        self.assertEqual(got["out"], str(loop.gen_dir(3) / "train"))
        self.assertNotIn("variants", got)
        self.assertNotIn("screen", got)
        self.assertEqual(loop.conf["train"]["schedule"], {"kind": "onecycle", "pct_start": 0.25})

    def test_locked_without_schedule_inherits_base(self):
        loop = Loop(self._onecycle_conf(), self.tmp)
        got = loop._fixed_train_conf(loop.mapping(3, "/w/c.pt"), {"optimizer": {"lr": 5e-05}})
        self.assertEqual(got["schedule"], {"kind": "onecycle", "pct_start": 0.25})

    def test_no_schedule_anywhere_leaves_key_absent(self):
        """两边都没有 schedule 时不写这个键（保持 TrainConfig 的缺省，配置哈希不变）。"""
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        self.assertNotIn("schedule", loop.conf["train"])
        got = loop._fixed_train_conf(loop.mapping(3, "/w/c.pt"), {"optimizer": {"lr": 1e-05}})
        self.assertNotIn("schedule", got)

    def test_variant_and_locked_sides_agree(self):
        """同一个 variant 在枚举代与锁定代吃到的 schedule 必须一致。"""
        loop = Loop(self._onecycle_conf(), self.tmp)
        m = loop.mapping(0, "/w/champ.pt")
        variant = loop.variants[0]                                  # ctrl_const
        self.assertEqual(loop._variant_train_conf(variant, m)["schedule"], {"kind": "constant"})
        self.assertEqual(loop._fixed_train_conf(m, {"schedule": {"kind": "constant"}})["schedule"],
                         {"kind": "constant"})

    def test_locked_generation_writes_replaced_schedule_to_disk(self):
        """run() 的 train 段落盘的 ``train.json``（子进程真正吃的）不能带 pct_start。"""
        conf = self._onecycle_conf()
        conf["generations"] = 6                      # state 停在 gen 3，得让它进循环体
        loop = Loop(conf, self.tmp)
        gd = loop.gen_dir(3)
        (gd / "train").mkdir(parents=True)
        (gd / "train" / "final.pt").write_bytes(b"\0")
        loop.state_path.write_text(json.dumps({
            "generation": 3, "phase": "train", "champion": "/w/c.pt", "history": [],
            "variant": "ctrl_const",
            "train_variant": {"optimizer": {"lr": 2e-05, "weight_decay": 1e-04},
                              "schedule": {"kind": "constant"}}}), encoding="utf-8")
        seen = {}

        def fake_run(self, subcmd, cfg_path, log_path, extra=()):
            seen["subcmd"], seen["conf"] = subcmd, json.loads(cfg_path.read_text(encoding="utf-8"))
            raise _Stop

        with mock.patch.object(Loop, "_run", fake_run):
            with self.assertRaises(_Stop):
                loop.run()
        self.assertEqual(seen["subcmd"], "train")
        self.assertEqual(seen["conf"]["schedule"], {"kind": "constant"})
        self.assertEqual(seen["conf"]["optimizer"]["lr"], 2e-05)
        self.assertEqual(seen["conf"]["out"], str(gd / "train"))


class TestStaleGridResults(unittest.TestCase):
    """``search.json`` 里混着旧网格的 label：必须丢掉重跑，不能把 loop 永久卡死。

    2026-09-27 loop_p4_v2：gen 0 的 ``search.json`` 是旧网格（10 个恒定 lr 变体）写的，
    换成 8 变体新网格后重启 loop。旧 label 不在新的 ``train.variants`` 里 → 新 label 全都要
    重跑（1.5h GPU）→ ``len(results)`` = 10 + 8 = 18 ≠ 8 → 「变体结果不齐」报错；而旧 label
    还留在文件里，之后每次重启都撞同一个错，只能手工删 ``search.json``（已跑出的新变体
    成绩一并丢掉）。网格换过了，旧 label 的成绩对新网格没有意义，扔掉才对。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def _with_all_current_variants(self, loop: Loop) -> Path:
        gd = loop.gen_dir(0)
        gd.mkdir(parents=True, exist_ok=True)
        for label, score in (("lr1e-5", 0.55), ("lr5e-6", 0.61), ("lr1e-6", 0.50)):
            (gd / f"train_{label}").mkdir(exist_ok=True)
            (gd / f"train_{label}" / "final.pt").write_bytes(b"\0")
            (gd / f"screen_{label}.jsonl.summary.json").write_text(
                json.dumps({"games": 8, "score_a": score, "elo": 10.0}), encoding="utf-8")
        return gd

    def test_stale_labels_are_dropped_and_new_grid_completes(self):
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = self._with_all_current_variants(loop)
        (gd / "search.json").write_text(json.dumps({
            "old_2e-5": {"label": "old_2e-5", "config": {"optimizer": {"lr": 2e-05}},
                         "score_a": 0.9, "elo": 200.0, "games": 8},
            "old_1e-5": {"label": "old_1e-5", "config": {"optimizer": {"lr": 1e-05}},
                         "score_a": 0.8, "elo": 150.0, "games": 8}}), encoding="utf-8")
        with self.assertWarns(UserWarning):
            best = loop._search_all(0, loop.mapping(0, "/w/c.pt"))
        self.assertEqual(best, "lr5e-6")                # 旧 label 的 0.9 不作数
        res = loop._search_results(0)
        self.assertEqual(sorted(k for k in res if not k.startswith("_")),
                         ["lr1e-5", "lr1e-6", "lr5e-6"])
        self.assertEqual(res["_selected"], "lr5e-6")

    def test_same_grid_results_are_kept(self):
        """没换网格时一条都不丢（续跑不能白跑已花的 GPU）。"""
        loop = Loop(_variant_conf(self.tmp), self.tmp)
        gd = self._with_all_current_variants(loop)
        (gd / "search.json").write_text(json.dumps({
            "lr1e-5": {"label": "lr1e-5", "config": {"steps": 800}, "score_a": 0.55,
                       "elo": 10.0, "games": 8}}), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            best = loop._search_all(0, loop.mapping(0, "/w/c.pt"))
        self.assertEqual(best, "lr5e-6")
        self.assertIn("lr1e-5", loop._search_results(0))


class TestConfigSanityWarnings(unittest.TestCase):
    """启动时的配置体检：宁可在日志里喊一句，也别静默跑一场不相干的筛选用例。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_screen_and_arena_openings_must_match(self):
        """``MatchConfig.openings`` 缺省 bundled：一边写合成库一边不写，筛选赛就用错开局库。

        2026-09-27 loop_p4_v2 gen 0 的实况：旧进程按内存里的旧配置写出
        ``openings: "bundled"``，十场筛选赛跑在 34 条开局上（duplicate_rate 0.27~0.375），
        arena 用合成库——两边开局分布不同，筛选等于白跑。
        """
        only_arena = _variant_conf(self.tmp)
        only_arena["arena"]["match"]["openings"] = "/x/synth.txt"
        with self.assertWarns(UserWarning) as cm:
            Loop(only_arena, self.tmp)
        self.assertIn("openings", str(cm.warning))
        only_screen = _variant_conf(self.tmp)
        only_screen["train"]["screen"]["openings"] = "/x/synth.txt"
        with self.assertWarns(UserWarning) as cm:
            Loop(only_screen, self.tmp)
        self.assertIn("openings", str(cm.warning))
        both = _variant_conf(self.tmp)
        both["arena"]["match"]["openings"] = "/x/synth.txt"
        both["train"]["screen"]["openings"] = "/x/synth.txt"
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # 一致就不该有任何告警
            Loop(both, self.tmp)
        neither = _variant_conf(self.tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Loop(neither, self.tmp)

    def test_misspelled_placeholder_is_reported(self):
        """拼错的占位符不会被替换，会原样带进子进程配置里，直到要写文件才炸。"""
        conf = _variant_conf(self.tmp)
        conf["sink"]["kwargs"]["path"] = "{selfplay_dirr}/s.sp.bin"
        with self.assertWarns(UserWarning) as cm:
            Loop(conf, self.tmp)
        self.assertIn("{selfplay_dirr}", str(cm.warning))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Loop(_variant_conf(self.tmp), self.tmp)   # 正确拼写不告警
            Loop(_conf(self.tmp), self.tmp)           # 无 variants 的配置也不告警

    def test_warns_when_locked_variant_left_the_grid(self):
        """锁定配置来自已不在 ``train.variants`` 里的变体（换过网格）时要喊出来。

        2026-09-27：gen 0 存的 ``train_variant`` 是旧恒定 lr 网格的胜者
        ``{lr 2e-5, steps 1200}``，换新网格后 gen 3+ 会**静默**按它训练，历史记录里
        看不出配置已经换过代。
        """
        loop = Loop(_variant_conf(self.tmp, enumerate_generations=1), self.tmp)
        loop.out.mkdir(parents=True, exist_ok=True)
        (loop.gen_dir(1) / "train").mkdir(parents=True)
        (loop.gen_dir(1) / "train" / "final.pt").write_bytes(b"\0")
        loop.state_path.write_text(json.dumps({
            "generation": 1, "phase": "train", "champion": "/w/c.pt", "history": [],
            "variant": "old_grid_2e-5",
            "train_variant": {"optimizer": {"lr": 2e-05}, "steps": 1200}}),
            encoding="utf-8")

        def fake_train(self, g, m, overrides=None):
            seen["overrides"] = overrides
            raise _Stop

        seen = {}
        with mock.patch.object(Loop, "phase_train", fake_train):
            with self.assertWarns(UserWarning) as cm:     # 喊出来：网格换过了
                with self.assertRaises(_Stop):
                    loop.run()
        self.assertIn("old_grid_2e-5", str(cm.warning))
        self.assertEqual(seen["overrides"], {"optimizer": {"lr": 2e-05}, "steps": 1200})

    def test_no_warning_when_locked_variant_still_in_grid(self):
        loop = Loop(_variant_conf(self.tmp, enumerate_generations=1), self.tmp)
        loop.out.mkdir(parents=True, exist_ok=True)
        (loop.gen_dir(1) / "train").mkdir(parents=True)
        (loop.gen_dir(1) / "train" / "final.pt").write_bytes(b"\0")
        loop.state_path.write_text(json.dumps({
            "generation": 1, "phase": "train", "champion": "/w/c.pt", "history": [],
            "variant": "lr5e-6",
            "train_variant": {"optimizer": {"lr": 5e-06}, "steps": 1200}}),
            encoding="utf-8")
        with mock.patch.object(Loop, "phase_train", lambda *a, **kw: (_ for _ in ()).throw(_Stop)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with self.assertRaises(_Stop):
                    loop.run()


class TestPerGenerationSeeds(unittest.TestCase):
    """``seed_base``：每代重新掷种子；不配则完全维持原行为。

    Transformer loop_p4_v2 的真实动机（2026-09-30）：arena 连着六代用同一个
    ``match.seed``，开局与配色都由那条固定流决定，等于每次都在**同一批固定局面**
    上判决——测试集不是随机样本。训练的固定 seed 也让静态语料的洗牌顺序每代一样。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))

    def test_without_seed_base_nothing_changes(self):
        """默认（不配 seed_base）：三个子配置的 seed 一律不动。"""
        conf = _conf(self.tmp, train={"task": {"factory": "No.such:make_task",
                                              "kwargs": {"base": "{weights}"}},
                                      "steps": 5, "device": "cpu",
                                      "optimizer": {"lr": 5e-06}})
        loop = Loop(conf, self.tmp)
        self.assertIsNone(loop._gen_seed(0, 1))
        for g in (0, 1, 7):
            loop._subst_seeds(g)
        self.assertNotIn("seed", loop.conf["train"])
        self.assertNotIn("seed", loop.conf["selfplay"])
        self.assertNotIn("seed", loop.conf["arena"]["match"])

    def test_seeds_differ_across_generations_but_stable_within(self):
        conf = _conf(self.tmp, seed_base=20260930)
        loop = Loop(conf, self.tmp)
        a = loop._gen_seed(6, 1)
        b = loop._gen_seed(7, 1)
        self.assertNotEqual(a, b)
        # 同一代重复派生必须相同（中断续跑不能换种子）
        self.assertEqual(loop._gen_seed(6, 1), a)
        # 同一代三种用途必须互不相同
        s = {loop._gen_seed(6, w) for w in (1, 2, 3)}
        self.assertEqual(len(s), 3)
        # 必须在合法 int31 区间
        for v in s:
            self.assertTrue(0 <= v < 2 ** 31)

    def test_subst_seeds_writes_all_three_and_idempotent(self):
        conf = _conf(self.tmp, seed_base=20260930)
        loop = Loop(conf, self.tmp)
        loop._subst_seeds(7)
        first = (dict(loop.conf["train"]), dict(loop.conf["selfplay"]),
                 dict(loop.conf["arena"]["match"]))
        loop._subst_seeds(7)                      # 再来一次必须不变（幂等）
        self.assertEqual((dict(loop.conf["train"]), dict(loop.conf["selfplay"]),
                          dict(loop.conf["arena"]["match"])), first)
        self.assertEqual(loop.conf["train"]["seed"], loop._gen_seed(7, 1))
        self.assertEqual(loop.conf["selfplay"]["seed"], loop._gen_seed(7, 2))
        self.assertEqual(loop.conf["arena"]["match"]["seed"], loop._gen_seed(7, 3))
        # 换代会换种子
        loop._subst_seeds(8)
        self.assertNotEqual(loop.conf["train"]["seed"], first[0]["seed"])
        self.assertNotEqual(loop.conf["arena"]["match"]["seed"], first[2]["seed"])

    def test_run_writes_seeds_into_history(self):
        """整代跑通后，history 里要留下本代实际用的三个种子（换 seed_base 好对账）。"""
        conf = _conf(self.tmp, seed_base=20260930, generations=2)
        loop = Loop(conf, self.tmp)
        loop.out.mkdir(parents=True, exist_ok=True)
        gd = loop.gen_dir(1)
        (gd / "train").mkdir(parents=True)
        (gd / "train" / "final.pt").write_bytes(b"\0")
        loop.state_path.write_text(json.dumps({
            "generation": 1, "phase": "arena", "champion": "/w/c.pt", "history": [],
            "variant": None}), encoding="utf-8")

        def fake_arena(self, g, m):
            (loop.gen_dir(g) / "arena.jsonl.summary.json").write_text(json.dumps(
                {"score_a": 0.5, "elo": 0.0, "games": 10,
                 "sprt": {"verdict": "H0"}}), encoding="utf-8")
            return {"score_a": 0.5, "elo": 0.0, "games": 10, "sprt": {"verdict": "H0"}}

        with mock.patch.object(Loop, "phase_arena", fake_arena):
            loop.run()
        rec = json.loads(loop.state_path.read_text(encoding="utf-8"))["history"][-1]
        self.assertEqual(set(rec["seeds"]), {"train", "selfplay", "match"})
        self.assertEqual(rec["seeds"]["train"], loop._gen_seed(1, 1))
        self.assertEqual(rec["seeds"]["match"], loop._gen_seed(1, 3))


class TestArenaDeterministicSelection(unittest.TestCase):
    """arena 两侧必须确定性选着（temperature=0），不受自对弈配置影响。

    动机（2026-10-02）：给自对弈加 ``temperature>0`` 让对局有变化时发现
    ``engine.kwargs`` 是自对弈与 arena 共用的——不挡一手，自对弈的
    temperature 会渗进 arena，同一局能下出不同结果，SPRT 方差白涨。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kit_loop_"))
        self.loop = Loop(_conf(self.tmp, engine={
            "factory": "No.such:factory",
            "kwargs": {"checkpoint": "{weights}", "simulations": 800,
                       "temperature": 2.0, "dirichlet_eps": 0.1,
                       "root_min_visits": 4}}), self.tmp)

    @staticmethod
    def _fake_run(self, subcmd, cfg_path, log_path, extra=()):
        """替掉子进程：只按需造一个假 summary，让 phase_arena 的返回路径走通。"""
        gd = Path(cfg_path).parent
        if subcmd == "match":
            (gd / "arena.jsonl.summary.json").write_text(
                json.dumps({"score_a": 0.5, "sprt": {"verdict": "H0"}}), encoding="utf-8")

    def _write_arena(self, g=3):
        gd = self.loop.gen_dir(g)
        gd.mkdir(parents=True, exist_ok=True)
        m = self.loop.mapping(g, "/w/champ.pt")
        with mock.patch.object(Loop, "_run", self._fake_run):
            self.loop.phase_arena(g, m)
        return json.loads((gd / "arena.json").read_text(encoding="utf-8"))

    def test_arena_forces_temperature_zero(self):
        cfg = self._write_arena()
        for side in ("a", "b"):
            self.assertEqual(cfg[side]["kwargs"]["temperature"], 0,
                             f"arena {side} 必须确定性选着")
        # 其它搜索参数保持透传，不被顺手改掉
        for side in ("a", "b"):
            self.assertEqual(cfg[side]["kwargs"]["dirichlet_eps"], 0.1)
            self.assertEqual(cfg[side]["kwargs"]["root_min_visits"], 4)
            self.assertEqual(cfg[side]["kwargs"]["simulations"], 800)
        # 双方权重必须不同（候选 vs 冠军）
        self.assertNotEqual(cfg["a"]["kwargs"]["checkpoint"],
                            cfg["b"]["kwargs"]["checkpoint"])
        self.assertEqual(cfg["a"]["label"], "gen3")
        self.assertEqual(cfg["b"]["label"], "champion")

    def test_arena_without_temperature_still_zero(self):
        """没配 temperature 时也显式为 0（不依赖默认值）。"""
        loop = Loop(_conf(self.tmp), self.tmp)
        loop.out.mkdir(parents=True, exist_ok=True)
        gd = loop.gen_dir(1)
        gd.mkdir(parents=True, exist_ok=True)
        m = loop.mapping(1, "/w/champ.pt")
        with mock.patch.object(Loop, "_run", self._fake_run):
            loop.phase_arena(1, m)
        cfg = json.loads((gd / "arena.json").read_text(encoding="utf-8"))
        for side in ("a", "b"):
            self.assertEqual(cfg[side]["kwargs"]["temperature"], 0)


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

    def test_schedule_is_replaced_not_merged(self):
        """variant 的 schedule 整体替换：合并会把 onecycle 的 pct_start 带给 constant，
        ``build_schedule`` 会以「未知参数」报错（2026-09-27 写配方时踩到）。"""
        conf = _variant_conf(self.tmp)
        conf["train"]["schedule"] = {"kind": "onecycle", "pct_start": 0.25}
        conf["train"]["variants"] = [
            {"label": "const", "schedule": {"kind": "constant"}, "steps": 10},
            {"label": "1c", "steps": 10},                          # 不覆盖 → 继承基座
        ]
        loop = Loop(conf, self.tmp)
        m = loop.mapping(0, "/w/champ.pt")
        got_const = loop._variant_train_conf(loop.variants[0], m)
        got_inherit = loop._variant_train_conf(loop.variants[1], m)
        self.assertEqual(got_const["schedule"], {"kind": "constant"})
        self.assertEqual(got_inherit["schedule"], {"kind": "onecycle", "pct_start": 0.25})

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
