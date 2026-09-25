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
                  "steps": 5, "device": "cpu"},
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


if __name__ == "__main__":
    unittest.main()
