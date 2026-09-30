"""Trainer：配置、循环、续跑、SIGTERM、边界。

用一个「确定性小模型 + 合成数据」的 Task 在 CPU 上跑（不需要任何引擎、不需要 GPU）：
- 每一步的 loss 由参数决定 → 中断续跑后的 loss 序列必须与一次跑完**逐位相同**；
- 配置哈希进 `latest.pt`：换了 task 参数再续跑必须拒绝（结果文件的纪律）；
- `save_every=0` / 步数到达时必须落盘，否则「跑完但没 latest.pt」；
- SIGTERM：下一次日志点保存并返回 `state="stopped"`，`latest.pt` 存在；
- `export.final` / `export.best`（无 validate 时）按配置写出；
- `export.select_best_by="train"`（无 validate）时 best 落在**训练 loss 最低**的那一步，
  而不是 final 的副本；默认 `"none"` 保持旧行为（best == final）；
- `export.select_best_by="train"`（无 validate）时 best 落在**训练 loss 最低**的那一步，
  而不是 final 的副本——`"none"` 时保持旧行为（best == final）；
- accum>1 时 backward 的是 `loss/accum`（R iteration46 / T t20m 的口径），
  而日志里记录的是未除的 total（旧脚本 `.item()` 的记录口径）。

学习率调度（``Kit.train.schedule``）的构建与参数体检由 ``TestSchedule`` 钉死，
特别是两个手写调度的 ``warmup`` / 余弦窗口 **缺参数或 ≤ 0 时要有清楚报错**
（loop 的变体会整体替换 ``schedule``，替换后少了 ``warmup`` 是真实场景）。

``TestSelectBestByTrain`` 用的 ``VShapedLossTask`` 在 CE 之外叠一个以 step 为自变量的
U 形偏置，loss 序列必然"先降后升"——1:1 复刻 loop_p4_v2 gen 4 的形状。只有这样
"best = loss 最低点"与"best = final 副本"才分得开。
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Kit.train import TrainConfig, Trainer
from Kit.train.schedule import build_schedule

KIT_ROOT = Path(__file__).resolve().parent.parent.parent


class ToyTask:
    """确定性任务：模型 = 单个 Linear(4 → 2)，数据 = 固定随机张量。

    `loss(model, batch, step)` 只取决于 model 参数和 batch，不依赖随机数，
    所以「一次跑完」与「中断后续跑」的 loss 序列必须逐位一致。
    """

    def __init__(self, dim=4, batch=8, bs_seed=0, tag="toy"):
        self.dim, self.batch_size, self.bs_seed, self.tag = dim, batch, bs_seed, tag
        g = torch.Generator().manual_seed(bs_seed)
        self.x = torch.randn(64, dim, generator=g)
        self.y = (self.x.sum(1) > 0).long()
        self._i = 0

    # ---- TrainTask ----
    def build_model(self):
        torch.manual_seed(1234)
        return torch.nn.Linear(self.dim, 2)

    def param_groups(self, model):
        return [{"params": list(model.parameters())}]

    def batches(self, ctx):
        self._i = ctx.step * self.batch_size
        while True:
            if self._i + self.batch_size > len(self.x):
                self._i = 0
            b = self.x[self._i:self._i + self.batch_size]
            self._i += self.batch_size
            yield (b, self.y[self._i - self.batch_size:self._i])

    def loss(self, model, batch, step):
        x, y = batch
        return F.cross_entropy(model(x), y), {"toy": 1.0}

    def export(self, model, step):
        return {"model": model.state_dict(), "step": step, "task": self.tag}

    def state_dict(self):
        return {"i": self._i, "tag": self.tag}

    def load_state_dict(self, st):
        self._i = int(st.get("i", 0))


def make_config(out, *, steps=6, accum=1, task=None, **kw):
    d = {"task": {"factory": "tests.test_train:ToyTaskFactory", "kwargs": task or {}},
         "out": str(out), "steps": steps, "accum": accum, "seed": 0, "log_every": 1,
         "save_every": 0, "device": "cpu", "optimizer": {"lr": 0.05, "weight_decay": 0.0},
         "schedule": {"kind": "constant"}}
    d.update(kw)
    return TrainConfig.from_dict(d)


class ToyTaskFactory:
    """配置里用『包.模块:函数』引用：真实训练走 registry，这里直接本地构造。"""

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, **kw):
        kw.update(self.kw)
        return ToyTask(**kw)


def _losses(path, drop=("steps_per_s", "sec")):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            for k in drop:
                rec.pop(k, None)
            out.append(rec)
    return out


class TestTrainerLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p19tr_")
        d = Path(self.tmp.name)
        self.d = d
        # 把工厂挂到 tests 包上，registry 的 load_object 按名字找得到
        sys.modules.setdefault("tests.test_train", sys.modules[__name__])
        setattr(sys.modules[__name__], "ToyTaskFactory", ToyTaskFactory)

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_and_logs(self):
        cfg = make_config(self.d / "a", steps=5)
        res = Trainer(cfg, ToyTask()).run()
        self.assertEqual(res["state"], "completed")
        self.assertEqual(res["step"], 5)
        recs = _losses(self.d / "a" / "train.jsonl")
        self.assertEqual([r["step"] for r in recs], [1, 2, 3, 4, 5])
        self.assertTrue((self.d / "a" / "config.json").exists())
        # save_every=0：只在结束保存
        self.assertTrue((self.d / "a" / "latest.pt").exists())
        # 无 validate 时导出 best
        self.assertTrue((self.d / "a" / "best.pt").exists())
        # loss 在下降（0.05 lr、6 步）
        self.assertLess(recs[-1]["loss"], recs[0]["loss"])

    def test_resume_matches_single_run(self):
        """中断后续跑：loss 序列与参数都应与一次跑完逐位一致。

        先跑到 end 得到基准；再用 stop_event 在第 cut 步真正停下，然后按原配置续跑。
        （早先是手工改 latest.pt 的 step——那样优化器 / RNG 状态与 step 不符，必然对不上。）
        """
        import threading

        steps, cut = 24, 6
        if not (self.d / "full" / "train.jsonl").exists():
            Trainer(make_config(self.d / "full", steps=steps, log_every=1), ToyTask()).run()
        recs_full = _losses(self.d / "full" / "train.jsonl")
        self.assertEqual(len(recs_full), steps)

        # 用 task.on_train_start 在走到 cut 步时请求停止：确定性，不依赖墙钟轮询
        class StopAtTask(ToyTask):
            stop_ev = None

            def loss(self, model, batch, step):
                if step + 1 >= cut and self.stop_ev is not None:
                    self.stop_ev.set()
                return super().loss(model, batch, step)

        ev = threading.Event()
        task = StopAtTask()
        task.stop_ev = ev
        res1 = Trainer(make_config(self.d / "part", steps=steps, log_every=1, save_every=1),
                       task, stop_event=ev).run()
        self.assertEqual(res1["state"], "stopped")
        self.assertEqual(res1["step"], cut)
        res2 = Trainer(make_config(self.d / "part", steps=steps, log_every=1, save_every=1),
                       ToyTask()).run()
        self.assertEqual(res2["state"], "completed")
        self.assertEqual(_losses(self.d / "part" / "train.jsonl"), recs_full)
        # 参数逐位相同
        a = torch.load(self.d / "full" / "best.pt", weights_only=False)["model"]
        b = torch.load(self.d / "part" / "best.pt", weights_only=False)["model"]
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_config_hash_refuses_resume(self):
        cfg = make_config(self.d / "x", steps=3, task={"dim": 4})
        Trainer(cfg, ToyTask()).run()               # dim=4
        cfg2 = make_config(self.d / "x", steps=3, task={"dim": 6})
        with self.assertRaises(RuntimeError) as cm:
            Trainer(cfg2, ToyTask(dim=6)).run()
        self.assertIn("配置哈希", str(cm.exception))
        # 换目录则可以（同配置不同 out）
        cfg3 = make_config(self.d / "y", steps=3, task={"dim": 6})
        res = Trainer(cfg3, ToyTask(dim=6)).run()
        self.assertEqual(res["state"], "completed")

    def test_stop_event_saves(self):
        cfg = make_config(self.d / "s", steps=10_000, log_every=1)
        import threading
        ev = threading.Event()
        timer = threading.Timer(1.0, ev.set)
        timer.start()
        try:
            res = Trainer(cfg, ToyTask(), stop_event=ev).run()
        finally:
            timer.cancel()
        self.assertEqual(res["state"], "stopped")
        self.assertLess(res["step"], 10_000)
        self.assertTrue((self.d / "s" / "latest.pt").exists())

    def test_accum_divides_backward(self):
        """backward 的是 loss/accum；日志的 loss 是未除总量（旧脚本 `.item()` 口径）。"""
        cfg = make_config(self.d / "ac", steps=3, accum=4, log_every=1)
        seen = []
        orig = torch.Tensor.backward

        def spy(self, *a, **kw):
            if self.dim() == 0:
                seen.append(float(self.detach()))
            return orig(self, *a, **kw)

        torch.Tensor.backward = spy
        try:
            Trainer(cfg, ToyTask()).run()
        finally:
            torch.Tensor.backward = orig
        recs = _losses(self.d / "ac" / "train.jsonl")
        # 4 个 microbatch 的 loss 之和 ≈ 日志的 loss
        self.assertEqual(len(seen), 12)
        by_step = [seen[i * 4:(i + 1) * 4] for i in range(3)]
        for r, chunk in zip(recs, by_step):
            self.assertAlmostEqual(sum(chunk), r["loss"], places=4)


class TestSelectBestByTrain(unittest.TestCase):
    """无 validation 时 ``export.select_best_by`` 的两套口径。

    2026-09-30 的真实事故（Transformer ``loop_p4_v2`` gen 4）：1200 步 onecycle 的
    loss 在 step 300 就见底、之后单调回升 0.0255，而 ``export.final`` 只导最后一步
    ——导出的正是全程最差的点，候选因此比 base 弱 74.8 Elo 判 H0。
    ``select_best_by="train"`` 就是兜住"训过头"：best 落在 loss 最低那一步。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p19tr_")
        self.d = Path(self.tmp.name)
        sys.modules.setdefault("tests.test_train", sys.modules[__name__])
        setattr(sys.modules[__name__], "ToyTaskFactory", ToyTaskFactory)

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self, out, *, export_extra, steps=40, log_every=1):
        return make_config(out, steps=steps, log_every=log_every,
                           export={"best": "best.pt", "final": "final.pt", **export_extra})

    def _loss_seq(self, name="train"):
        recs = _losses(self.d / name / "train.jsonl")
        return recs, [r["loss"] for r in recs]

    def test_best_falls_on_min_loss_step_not_final(self):
        cfg = self._cfg(self.d / "train", export_extra={"select_best_by": "train"})
        res = Trainer(cfg, VShapedLossTask(mid=20)).run()
        self.assertEqual(res["state"], "completed")
        recs, losses = self._loss_seq("train")
        argmin = int(np.argmin(losses)) + 1                 # 1-based 步号
        best = torch.load(self.d / "train" / "best.pt", weights_only=False)
        final = torch.load(self.d / "train" / "final.pt", weights_only=False)
        # 前提：argmin 必须严格落在中间，否则本测试什么都证明不了
        self.assertLess(argmin, len(losses), "argmin 落在最后一步，测试失去意义")
        self.assertEqual(best["step"], argmin, "best 应落在 loss 最低的那一步")
        self.assertEqual(final["step"], len(losses))
        self.assertNotEqual(best["step"], final["step"])
        self.assertLess(min(losses), losses[-1], "终值应高于最低点")
        # 只有 argmin 那一步记 improved，且 best 收敛到最小 loss
        improved = [r for r in recs if r.get("improved")]
        self.assertTrue(improved, "没有任何一步被记成 improved")
        self.assertAlmostEqual(improved[-1]["best"], min(losses), places=9)
        # 返回值的 best 变成训练 loss，不再是 Infinity
        self.assertTrue(math.isfinite(res["best"]))
        self.assertAlmostEqual(res["best"], min(losses), places=9)

    def test_select_best_by_none_keeps_best_equals_final(self):
        """默认 'none'：保持旧行为，best 就是 final 的副本。"""
        cfg = self._cfg(self.d / "none", export_extra={})
        res = Trainer(cfg, VShapedLossTask(mid=20)).run()
        recs, losses = self._loss_seq("none")
        best = torch.load(self.d / "none" / "best.pt", weights_only=False)
        final = torch.load(self.d / "none" / "final.pt", weights_only=False)
        self.assertEqual(best["step"], final["step"])
        self.assertEqual(best["step"], len(recs))
        self.assertEqual(res["best"], math.inf)
        for k in final["model"]:
            self.assertTrue(torch.equal(best["model"][k], final["model"][k]),
                            f"'none' 口径下 best 应与 final 逐位相同：{k}")
        # 同一份 V 形 loss，'none' 把最低点让给了 final
        self.assertNotEqual(final["step"], int(np.argmin(losses)) + 1)

    def test_select_best_by_rejects_unknown_value(self):
        with self.assertRaises(ValueError) as cm:
            self._cfg(self.d / "bad", export_extra={"select_best_by": "loss"})
        self.assertIn("select_best_by", str(cm.exception))

    def test_select_best_by_train_survives_resume(self):
        """中断续跑后，best 仍应是全程 loss 最低点（best 随 latest.pt 传递）。"""
        import threading

        steps, cut = 24, 8
        extra = {"select_best_by": "train"}
        Trainer(self._cfg(self.d / "full", export_extra=extra, steps=steps),
                VShapedLossTask(mid=12)).run()
        recs_full, losses_full = self._loss_seq("full")
        argmin_full = int(np.argmin(losses_full)) + 1

        ev = threading.Event()
        task = VShapedLossTask(mid=12, on_step=lambda s: ev.set() if s + 1 >= cut else None)
        res1 = Trainer(self._cfg(self.d / "part", export_extra=extra, steps=steps),
                       task, stop_event=ev).run()
        self.assertEqual(res1["state"], "stopped")
        self.assertLess(res1["step"], steps)
        Trainer(self._cfg(self.d / "part", export_extra=extra, steps=steps),
                VShapedLossTask(mid=12)).run()
        recs_part, losses_part = self._loss_seq("part")
        self.assertEqual(losses_part, losses_full, "续跑的 loss 序列应与一次跑完一致")
        best = torch.load(self.d / "part" / "best.pt", weights_only=False)
        self.assertEqual(best["step"], argmin_full,
                         "续跑后 best 仍应是全程 loss 最低点")


class VShapedLossTask(ToyTask):
    """在 CE 之外叠加一个以 step 为自变量的 U 形偏置，loss 序列必然"先降后升"。

    1:1 复刻 2026-09-30 loop_p4_v2 gen 4 的形状（loss 在 step 300 见底后单调回升）。
    偏置项对参数梯度为 0，不改变训练本身，只把 argmin 钉在中间某一步——
    这样"best = loss 最低点"和"best = final 副本"两种口径才能被真正区分。

    ``on_step(step)`` 是钩子，测试用来在指定步请求停止（确定性中断，不依赖墙钟）。
    """

    def __init__(self, mid=None, depth=0.01, on_step=None, **kw):
        super().__init__(**kw)
        self.mid, self.depth, self.on_step = mid, depth, on_step

    def loss(self, model, batch, step):
        x, y = batch
        if self.on_step is not None:
            self.on_step(step)
        ce = F.cross_entropy(model(x), y)
        dip = self.depth * (step - self.mid) ** 2 if self.mid is not None else 0.0
        return ce + dip, {"toy": 1.0}


class TestSchedule(unittest.TestCase):
    """``build_schedule``：kind 分派、参数体检、手写调度的 warmup 语义。

    2026-09-27 loop 的实况：``train.variants`` 里一个变体写
    ``{"kind": "warmup_cosine_floor", "floor": 0.1, "scale": 0.9}``（同 kind 的其他参数
    想沿用基座），而 schedule 是**整体替换**，基座的 ``warmup: 2000`` 整个丢掉 →
    ``_resolve_warmup`` 退到 0 → ``(step + 1) / warmup`` 除零，训练循环里才炸。
    缺参数 / 余弦窗口 0 / 负 warmup 都必须在**构建时**就报清楚。
    """

    def _opt(self, lr=1e-3):
        p = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.AdamW([p], lr=lr)

    def test_unknown_kind_and_constant(self):
        sched, manual = build_schedule({"kind": "constant"}, self._opt(), 10, 1e-3)
        self.assertIsNone(sched)                      # constant 不需要调度器
        self.assertIsNone(manual)
        with self.assertRaises(ValueError):
            build_schedule({"kind": "nope"}, self._opt(), 10, 1e-3)

    def test_onecycle_uses_total_steps_default(self):
        opt = self._opt()
        sched, manual = build_schedule({"kind": "onecycle", "pct_start": 0.25}, opt, 100, 1e-3)
        self.assertIsNotNone(sched)
        self.assertIsNone(manual)
        self.assertEqual(sched.total_steps, 100)      # 缺省 total_steps = 训练步数
        sched.step()
        self.assertGreater(opt.param_groups[0]["lr"], 0.0)

    def test_warmup_cosine_floor_without_warmup_is_full_lr(self):
        """``warmup`` 缺省 / 为 0 = 没有 warmup，第 0 步就是全 lr（不再除零）。"""
        sched, manual = build_schedule({"kind": "warmup_cosine_floor", "floor": 0.1,
                                        "scale": 0.9}, self._opt(), 100, 1e-3)
        self.assertIsNone(sched)
        self.assertEqual(manual.warmup, 0)
        self.assertAlmostEqual(manual.lr_at(0), 1e-3)      # 0.1 + 0.9·0.5·(1+cos 0) = 1.0
        self.assertAlmostEqual(manual.lr_at(100), 1e-4)    # 末端：0.1 + 0 = 0.1

    def test_warmup_cosine_floor_warmup_factor(self):
        """有 warmup 时前 warmup 步按 (s+1)/warmup 线性放大（旧 R iteration46 口径）。"""
        _, manual = build_schedule({"kind": "warmup_cosine_floor", "warmup": 4, "floor": 0.0,
                                    "scale": 1.0}, self._opt(), 100, 1e-3)
        self.assertEqual(manual.warmup, 4)
        # 前 warmup 步只有全 lr 的 (s+1)/warmup，余弦部分照算
        self.assertAlmostEqual(manual.lr_at(0),
                               1e-3 * 0.25 * 0.5 * (1 + math.cos(0.0)))
        self.assertAlmostEqual(manual.lr_at(3),
                               1e-3 * 1.0 * 0.5 * (1 + math.cos(math.pi * 3 / 100)))
        # 到点之后不再被放大/压住：倍率恒为 1
        self.assertAlmostEqual(manual.lr_at(4),
                               1e-3 * 1.0 * 0.5 * (1 + math.cos(math.pi * 4 / 100)))

    def test_warmup_then_cosine_without_warmup(self):
        sched, manual = build_schedule({"kind": "warmup_then_cosine", "floor": 0.1,
                                        "scale": 0.9}, self._opt(), 100, 1e-3)
        self.assertIsNone(sched)
        self.assertEqual(manual.warmup, 0)
        self.assertAlmostEqual(manual.lr_at(0), 1e-3)
        self.assertAlmostEqual(manual.lr_at(100), 1e-4)

    def test_manual_schedule_rejects_bad_params(self):
        for spec in ({"kind": "warmup_cosine_floor", "scale": 0.9},            # 缺 floor
                     {"kind": "warmup_cosine_floor", "floor": 0.1},            # 缺 scale
                     {"kind": "warmup_then_cosine", "floor": 0.1, "scale": 0.9,
                      "steps": 0},                                              # 余弦窗口 0
                     {"kind": "warmup_cosine_floor", "floor": 0.1, "scale": 0.9,
                      "warmup": -1}):                                           # 负 warmup
            with self.assertRaises(ValueError, msg=f"{spec} 该报错"):
                build_schedule(spec, self._opt(), 100, 1e-3)

    def test_object_schedules_reject_empty_window(self):
        with self.assertRaises(ValueError):
            build_schedule({"kind": "cosine", "t_max": 0}, self._opt(), 100, 1e-3)
        with self.assertRaises(ValueError):
            build_schedule({"kind": "onecycle", "total_steps": 0}, self._opt(), 100, 1e-3)


if __name__ == "__main__":
    unittest.main()
