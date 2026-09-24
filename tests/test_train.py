"""Trainer：配置、循环、续跑、SIGTERM、边界。

用一个「确定性小模型 + 合成数据」的 Task 在 CPU 上跑（不需要任何引擎、不需要 GPU）：
- 每一步的 loss 由参数决定 → 中断续跑后的 loss 序列必须与一次跑完**逐位相同**；
- 配置哈希进 `latest.pt`：换了 task 参数再续跑必须拒绝（结果文件的纪律）；
- `save_every=0` / 步数到达时必须落盘，否则「跑完但没 latest.pt」；
- SIGTERM：下一次日志点保存并返回 `state="stopped"`，`latest.pt` 存在；
- `export.final` / `export.best`（无 validate 时）按配置写出；
- accum>1 时 backward 的是 `loss/accum`（R iteration46 / T t20m 的口径），
  而日志里记录的是未除的 total（旧脚本 `.item()` 的记录口径）。
"""
from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
