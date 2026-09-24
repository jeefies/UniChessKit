"""学习率调度。

对象式调度一律用 torch 自带的类（``CosineAnnealingLR`` 是递推式、不是闭式，自己写公式会差最后几位），
每个优化步之后 ``step()`` 一次。手写公式的调度（``warmup_cosine_floor``）在每步前直接写
``param_group["lr"]``，表达式的运算顺序与旧 R iteration46 脚本逐字一致。

kind：
    constant              lr 不变
    cosine                CosineAnnealingLR(T_max=t_max 或 steps, eta_min)
    onecycle              OneCycleLR(max_lr=lr, total_steps=steps, pct_start, anneal_strategy,
                          div_factor, final_div_factor)
    lambda                LambdaLR(fn(step) → 倍率)，fn = "包.模块:函数"，fn_kwargs 传给工厂
                          （``fn`` 指向工厂时用 fn_kwargs 调一次得到 lambda；无 fn_kwargs 则直接当 lambda）
    warmup_cosine_floor   lr·min(1,(s+1)/warmup)·(floor + scale·0.5·(1+cos(π·s/steps)))，每步前手写
"""
from __future__ import annotations

import math

import torch

KINDS = ("constant", "cosine", "onecycle", "lambda", "warmup_cosine_floor")


class ManualSchedule:
    """每步前调用 ``apply(opt, step)``；无内部状态（续训不需要保存）。"""

    def __init__(self, lr: float, warmup: int, floor: float, scale: float, steps: int):
        self.lr, self.warmup, self.floor, self.scale, self.steps = lr, warmup, floor, scale, steps

    def lr_at(self, step: int) -> float:
        return self.lr * min(1, (step + 1) / self.warmup) * (
            self.floor + self.scale * 0.5 * (1 + math.cos(math.pi * step / self.steps)))

    def apply(self, opt, step: int) -> float:
        lr = self.lr_at(step)
        for group in opt.param_groups:
            group["lr"] = lr
        return lr


def build_schedule(spec: dict, opt, steps: int, lr: float):
    """→ (torch 调度器或 None, ManualSchedule 或 None)。"""
    spec = dict(spec or {"kind": "constant"})
    kind = spec.pop("kind", "constant")
    if kind not in KINDS:
        raise ValueError(f"未知调度 {kind!r}，可选 {KINDS}")

    if kind == "constant":
        sched, manual = None, None
    elif kind == "cosine":
        t_max = spec.pop("t_max", steps)
        eta_min = spec.pop("eta_min", 0.0)
        sched, manual = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max,
                                                                   eta_min=eta_min), None
    elif kind == "onecycle":
        kw = {k: spec.pop(k) for k in ("pct_start", "anneal_strategy", "div_factor",
                                       "final_div_factor") if k in spec}
        total = spec.pop("total_steps", steps)
        sched, manual = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                                            **kw), None
    elif kind == "lambda":
        from ..registry import load_object
        fn = load_object(spec.pop("fn"))
        if "fn_kwargs" in spec:
            fn = fn(**spec.pop("fn_kwargs"))
        sched, manual = torch.optim.lr_scheduler.LambdaLR(opt, fn), None
    else:
        manual = ManualSchedule(lr, spec.pop("warmup"), spec.pop("floor"), spec.pop("scale"),
                                spec.pop("steps", steps))
        sched = None
    if spec:
        raise ValueError(f"调度 {kind} 有未知参数 {sorted(spec)}")
    return sched, manual
