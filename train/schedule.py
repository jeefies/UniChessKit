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
    warmup_then_cosine    warmup 段 (s+1)/warmup；之后 floor + scale·0.5·(1+cos(π·(s−warmup)/(steps−warmup)))
                          （S 的 Stage A/B 口径，余弦窗口从 warmup 结束处起算）
    两个 warmup 调度的 ``warmup`` 可以是固定步数，也可以用 warmup_frac（×steps）与
    warmup_max 推导（S Stage B 的 min(200, 步数×10%)）。
"""
from __future__ import annotations

import math

import torch

KINDS = ("constant", "cosine", "onecycle", "lambda", "warmup_cosine_floor",
         "warmup_then_cosine")


class ManualSchedule:
    """每步前调用 ``apply(opt, step)``；无内部状态（续训不需要保存）。

    ``warmup <= 0`` 表示**没有 warmup**（从第 0 步就是全 lr）。0 是 ``_resolve_warmup``
    的缺省值，这里不能再拿它做除数——2026-09-27 的实况是：基座带 ``warmup: 2000`` 的
    ``warmup_cosine_floor`` 被变体整体替换成不带 ``warmup`` 的同 kind 小字典，
    ``(step + 1) / warmup`` 当场 ZeroDivisionError（模型已建、数据流已开）。
    """

    def __init__(self, lr: float, warmup: int, floor: float, scale: float, steps: int):
        self.lr, self.warmup, self.floor, self.scale, self.steps = lr, warmup, floor, scale, steps

    def lr_at(self, step: int) -> float:
        warm = min(1, (step + 1) / self.warmup) if self.warmup > 0 else 1.0
        return self.lr * warm * (
            self.floor + self.scale * 0.5 * (1 + math.cos(math.pi * step / self.steps)))

    def apply(self, opt, step: int) -> float:
        lr = self.lr_at(step)
        for group in opt.param_groups:
            group["lr"] = lr
        return lr


class WarmupThenCosine:
    """先线性 warmup，再在**剩余步数**上做余弦（S 的 Stage A / B 口径）。

    ``warmup`` 段倍率 ``(s+1)/warmup``；之后
    ``floor + scale·0.5·(1+cos(π·min((s−warmup)/max(steps−warmup,1),1)))``——
    余弦窗口从 warmup 结束处开始，而不是从第 0 步。与旧 ``train/stage_a.py`` 的
    ``lr_lambda`` 逐字一致（旧式是 ``0.1 + 0.45*(1+cos)``，即 floor=0.1 / scale=0.9）。
    """

    def __init__(self, lr: float, warmup: int, floor: float, scale: float, steps: int):
        self.lr, self.warmup = lr, int(warmup)
        self.floor, self.scale, self.steps = floor, scale, int(steps)

    def lr_at(self, step: int) -> float:
        if step < self.warmup:
            return self.lr * ((step + 1) / self.warmup)
        t = (step - self.warmup) / max(self.steps - self.warmup, 1)
        return self.lr * (self.floor + self.scale * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    def apply(self, opt, step: int) -> float:
        lr = self.lr_at(step)
        for group in opt.param_groups:
            group["lr"] = lr
        return lr


def _resolve_warmup(spec: dict, steps: int) -> int:
    """``warmup`` 可以是固定步数，也可以用 ``warmup_frac`` / ``warmup_max`` 由步数推。

    S Stage B 的口径是 ``min(200, int(总步数×10%))``——总步数本身由数据量决定（见 Trainer
    的 ``auto_steps``），所以 warmup 也必须跟着推；写死 200 会把前几步的学习率压错 50 倍。
    """
    if "warmup_frac" not in spec and "warmup_max" not in spec:
        return int(spec.pop("warmup", 0))
    frac = float(spec.pop("warmup_frac", 1.0))
    w = int(steps * frac)
    wmax = spec.pop("warmup_max", None)
    if wmax is not None:
        w = min(int(wmax), w)
    if "warmup" in spec:
        raise ValueError("warmup 与 warmup_frac/warmup_max 不能同时给")
    spec.pop("warmup", None)
    return w


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
        if int(t_max) <= 0:
            raise ValueError(f"调度 cosine 的 t_max 应 > 0，收到 {t_max}")
        sched, manual = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max,
                                                                   eta_min=eta_min), None
    elif kind == "onecycle":
        kw = {k: spec.pop(k) for k in ("pct_start", "anneal_strategy", "div_factor",
                                       "final_div_factor") if k in spec}
        total = spec.pop("total_steps", steps)
        if int(total) <= 0:
            raise ValueError(f"调度 onecycle 的 total_steps 应 > 0，收到 {total}")
        sched, manual = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total,
                                                            **kw), None
    elif kind == "lambda":
        from ..registry import load_object
        fn = load_object(spec.pop("fn"))
        if "fn_kwargs" in spec:
            fn = fn(**spec.pop("fn_kwargs"))
        sched, manual = torch.optim.lr_scheduler.LambdaLR(opt, fn), None
    elif kind in ("warmup_cosine_floor", "warmup_then_cosine"):
        # 两个手写调度吃同一组参数；floor / scale / 余弦窗口（steps）缺一个或 ≤ 0 都要
        # 在**构建时**说清楚，不能等训练循环里除零 / KeyError 才炸
        missing = [k for k in ("floor", "scale") if k not in spec]
        if missing:
            raise ValueError(f"调度 {kind} 缺少必填参数 {missing}")
        warmup = _resolve_warmup(spec, steps)
        if warmup < 0:
            raise ValueError(f"调度 {kind} 的 warmup 应 >= 0（0 = 没有 warmup），收到 {warmup}")
        window = int(spec.pop("steps", steps))
        if window <= 0:
            raise ValueError(f"调度 {kind} 的余弦窗口 steps 应 > 0，收到 {window}")
        if kind == "warmup_cosine_floor":
            manual = ManualSchedule(lr, warmup, spec.pop("floor"), spec.pop("scale"), window)
        else:
            manual = WarmupThenCosine(lr, warmup, spec.pop("floor"), spec.pop("scale"), window)
        sched = None
    if spec:
        raise ValueError(f"调度 {kind} 有未知参数 {sorted(spec)}")
    return sched, manual
