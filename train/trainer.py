"""统一训练器：``TrainTask`` 协议 + ``Trainer``。

引擎仓库实现 ``TrainTask``（模型结构、数据、损失、导出格式），训练循环、优化器、调度、
混合精度、梯度累积 / 裁剪、检查点、续训、SIGTERM 都在这里。

**一个优化步**（与旧脚本逐位对齐的顺序）::

    [手写调度写 lr] → zero_grad(set_to_none) →
    accum × { batch = next(流); autocast{loss = task.loss(model, batch, step)};
              (scaler.scale)(loss / accum if accum > 1 else loss).backward() } →
    [scaler.unscale_] → clip_grad_norm_ → [非有限检查] → scaler.step / opt.step → scaler.update →
    [调度器 step] → step += 1

**RNG 次序**：seed → ``task.build_model()``（参数初始化）→ 优化器 / 调度器 → 数据流在第一次取批时
才建立（DataLoader 抽 base_seed、采样器抽种子）。与旧 R stage1 / T 脚本相同。

**续训**：``latest.pt`` 存模型、优化器、调度器、scaler、全部 RNG、step、best、``task.state_dict()``
与配置哈希；哈希不符拒绝续训。续训时先恢复 RNG 再建数据流；DataLoader 建迭代器时会额外抽一次
全局 CPU RNG（base_seed），训练里没有别的地方用全局 CPU RNG（dropout 在 GPU 上用 CUDA RNG），
所以不影响结果——``tests/test_train.py`` 验证了中断续跑与一次跑完逐位相同。

**SIGTERM / SIGINT**：做完当前优化步，写 ``latest.pt`` 后返回 ``{"state": "stopped"}``。

目录内容（``cfg.out``）::

    latest.pt       续训状态（原子写）
    <export.best>   task.validate 的 score 创新低时导出（引擎可直接加载的格式）；
                    没有 validation 时由 ``export.select_best_by`` 决定口径
                    （``train`` = 训练 loss 最低的那一步；``none`` = 训练结束时的 final）
    <export.final>  训练结束时导出
    train.jsonl     每 log_every 步一行：step / lr / 各项损失（accum 内平均）/ grad_norm / 吞吐
    config.json     本次配置（含哈希）
    train.lock      进程存活期间持有，防止两个训练器写同一目录
"""
from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

# 16 GB 卡上训练显存的大头是激活（microbatch 512），caching allocator 的碎片能到 1–2 GB。
# expandable_segments 只改分配器行为、不影响任何数值；放在 torch 之前（CUDA 首次分配前读到）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from ..registry import load_object
from ..runtime.locks import FileLock
from .checkpoint import (optimizer_state_like_params, rng_state, save_atomic, seed_everything,
                         set_rng_state)
from .config import TrainConfig
from .schedule import build_schedule

CKPT_FORMAT = "kit-train-v1"


@runtime_checkable
class TrainTask(Protocol):
    def build_model(self) -> torch.nn.Module: ...

    def param_groups(self, model) -> list: ...

    def batches(self, ctx: "TrainContext") -> Iterator[Any]: ...

    def loss(self, model, batch, step: int): ...

    def export(self, model, step: int) -> dict: ...


class TrainContext:
    """传给 ``task.batches`` / ``task.validate``：模型、设备、配置、当前步。"""

    def __init__(self, model, device, cfg: TrainConfig, out: Path):
        self.model, self.device, self.cfg, self.out = model, device, cfg, out
        self.step = 0
        self.runtime = dict(cfg.runtime)


def build_task(cfg: TrainConfig):
    factory = load_object(cfg.task["factory"])
    task = factory(**cfg.task.get("kwargs", {}), runtime=dict(cfg.runtime))
    for name in ("build_model", "param_groups", "batches", "loss", "export"):
        if not callable(getattr(task, name, None)):
            raise TypeError(f"{cfg.task['factory']} 返回的任务缺少 {name}()")
    return task


def _optional(task, name):
    fn = getattr(task, name, None)
    return fn if callable(fn) else None


def _to_float(v) -> float:
    if torch.is_tensor(v):
        return float(v.detach().float().item())
    return float(v)


class Trainer:
    def __init__(self, cfg: TrainConfig, task=None, *, stop_event: Optional[threading.Event] = None):
        self.cfg = cfg
        self.task = task
        self.out = Path(cfg.out)
        self.stop_event = stop_event or threading.Event()

    # ------------------------------------------------------------------ 设置
    def _torch_settings(self) -> None:
        t = self.cfg.torch
        if t.get("num_threads"):
            torch.set_num_threads(int(t["num_threads"]))
        if "cudnn_benchmark" in t:
            torch.backends.cudnn.benchmark = bool(t["cudnn_benchmark"])
        if "tf32" in t:
            torch.backends.cuda.matmul.allow_tf32 = bool(t["tf32"])
            torch.backends.cudnn.allow_tf32 = bool(t["tf32"])
        if t.get("deterministic"):
            torch.use_deterministic_algorithms(True)
        frac = t.get("cuda_mem_fraction")          # 很少用：大模型 + 多进程并训时才限制
        if frac and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(frac), 0)

    def _optimizer(self, model):
        o = dict(self.cfg.optimizer)
        groups = self.task.param_groups(model)
        kw = {"lr": o.get("lr", 1e-3), "weight_decay": o.get("weight_decay", 1e-2),
              "betas": tuple(o.get("betas", (0.9, 0.999))), "eps": o.get("eps", 1e-8)}
        fused = o.get("fused")
        if fused is not None:
            kw["fused"] = bool(fused)
        return torch.optim.AdamW(groups, **kw)

    def _clip_params(self, model, opt) -> list:
        fn = _optional(self.task, "clip_params")
        if fn is not None:
            return list(fn(model))
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        return [p for p in model.parameters() if id(p) in in_opt]   # 模型顺序（与旧脚本一致）

    # ------------------------------------------------------------------ 检查点
    def _state(self, model, opt, sched, scaler, step, best) -> dict:
        task_state = _optional(self.task, "state_dict")
        return {"format": CKPT_FORMAT, "config_hash": self.cfg.hash(),
                "config": self.cfg.to_dict(), "step": step, "best": best,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None,
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "rng": rng_state(), "task": task_state() if task_state else None}

    def _export(self, model, step, name) -> None:
        if not name:
            return
        path = self.out / name.format(step=step)
        save_atomic(self.task.export(model, step), path)

    # ------------------------------------------------------------------ 主循环
    def run(self) -> dict:
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.out / "train.lock")
        if not lock.acquire(timeout=0):
            raise RuntimeError(f"{self.out} 正被另一个训练进程使用")
        try:
            return self._run(lock)
        finally:
            lock.release()

    def _run(self, lock) -> dict:
        cfg = self.cfg
        self._torch_settings()
        device = torch.device(cfg.device)
        latest = self.out / "latest.pt"
        ck = None
        if latest.exists():
            ck = torch.load(latest, map_location="cpu", weights_only=False)
            if ck.get("format") != CKPT_FORMAT:
                raise RuntimeError(f"{latest} 不是 kit 训练检查点")
            if ck["config_hash"] != cfg.hash():
                raise RuntimeError(f"{latest} 的配置哈希 {ck['config_hash']} 与当前配置 "
                                   f"{cfg.hash()} 不符，拒绝续训（换目录或恢复原配置）")
        if cfg.seed is not None:
            seed_everything(cfg.seed)
        if self.task is None:
            self.task = build_task(cfg)
        task = self.task
        if cfg.steps == 0:      # "跑一个 epoch"：步数由任务的 auto_steps(accum) 从数据量算，
            auto = _optional(task, "auto_steps")   # 必须在建模型/调度之前定下来
            resolved = auto(cfg.accum) if auto else None
            if not resolved or int(resolved) <= 0:
                raise RuntimeError("steps=0 需要任务提供 auto_steps(accum) 返回正步数")
            cfg.steps = int(resolved)
            print(f"steps=0 → 任务的 auto_steps(accum={cfg.accum}) = {cfg.steps}", flush=True)
        model = task.build_model().to(device)
        opt = self._optimizer(model)
        sched, manual = build_schedule(cfg.schedule, opt, cfg.steps,
                                       cfg.optimizer.get("lr", 1e-3))
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.grad_scaler)
        step, best = 0, math.inf
        if ck is not None:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            optimizer_state_like_params(opt)
            if sched is not None:
                sched.load_state_dict(ck["scheduler"])
            if scaler.is_enabled() and ck["scaler"] is not None:
                scaler.load_state_dict(ck["scaler"])
            if _optional(task, "load_state_dict") and ck["task"] is not None:
                task.load_state_dict(ck["task"])
            set_rng_state(ck["rng"])
            step, best = int(ck["step"]), float(ck["best"])
            del ck
        (self.out / "config.json").write_text(
            json.dumps({"config_hash": cfg.hash(), **cfg.to_dict()}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        export = cfg.export
        if step >= cfg.steps:
            # 已经训完又被拉起（续跑 / loop 重入 phase_train）：照样要按配置导出，
            # 否则调用方拿不到候选文件。2026-09-30 实测：loop 改候选取 best_model.pt
            # 后这里直接 return，phase_train 报「训练结束但没有导出」而崩。
            # 此时只有 latest.pt 的**最终**状态，best 的真实权重已不可重建；
            # 磁盘上已有 best（上一次跑出来的）就原样留着，别用 final 覆盖掉它，
            # 只有文件缺失时才拿当前步顶上，保证调用方按 export.best 找得到东西。
            self._export(model, step, export.get("final", "final.pt"))
            best_name = export.get("best", "best.pt")
            if best_name and not (self.out / best_name).exists():
                self._export(model, step, best_name)
                best = float(step)
            return {"state": "completed", "step": step, "best": best,
                    "peak_vram_mb": _peak_vram_mb()}

        ctx = TrainContext(model, device, cfg, self.out)
        ctx.step = step
        on_start = _optional(task, "on_train_start")
        if on_start:
            on_start(model, ctx)
        clip_params = self._clip_params(model, opt)
        validate = _optional(task, "validate")
        amp = (torch.autocast(device.type, dtype=torch.bfloat16) if cfg.precision == "bf16"
               else contextlib.nullcontext())
        stream = task.batches(ctx)
        # 有模型的损失依赖"训练总步数"做退火（如 S 的 recon 权重 1.0→0.1）。任务的 loss()
        # 若声明了 total_steps 形参就传给它，否则保持老的三参数调用。
        loss_fn = task.loss
        try:
            loss_wants_total = "total_steps" in inspect.signature(loss_fn).parameters
        except (TypeError, ValueError):        # 内置函数 / C 实现的 callable
            loss_wants_total = False
        log_path = self.out / "train.jsonl"

        prev = {}
        in_main = threading.current_thread() is threading.main_thread()
        if in_main:
            def request_stop(signum, frame):
                self.stop_event.set()
            for sig in (signal.SIGTERM, signal.SIGINT):
                prev[sig] = signal.signal(sig, request_stop)
        state = "completed"
        t0, step0 = time.time(), step
        sums: dict = {}
        n_sum = 0
        try:
            model.train()
            while step < cfg.steps:
                ctx.step = step
                logged_loss = None            # 本轮窗口的训练 loss，选 best 时用
                lr = manual.apply(opt, step) if manual is not None else None
                opt.zero_grad(set_to_none=True)
                total = None
                for _ in range(cfg.accum):
                    batch = next(stream)
                    with amp:
                        loss, parts = (task.loss(model, batch, step, total_steps=cfg.steps)
                                       if loss_wants_total
                                       else task.loss(model, batch, step))
                    scaled = loss / cfg.accum if cfg.accum > 1 else loss
                    (scaler.scale(scaled) if scaler.is_enabled() else scaled).backward()
                    d = loss.detach()
                    total = d if total is None else total + d
                    for k, v in (parts or {}).items():
                        # parts 既可以是 0 维张量，也可以是 Python 数值（旧脚本的 metrics.jsonl
                        # 就是纯 float）；只收标量，list / 多维张量一律跳过。
                        if isinstance(v, torch.Tensor):
                            if v.ndim != 0:
                                continue
                            v = float(v.detach().float())
                        elif isinstance(v, bool) or not isinstance(v, (int, float)):
                            continue
                        sums[k] = sums.get(k, 0.0) + float(v) / cfg.accum
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                norm = (torch.nn.utils.clip_grad_norm_(clip_params, cfg.clip) if cfg.clip > 0
                        else None)
                if cfg.nonfinite != "ignore":
                    ok = bool(torch.isfinite(total)) and (norm is None or bool(torch.isfinite(norm)))
                    if not ok:
                        if cfg.nonfinite == "raise":
                            raise FloatingPointError(f"step {step}: 损失或梯度非有限")
                        opt.zero_grad(set_to_none=True)          # skip：丢掉这一步
                        step += 1
                        if sched is not None:
                            sched.step()
                        continue
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                if sched is not None:
                    sched.step()
                step += 1
                ctx.step = step
                sums["loss"] = sums.get("loss", 0.0) + total.float() / cfg.accum
                n_sum += 1
                if norm is not None:
                    sums["grad_norm"] = sums.get("grad_norm", 0.0) + norm.float()

                if step % cfg.log_every == 0 or step == step0 + 1 or step == cfg.steps:
                    rec = {"step": step, "lr": lr if lr is not None else opt.param_groups[0]["lr"]}
                    rec.update({k: _to_float(v) / n_sum for k, v in sums.items()})
                    dt = time.time() - t0
                    rec["steps_per_s"] = round((step - step0) / max(dt, 1e-9), 3)
                    rec["sec"] = round(dt, 1)
                    # 把本窗口的训练 loss 留下来：下面选 best 可能用到（validate 拿不到
                    # score 时就靠它），也可能用不到（有真 validation 时）。
                    logged_loss = float(rec["loss"]) if "loss" in rec else None
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(json.dumps(rec, ensure_ascii=False), flush=True)
                    sums, n_sum = {}, 0

                stop = self.stop_event.is_set()
                do_val = validate is not None and (
                    step in cfg.validate_at or (cfg.validate_every and step % cfg.validate_every == 0)
                    or step == cfg.steps)
                # 「按训练 loss 选最优步」的总开关。注意判的是**拿不到 validation score**，
                # 不是「validate 方法不存在」——T/R 的 Planes19Task 自带 validate 方法，
                # 但没配验证数据源时它返回空 dict、score 为 None，行为与没有方法完全一致。
                # 2026-09-30 实测踩过：写成 `validate is None` 导致分支永不触发，
                # best 恒为 Infinity、best_model.pt 不导出，loop 的 phase_train
                # 直接 fail-fast 崩掉（gen_0006 训练完成但 arena 没跑）。
                by_train_loss = (export.get("best")
                                 and export.get("select_best_by", "none") == "train")
                if do_val:
                    model.eval()
                    with torch.no_grad():
                        res = validate(model, ctx) or {}
                    model.train()
                    score = res.get("score")
                    improved = score is not None and score < best
                    if improved:
                        best = float(score)
                        self._export(model, step, export.get("best", "best.pt"))
                    elif (score is None and by_train_loss and logged_loss is not None
                            and logged_loss < best):
                        # validate 存在但没配数据源：退回到按训练 loss 选点
                        best, improved = logged_loss, True
                        self._export(model, step, export.get("best"))
                    rec = {"step": step, "validation": {k: _to_float(v) for k, v in res.items()},
                           "best": best, "improved": improved}
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(json.dumps(rec, ensure_ascii=False), flush=True)
                elif (by_train_loss and logged_loss is not None and logged_loss < best):
                    # 完全没有 validate 方法：按训练 loss 选「最优步」导出 best。
                    # 否则 export.best 只是 final 的副本——``best`` 恒为 Infinity、
                    # 日志里永远 ``improved=false``，看着像有早停其实没有。
                    # 2026-09-30 实测教训（Transformer loop_p4_v2 gen 4）：1200 步 onecycle
                    # 的 loss 在 step 300（lr 峰值处）就见底，之后单调回升 0.0255，
                    # 而导出的 final 正是全程最差的点；候选因此比 base 弱 74.8 Elo。
                    # 口径说明：logged_loss 是最近 log_every 个**优化步**的平均，
                    # 覆盖 log_every×accum×batch 个样本（循环里 50×4×512≈10 万），
                    # 比单步 loss 稳得多，可以拿来选点。
                    best = logged_loss
                    self._export(model, step, export.get("best"))
                    rec = {"step": step, "best": best, "improved": True}
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(json.dumps(rec, ensure_ascii=False), flush=True)
                if export.get("every") and step % int(export["every"]) == 0:
                    self._export(model, step, export.get("every_name", "step_{step:08d}.pt"))
                # save_every = 0 表示「只在结束 / 收到停止信号时保存」
                if stop or step == cfg.steps or (cfg.save_every > 0
                                                 and step % cfg.save_every == 0):
                    save_atomic(self._state(model, opt, sched, scaler, step, best),
                                self.out / "latest.pt")
                if stop and step < cfg.steps:
                    state = "stopped"
                    break
            if state == "completed":
                self._export(model, step, export.get("final", "final.pt"))
                # best 已在上面选出（validation score 或训练 loss）；只有一步都没选出
                # 来（极短训练 / 续跑紧贴结尾、整个区间没落到 log 点上）才退回复制 final，
                # 免得调用方按 export.best 找不到文件。
                if (not math.isfinite(best) and export.get("best", "best.pt")):
                    self._export(model, step, export.get("best", "best.pt"))
        finally:
            for sig, h in prev.items():
                signal.signal(sig, h)
            close = getattr(stream, "close", None)
            if close:
                close()
            task_close = _optional(task, "close")
            if task_close:
                task_close()
        return {"state": state, "step": step, "best": best,
                "peak_vram_mb": _peak_vram_mb()}


def _peak_vram_mb():
    """torch 分配器的历史峰值（MiB）；不含 CUDA context。只用于显存调参对照。"""
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
    except Exception:  # noqa: BLE001 —— 记录用，绝不影响训练
        pass
    return None


def run_train(config_path, **overrides) -> dict:
    cfg = TrainConfig.load(config_path)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.validate()
    return Trainer(cfg).run()
