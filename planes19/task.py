"""T/R 通用监督 / 自对弈训练任务 ``Planes19Task``（实现 ``Kit.train.TrainTask``）。

引擎仓库只提供**模型适配器**（``model.factory`` 返回的对象）::

    build()                          → nn.Module（含加载底座权重、channels_last 等）
    forward(model, x, bucket=None, *, mlh=False)
                                     → (policy_logits, promo_logits, wdl_logits[, mlh])
    export(model, step)              → 引擎可直接加载的检查点字典
    trainable(model)（可选）         → 参与训练的子模块（默认整个模型；分阶段专家返回该专家）
    num_buckets（可选，默认 1）       R 的子力分桶头
    channels_last（可选，默认 False） 输入是否转 channels_last

任务配置（``task.kwargs``）::

    {"model": {"factory": "ResNet.train:make_model", "kwargs": {...}},
     "data": {"kind": "loader" | "pool" | "curriculum" | "mix", ...},
     "loss": {"kind": "r_stage1" | "r_iter" | "t_chess", ...},
     "validation": {"kind": "pool" | "sequential", ...}  （可选）}

数据来源 ``shards``：``{"dir": ..., "pattern": "*.bin", "slice": [start, stop]}``，
``slice`` 按 Python 切片语义作用于排好序的分片列表（``[null, -4]`` = 留出最后 4 个）。

各 kind 的参数见 ``batches`` 下各 ``_*_batches``、``loss``、``validate``。
"""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..registry import load_object
from ..train.data import resumable_loader
from .dataset import (BatchShardDataset, CurriculumStream, Decoder, ShardSet, SpecialtyPool,
                      list_shards)
from .losses import ChessLoss, legal_from_to_mask, mlh_target, r_iter_loss, r_stage1_loss

DATA_KINDS = ("loader", "pool", "curriculum", "mix")
LOSS_KINDS = ("r_stage1", "r_iter", "t_chess")


def resolve_shards(spec: dict) -> list:
    spec = dict(spec)
    files = spec.pop("files", None)
    if "dir" in spec:
        paths = list_shards(spec.pop("dir"), spec.pop("pattern", "*.bin"),
                            exclude_selfplay=spec.pop("exclude_selfplay", True))
        sl = spec.pop("slice", None)
        if sl is not None:
            paths = paths[slice(*sl)]
    else:
        paths = None                              # 只给 files：换代循环按代给出
    if spec:
        raise ValueError(f"shards 有未知字段 {sorted(spec)}")
    if files is not None:                          # 显式列表（优先于目录枚举）
        paths = [Path(f) for f in files]
    if not paths:
        raise FileNotFoundError("没有分片（给 dir 时按目录枚举/切片，给 files 时按显式列表）")
    return paths


def _check_keys(d: dict, allowed, what: str) -> None:
    unknown = set(d) - set(allowed)
    if unknown:
        raise ValueError(f"{what} 有未知字段 {sorted(unknown)}")


class Planes19Task:
    def __init__(self, *, model: dict, data: dict, loss: dict, validation: dict = None,
                 runtime: dict = None):
        self.runtime = dict(runtime or {})
        factory = load_object(model["factory"])
        self.adapter = factory(**model.get("kwargs", {}))
        self.data_cfg = dict(data)
        self.loss_cfg = dict(loss)
        self.val_cfg = dict(validation) if validation else None
        if self.data_cfg.get("kind") not in DATA_KINDS:
            raise ValueError(f"data.kind 应为 {DATA_KINDS}")
        if self.loss_cfg.get("kind") not in LOSS_KINDS:
            raise ValueError(f"loss.kind 应为 {LOSS_KINDS}")
        _check_keys(self.loss_cfg, ("kind", "legal_mask", "check_leak", "value_weight",
                                    "promo_weight", "policy_weight", "wdl_weight", "mlh_weight",
                                    "mlh", "policy_loss_type"), "loss")
        self.num_buckets = int(getattr(self.adapter, "num_buckets", 1))
        self.channels_last = bool(getattr(self.adapter, "channels_last", False))
        self.state: dict = {}
        self._executor = None
        if self.loss_cfg["kind"] == "t_chess":
            lc = self.loss_cfg
            self._chess_loss = ChessLoss(lc.get("policy_weight", 1.0), lc.get("promo_weight", 0.1),
                                         lc.get("wdl_weight", 1.0), lc.get("mlh_weight", 0.05),
                                         policy_loss_type=lc.get("policy_loss_type",
                                                                "cross_entropy"))
        self._val_pool = None

    # ------------------------------------------------------------ TrainTask
    def build_model(self):
        return self.adapter.build()

    def trainable(self, model):
        fn = getattr(self.adapter, "trainable", None)
        return fn(model) if fn is not None else model

    def param_groups(self, model):
        sub = self.trainable(model)
        if sub is not model:                       # 冻结其余部分
            keep = {id(p) for p in sub.parameters()}
            for p in model.parameters():
                if id(p) not in keep:
                    p.requires_grad_(False)
        return [{"params": list(sub.parameters())}]

    def state_dict(self) -> dict:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.state.items()}

    def load_state_dict(self, st: dict) -> None:
        self.state = dict(st or {})

    def export(self, model, step: int) -> dict:
        return self.adapter.export(model, step)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # ------------------------------------------------------------ 数据
    def _decoder(self, d: dict) -> Decoder:
        return Decoder(repair_castling=d.get("repair_castling", True), num_buckets=self.num_buckets,
                       q_ratio=d.get("q_ratio", 0.0), channels_last=self.channels_last)

    def _to(self, batch, device):
        return tuple(t.to(device, non_blocking=True) for t in batch)

    def batches(self, ctx):
        d = dict(self.data_cfg)
        kind = d.pop("kind")
        device = ctx.device
        if kind == "loader":
            _check_keys(d, ("shards", "batch_size", "shuffle", "repair_castling", "q_ratio"),
                        "data(loader)")
            ds = BatchShardDataset(ShardSet(resolve_shards(d["shards"])), self._decoder(d))
            st = self.state.setdefault("loader", {})
            stream = resumable_loader(ds, int(d["batch_size"]), st,
                                      num_workers=int(self.runtime.get("num_workers", 4)),
                                      pin_memory=bool(self.runtime.get("pin_memory",
                                                                       device.type == "cuda")),
                                      shuffle=d.get("shuffle", True))
            for batch in stream:
                yield self._to(batch, device)
        elif kind == "pool":
            yield from self._pool_batches(d, ctx)
        elif kind == "curriculum":
            yield from self._curriculum_batches(d, ctx)
        else:
            yield from self._mix_batches(d, ctx)

    def _pool_batches(self, d, ctx):
        """R iteration46：第 u 次更新用 ``default_rng(seed + u)`` 连续取 accum 个 mixed 批；
        后台线程预取下一次更新（结果只依赖更新号，与预取无关）。"""
        _check_keys(d, ("shards", "batch_size", "seed", "index_seed", "cap", "cache",
                        "repair_castling", "mode"), "data(pool)")
        cache = d.get("cache")
        cache = (Path(ctx.out) / cache) if cache and not Path(cache).is_absolute() else cache
        pool = SpecialtyPool(ShardSet(resolve_shards(d["shards"])), cache,
                             index_seed=d.get("index_seed", 20260909), cap=d.get("cap", 50000))
        dec = self._decoder(d)
        seed, bs, mode = int(d.get("seed", 20260909)), int(d["batch_size"]), d.get("mode", "mixed")
        accum = ctx.cfg.accum
        pin = bool(self.runtime.get("pin_memory", ctx.device.type == "cuda"))

        def prepare(update):
            rng = np.random.default_rng(seed + update)
            out = []
            for _ in range(accum):
                b = dec(pool.sample(rng, bs, mode))
                out.append(tuple(t.pin_memory() for t in b) if pin else b)
            return out

        self._executor = ThreadPoolExecutor(max_workers=1)
        update = ctx.step
        fut = self._executor.submit(prepare, update)
        while True:
            micro = fut.result()
            fut = self._executor.submit(prepare, update + 1)
            for b in micro:
                yield self._to(b, ctx.device)
            update += 1

    def _amp(self, ctx):
        return (torch.autocast(ctx.device.type, dtype=torch.bfloat16)
                if ctx.cfg.precision == "bf16" and ctx.device.type == "cuda"
                else contextlib.nullcontext())

    def _curriculum_batches(self, d, ctx):
        """T 分阶段专家课程：块内按 ``CE_policy + λ·CE_wdl``（当前模型，eval 模式）由易到难。"""
        _check_keys(d, ("shards", "batch_size", "filter", "chunk_size", "score_microbatch",
                        "score_lambda", "repair_castling"), "data(curriculum)")
        st = self.state.setdefault("curriculum", {})
        stream = CurriculumStream(ShardSet(resolve_shards(d["shards"])), piece_filter=d["filter"],
                                  chunk_size=int(d.get("chunk_size", 65536)),
                                  batch_size=int(d["batch_size"]), state=st)
        dec = Decoder(repair_castling=d.get("repair_castling", True), num_buckets=self.num_buckets)
        micro = int(d.get("score_microbatch", 2048))
        lam = float(d.get("score_lambda", 1.0))
        model, device, adapter = ctx.model, ctx.device, self.adapter

        @torch.no_grad()
        def score(recs):
            model.eval()
            scores = np.zeros(len(recs), dtype=np.float32)
            for s in range(0, len(recs), micro):
                b = self._to(dec(recs[s:s + micro]), device)
                x, p_t, w_t = b[0], b[1], b[3]
                with self._amp(ctx):
                    out = adapter.forward(model, x, b[4] if len(b) > 4 else None)
                    l_p = F.cross_entropy(out[0], p_t, reduction="none")
                    l_w = F.cross_entropy(out[2], w_t, reduction="none")
                scores[s:s + micro] = (l_p + lam * l_w).float().cpu().numpy()
            model.train()
            return scores

        full = self._decoder(d)
        for recs in stream.batches(score):
            yield self._to(full(recs), device)

    def _mix_batches(self, d, ctx):
        """多来源按比例拼批（自对弈 + 监督）：每个来源均匀随机抽样，第 u 次更新的第 k 个微批用
        ``default_rng([seed, u, k])``，只依赖步数，可续跑。各来源 ``{"shards", "weight", "q_ratio"}``。"""
        _check_keys(d, ("sources", "batch_size", "seed"), "data(mix)")
        bs = int(d["batch_size"])
        seed = int(d.get("seed", 0))
        sources = []
        weights = np.array([float(s["weight"]) for s in d["sources"]])
        sizes = np.floor(weights / weights.sum() * bs).astype(int)
        sizes[0] += bs - sizes.sum()
        for s, n in zip(d["sources"], sizes):
            _check_keys(s, ("shards", "weight", "q_ratio", "repair_castling"), "data(mix).source")
            sources.append((ShardSet(resolve_shards(s["shards"])), self._decoder(s), int(n)))
        update = ctx.step
        while True:
            for k in range(ctx.cfg.accum):
                rng = np.random.default_rng([seed, update, k])
                parts = [dec(shards.gather(rng.integers(len(shards), size=n)))
                         for shards, dec, n in sources if n > 0]
                batch = tuple(torch.cat(ts) for ts in zip(*parts))
                perm = torch.from_numpy(rng.permutation(bs))
                yield self._to(tuple(t[perm] for t in batch), ctx.device)
            update += 1

    # ------------------------------------------------------------ 损失
    def _forward(self, model, batch, mlh=False):
        bucket = batch[4] if len(batch) > 4 else None
        return self.adapter.forward(model, batch[0], bucket, mlh=mlh)

    def loss(self, model, batch, step: int):
        lc = self.loss_cfg
        kind = lc["kind"]
        x, p_t, pr_t, w_t = batch[:4]
        if kind == "t_chess":
            use_mlh = bool(lc.get("mlh", False))
            out = self._forward(model, batch, mlh=use_mlh)
            if use_mlh:
                res = self._chess_loss(out[0], out[1], out[2], p_t, pr_t, w_t,
                                       mlh_logits=out[3], mlh_target=mlh_target(x, w_t))
            else:
                res = self._chess_loss(out[0], out[1], out[2], p_t, pr_t, w_t)
            return res.total_loss, res.parts
        legal = legal_from_to_mask(x) if lc.get("legal_mask", kind == "r_stage1") else None
        if legal is not None and step == 0 and lc.get("check_leak", True):
            leaked = int(((p_t > 0) & ~legal).sum().item())
            if leaked:
                raise RuntimeError(f"合法着法掩码误杀了 {leaked} 个带标签的着法——"
                                   f"掩码规则与数据编码约定对不上")
        out = self._forward(model, batch)
        if kind == "r_stage1":
            loss, parts = r_stage1_loss(out[0], out[1], out[2], p_t, pr_t, w_t, legal=legal,
                                        value_weight=lc.get("value_weight", 1.0),
                                        promo_weight=lc.get("promo_weight", 0.1))
            parts = {k: v for k, v in parts.items() if k != "has_policy"}
            return loss, parts
        return r_iter_loss(out[0], out[1], out[2], p_t, pr_t, w_t, legal=legal)

    # ------------------------------------------------------------ 验证
    @torch.no_grad()
    def validate(self, model, ctx) -> dict:
        if not self.val_cfg:
            return {}
        v = dict(self.val_cfg)
        kind = v.pop("kind")
        device = ctx.device
        if kind == "pool":
            # R iteration46：每种模式各用 default_rng(seed) 取 batches × batch_size，score = 各模式均值
            _check_keys(v, ("shards", "modes", "batches", "batch_size", "seed", "cache",
                            "repair_castling"), "validation(pool)")
            if self._val_pool is None:
                cache = v.get("cache")
                cache = (Path(ctx.out) / cache) if cache and not Path(cache).is_absolute() else cache
                self._val_pool = SpecialtyPool(ShardSet(resolve_shards(v["shards"])), cache)
            dec = self._decoder(v)
            res = {}
            for mode in v.get("modes", ["general", "endgame", "promotion"]):
                rng = np.random.default_rng(v.get("seed", 991))
                losses = []
                for _ in range(int(v.get("batches", 8))):
                    b = self._to(dec(self._val_pool.sample(rng, int(v.get("batch_size", 128)),
                                                           mode)), device)
                    with self._amp(ctx):
                        loss, _ = self.loss(model, b, -1)
                    losses.append(loss.item())
                res[mode] = float(np.mean(losses))
            res["score"] = float(np.mean(list(res.values())))
            return res
        if kind == "sequential":
            _check_keys(v, ("shards", "batches", "batch_size", "repair_castling"),
                        "validation(sequential)")
            shards = ShardSet(resolve_shards(v["shards"]))
            dec = self._decoder(v)
            bs, nb = int(v.get("batch_size", 1024)), int(v.get("batches", 50))
            sums: dict = {}
            n = 0
            for i in range(min(nb, len(shards) // bs)):
                b = self._to(dec(shards.gather(np.arange(i * bs, (i + 1) * bs))), device)
                with self._amp(ctx):
                    loss, parts = self.loss(model, b, -1)
                sums["loss"] = sums.get("loss", 0.0) + loss.item()
                for k, t in parts.items():
                    if torch.is_tensor(t) and t.ndim == 0:
                        sums[k] = sums.get(k, 0.0) + t.item()
                n += 1
            res = {k: s / max(n, 1) for k, s in sums.items()}
            res["score"] = res.get("loss", float("inf"))
            return res
        raise ValueError(f"未知 validation.kind {kind!r}")


def make_task(runtime=None, **kwargs) -> Planes19Task:
    """``task.factory`` 入口：``"Kit.planes19.task:make_task"``。"""
    return Planes19Task(runtime=runtime, **kwargs)
