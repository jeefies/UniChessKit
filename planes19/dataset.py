"""T/R 训练数据流：分片集合、整批解码数据集、专长过采样池、课程（难度排序）流。

三种取数方式对应旧配方（逐位复刻其顺序与随机数流）：

- **loader**（R stage1、T t20m、T p3）：``BatchShardDataset`` + ``train.data.resumable_loader``，
  第 0 轮顺序与 ``BatchSampler(RandomSampler(ds), drop_last=True)`` 相同。
- **pool**（R iteration46）：``SpecialtyPool`` 按更新号播种 ``default_rng(seed + update)``，
  每次更新取 accum 个 ``mixed`` 批（1/2 全体、1/4 残局、其余升变候选）。
- **curriculum**（T 分阶段专家）：``CurriculumStream`` 按子力数过滤、按块（默认 65536 条）用
  当前模型打难度分、由易到难切批。

全部流都能从检查点续跑：loader 记 ``{seed0, consumed}``，pool 只依赖步数，
curriculum 记当前块（全局下标）与块内位置、读到的分片与未成块的余量。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .records import (RECORD_DTYPE, SELFPLAY_DTYPE, decode_batch, decode_targets,
                      piece_count_bucket, shard_dtype, shard_len)


def list_shards(shard_dir, pattern: str = "*.bin", *, exclude_selfplay: bool = True) -> list:
    """目录下按名字排序的分片。``*.bin`` 也匹配 ``*.sp.bin``，默认把自对弈分片排除。"""
    paths = sorted(Path(shard_dir).glob(pattern))
    if exclude_selfplay:
        paths = [p for p in paths if not p.name.endswith(".sp.bin")]
    if not paths:
        raise FileNotFoundError(f"{shard_dir} 下没有匹配 {pattern} 的分片")
    return paths


class ShardSet:
    """若干同格式分片拼成的连续下标空间。内存映射按进程懒打开（DataLoader worker 各自映射）。"""

    def __init__(self, paths: Sequence):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise FileNotFoundError("分片列表为空")
        dtypes = {shard_dtype(p) for p in self.paths}
        if len(dtypes) != 1:
            raise ValueError("一个 ShardSet 不能混用监督分片与自对弈分片")
        self.dtype = dtypes.pop()
        self.offsets = np.cumsum([0] + [shard_len(p) for p in self.paths])
        self.total = int(self.offsets[-1])
        self._maps: list = [None] * len(self.paths)

    def __len__(self) -> int:
        return self.total

    def __getstate__(self):                         # 传给 worker 时不带已打开的映射
        st = dict(self.__dict__)
        st["_maps"] = [None] * len(self.paths)
        return st

    def shard(self, i: int) -> np.memmap:
        if self._maps[i] is None:
            self._maps[i] = np.memmap(self.paths[i], dtype=self.dtype, mode="r")
        return self._maps[i]

    def gather(self, indices) -> np.ndarray:
        """全局下标 → 连续记录数组（保持下标顺序）。"""
        idx = np.asarray(indices, dtype=np.int64)
        shard = np.searchsorted(self.offsets, idx, side="right") - 1
        out = np.empty(len(idx), dtype=self.dtype)
        for s in np.unique(shard):
            sel = shard == s
            out[sel] = self.shard(int(s))[idx[sel] - self.offsets[s]]
        return out

    def fingerprint(self) -> list:
        return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in self.paths]


class Decoder:
    """记录 → 张量元组 (x, policy, promo, wdl[, bucket])。可 pickle（给 DataLoader worker）。"""

    def __init__(self, *, repair_castling: bool = True, num_buckets: int = 1, q_ratio: float = 0.0,
                 channels_last: bool = False):
        self.repair_castling = repair_castling
        self.num_buckets = int(num_buckets)
        self.q_ratio = float(q_ratio)
        self.channels_last = channels_last

    def __call__(self, recs: np.ndarray) -> tuple:
        import torch

        x = torch.from_numpy(decode_batch(recs))
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        p, pr, w = decode_targets(recs, repair_castling=self.repair_castling, q_ratio=self.q_ratio)
        out = [x, torch.from_numpy(p), torch.from_numpy(pr), torch.from_numpy(w)]
        if self.num_buckets > 1:
            out.append(torch.from_numpy(piece_count_bucket(recs, self.num_buckets)))
        return tuple(out)


class BatchShardDataset:
    """按**整批**下标取样（``DataLoader(batch_size=None)``，采样器给出整批下标）。

    向量化解码一次解 512 条与 1 条耗时相近，逐条取会让 DataLoader 成为瓶颈。
    """

    def __init__(self, shards: ShardSet, decoder: Decoder):
        self.shards, self.decoder = shards, decoder

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        return self.decoder(self.shards.gather(indices))


# ---------------------------------------------------------------- R iteration46 过采样池

ENDGAME_MAX_PIECES = 10
WHITE_ADVANCED = np.uint64(0x00FFFF0000000000)      # 白兵在第 6/7 横线
BLACK_ADVANCED = np.uint64(0x0000000000FFFF00)      # 黑兵在第 2/3 横线


class SpecialtyPool:
    """全体 + 残局（子力 ≤ 10）+ 升变候选（兵已推进到第 6/7 或 2/3 横线）三路采样。

    专长下标每个分片最多随机留 ``cap`` 条（``default_rng(index_seed)`` 的 ``choice``，按分片顺序
    连续抽），结果缓存在 ``cache``（npz）+ 同名 ``.json`` 指纹（路径、大小、mtime）；指纹不符重建。
    """

    def __init__(self, shards: ShardSet, cache=None, *, index_seed: int = 20260909,
                 cap: int = 50000, chunk: int = 262144, log: Callable = print):
        self.shards = shards
        self.total = shards.total
        cache = Path(cache) if cache is not None else None
        expected = json.dumps(shards.fingerprint())
        manifest = cache.with_suffix(".json") if cache is not None else None
        if cache is not None and cache.exists() and manifest.exists() and \
                manifest.read_text() == expected:
            with np.load(cache) as d:
                self.endgame, self.promotion = d["endgame"], d["promotion"]
        else:
            self.endgame, self.promotion = self._index(index_seed, cap, chunk, log)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, endgame=self.endgame, promotion=self.promotion)
                manifest.write_text(expected)
        if not len(self.endgame) or not len(self.promotion):
            raise ValueError("专长采样池为空")

    def _index(self, seed, cap, chunk, log):
        rng = np.random.default_rng(seed)
        groups = [[], []]
        for s in range(len(self.shards.paths)):
            data = self.shards.shard(s)
            base = self.shards.offsets[s]
            selected = [[], []]
            for start in range(0, len(data), chunk):
                r = data[start:start + chunk]
                occ = np.asarray(r["occ_white"] | r["occ_black"]).copy()
                counts = np.unpackbits(occ.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                advanced = (((r["pawns"] & r["occ_white"]) & WHITE_ADVANCED) != 0) | \
                           (((r["pawns"] & r["occ_black"]) & BLACK_ADVANCED) != 0)
                for i, mask in enumerate((counts <= ENDGAME_MAX_PIECES, advanced)):
                    selected[i].append(np.flatnonzero(mask) + start + base)
            for i in range(2):
                idx = np.concatenate(selected[i])
                if len(idx) > cap:
                    idx = rng.choice(idx, cap, replace=False)
                groups[i].append(idx.astype(np.int64))
            log(f"indexed {self.shards.paths[s]}")
        return [np.concatenate(g) for g in groups]

    def indices(self, rng, n: int, mode: str = "mixed") -> np.ndarray:
        if mode == "mixed":
            a, b = n // 2, n // 4
            idx = np.concatenate([rng.integers(self.total, size=a),
                                  rng.choice(self.endgame, size=b),
                                  rng.choice(self.promotion, size=n - a - b)])
            rng.shuffle(idx)
        elif mode == "general":
            idx = rng.integers(self.total, size=n)
        elif mode in ("endgame", "promotion"):
            idx = rng.choice(getattr(self, mode), size=n)
        else:
            raise ValueError(f"未知采样模式 {mode!r}")
        return idx

    def sample(self, rng, n: int, mode: str = "mixed") -> np.ndarray:
        return self.shards.gather(self.indices(rng, n, mode))


# ---------------------------------------------------------------- T 课程流

PIECE_FILTERS = {
    "all":        lambda c: np.ones(len(c), dtype=bool),
    "opening":    lambda c: c >= 24,
    "middlegame": lambda c: (c > 12) & (c < 24),
    "endgame":    lambda c: c <= 12,
}


class CurriculumStream:
    """按分片顺序读出满足子力过滤的记录，攒成 ``chunk_size`` 块；每块用 ``score_fn(recs) → 分数``
    打分后 ``np.argsort``（默认快排，与旧脚本一致）由易到难切成整批（丢弃不满一批的尾巴）。
    读完全部分片时最后一块可以不满 ``chunk_size``；之后从第一个分片重新开始。

    ``state``（原地维护，放进检查点）::

        shard     下一个要读的分片号
        pending   已过滤、未成块的全局下标
        chunk     当前块排好序的全局下标（None = 需要新块）
        pos       当前块里下一批的序号
        chunks    已打分的块数
    """

    def __init__(self, shards: ShardSet, *, piece_filter: str, chunk_size: int, batch_size: int,
                 state: dict):
        if piece_filter not in PIECE_FILTERS:
            raise ValueError(f"未知子力过滤 {piece_filter!r}，可选 {sorted(PIECE_FILTERS)}")
        self.shards = shards
        self.keep = PIECE_FILTERS[piece_filter]
        self.chunk_size, self.batch_size = int(chunk_size), int(batch_size)
        self.state = state
        state.setdefault("shard", 0)
        state.setdefault("pending", np.zeros(0, dtype=np.int64))
        state.setdefault("chunk", None)
        state.setdefault("pos", 0)
        state.setdefault("chunks", 0)

    def _filtered(self, s: int) -> np.ndarray:
        data = self.shards.shard(s)
        counts = np.bitwise_count(data["occ_white"] | data["occ_black"])
        return np.nonzero(self.keep(counts))[0].astype(np.int64) + int(self.shards.offsets[s])

    def _next_chunk(self) -> np.ndarray:
        st = self.state
        n_shards = len(self.shards.paths)
        while len(st["pending"]) < self.chunk_size:
            if st["shard"] >= n_shards:
                if len(st["pending"]):               # 一遍读完：余量单独成块
                    break
                st["shard"] = 0                      # 从头再来
                continue
            idx = self._filtered(st["shard"])
            st["shard"] += 1
            if len(idx):
                st["pending"] = np.concatenate([st["pending"], idx])
        chunk = st["pending"][:self.chunk_size]
        st["pending"] = st["pending"][self.chunk_size:]
        if not len(chunk):
            raise ValueError("课程过滤后没有任何记录")
        return chunk

    def batches(self, score_fn: Callable[[np.ndarray], np.ndarray]):
        """无限产出记录批（numpy 结构化数组）。"""
        st = self.state
        while True:
            if st["chunk"] is None:
                chunk = self._next_chunk()
                scores = score_fn(self.shards.gather(chunk))
                st["chunk"] = chunk[np.argsort(scores)]
                st["pos"] = 0
                st["chunks"] += 1
            chunk = st["chunk"]
            n_batches = len(chunk) // self.batch_size    # 丢掉不满一批的尾巴（与旧脚本一致）
            while st["pos"] < n_batches:
                b = st["pos"]
                st["pos"] = b + 1
                yield self.shards.gather(chunk[b * self.batch_size:(b + 1) * self.batch_size])
            st["chunk"] = None


__all__ = ["BatchShardDataset", "CurriculumStream", "Decoder", "PIECE_FILTERS", "RECORD_DTYPE",
           "SELFPLAY_DTYPE", "ShardSet", "SpecialtyPool", "list_shards"]
