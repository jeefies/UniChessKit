"""可续跑的随机批采样器。

``ResumableBatchSampler(n, batch)`` 产出的第 0 轮批序列与 torch 的
``BatchSampler(RandomSampler(ds), batch, drop_last=True)`` **逐位相同**：
种子同样在迭代开始时从全局 torch RNG 抽取（``torch.empty((), int64).random_()``），
再用独立 ``torch.Generator`` 做 ``randperm``。旧 R stage1 / T t20m / T p3 的数据顺序由此对齐。

与 torch 的差别（为了续训逐位一致）：
- 一次迭代覆盖所有轮次（无限流）；第 e 轮用同一个生成器的第 e 次 ``randperm``，
  不再每轮从全局 RNG 重新抽种子（旧脚本第 1 轮起的顺序因此不同，只影响超过一轮的训练）。
- ``seed0`` 与 ``start``（已消费的批数）可以从检查点恢复；恢复时重放前面各轮的 ``randperm``。

已消费批数由消费方计数（DataLoader 的 worker 会预取，采样器自身的进度不等于已训练的批数）。
"""
from __future__ import annotations

from typing import Iterator, Optional

import torch


def draw_torch_seed() -> int:
    """与 ``RandomSampler.__iter__``（generator=None）相同的种子抽取。"""
    return int(torch.empty((), dtype=torch.int64).random_().item())


class ResumableBatchSampler:
    def __init__(self, n: int, batch_size: int, *, seed0: Optional[int] = None, start: int = 0,
                 shuffle: bool = True):
        if n < batch_size:
            raise ValueError(f"样本数 {n} 小于批大小 {batch_size}")
        self.n, self.batch_size, self.shuffle = int(n), int(batch_size), shuffle
        self.seed0 = seed0
        self.start = int(start)
        self.per_epoch = self.n // self.batch_size          # drop_last

    def __len__(self) -> int:                              # DataLoader 不应依赖它（无限流）
        return self.per_epoch

    def position(self, consumed: int) -> tuple:
        return divmod(int(consumed), self.per_epoch)

    def __iter__(self) -> Iterator[list]:
        if self.shuffle and self.seed0 is None:
            self.seed0 = draw_torch_seed()
        gen = None
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed0)
        epoch, offset = self.position(self.start)
        for _ in range(epoch):                             # 续训：重放已完成轮次的 randperm
            if gen is not None:
                torch.randperm(self.n, generator=gen)
        while True:
            if gen is not None:
                order = torch.randperm(self.n, generator=gen).tolist()
            else:
                order = list(range(self.n))
            for b in range(offset, self.per_epoch):
                yield order[b * self.batch_size:(b + 1) * self.batch_size]
            offset = 0


def resumable_loader(dataset, batch_size: int, state: dict, *, num_workers: int = 4,
                     pin_memory: bool = False, shuffle: bool = True):
    """DataLoader（batch_size=None，采样器给整批下标）的无限批流；``state`` 原地维护
    ``{"seed0", "consumed"}``，由调用方放进检查点。"""
    from torch.utils.data import DataLoader

    sampler = ResumableBatchSampler(len(dataset), batch_size, seed0=state.get("seed0"),
                                    start=state.get("consumed", 0), shuffle=shuffle)
    loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=num_workers,
                        pin_memory=pin_memory, persistent_workers=False)
    state.setdefault("consumed", 0)
    for batch in loader:
        state["seed0"] = sampler.seed0
        state["consumed"] += 1
        yield batch
