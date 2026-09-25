"""训练配置（JSON）与配置哈希。

::

    {"task": {"factory": "ResNet.tasks:make_task", "kwargs": {...}},
     "out": "/home/jeefy/UniChess/ResNet/runs/stage1_v2",
     "steps": 187578, "accum": 1, "seed": 20260908, "device": "cuda",
     "precision": "bf16", "grad_scaler": false, "clip": 2.0, "nonfinite": "ignore",
     "optimizer": {"lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                   "fused": false},
     "schedule": {"kind": "onecycle", "pct_start": 0.05},
     "torch": {"num_threads": null, "cudnn_benchmark": false, "tf32": false},
     "log_every": 50, "save_every": 2000, "validate_every": 0, "validate_at": [],
     "export": {"best": "best.pt", "final": "final.pt", "every": 0, "every_name": "step_{step:08d}.pt"},
     "runtime": {...}}

- 相对路径（``out``）相对配置文件所在目录解析。
- **配置哈希**只含决定训练结果的字段（去掉 out / log_every / save_every / export / runtime）；
  ``latest.pt`` 里记着它，续训时不符直接拒绝（kit 纪律：结果文件不混口径）。
- ``runtime`` 原样传给任务工厂（``task_factory(**kwargs, runtime=runtime)``），只影响怎么跑
  （worker 数、pin_memory 等），不进哈希。
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

TRAIN_SCHEMA = 1

_PRECISIONS = ("bf16", "fp32")
_NONFINITE = ("ignore", "raise", "skip")


@dataclass
class TrainConfig:
    task: dict
    out: str
    steps: int
    accum: int = 1
    seed: Optional[int] = None
    device: str = "cuda"
    precision: str = "bf16"
    grad_scaler: bool = False
    clip: float = 0.0
    nonfinite: str = "ignore"
    optimizer: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=lambda: {"kind": "constant"})
    torch: dict = field(default_factory=dict)
    log_every: int = 50
    save_every: int = 1000
    validate_every: int = 0
    validate_at: list = field(default_factory=list)
    export: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)

    _NOT_HASHED = ("out", "log_every", "save_every", "export", "runtime")

    @classmethod
    def from_dict(cls, d: dict, base_dir=None) -> "TrainConfig":
        d = copy.deepcopy(d)
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"训练配置有未知字段 {sorted(unknown)}")
        cfg = cls(**d)
        if base_dir is not None and not Path(cfg.out).is_absolute():
            cfg.out = str((Path(base_dir) / cfg.out).resolve())
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path) -> "TrainConfig":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)

    def validate(self) -> None:
        if not isinstance(self.task, dict) or "factory" not in self.task:
            raise ValueError("task 应为 {'factory': '包.模块:函数', 'kwargs': {...}}")
        if set(self.task) - {"factory", "kwargs"}:
            raise ValueError(f"task 有未知字段 {sorted(set(self.task) - {'factory', 'kwargs'})}")
        if self.steps < 0 or self.accum <= 0:
            raise ValueError("steps / accum 必须为正（steps=0 表示由任务的 auto_steps() 决定，"
                             "见 Kit/train/trainer.py）")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision 应为 {_PRECISIONS}")
        if self.nonfinite not in _NONFINITE:
            raise ValueError(f"nonfinite 应为 {_NONFINITE}")
        unknown = set(self.optimizer) - {"lr", "weight_decay", "betas", "eps", "fused"}
        if unknown:
            raise ValueError(f"optimizer 有未知字段 {sorted(unknown)}")
        unknown = set(self.torch) - {"num_threads", "cudnn_benchmark", "tf32", "deterministic",
                                     "cuda_mem_fraction"}
        if unknown:
            raise ValueError(f"torch 有未知字段 {sorted(unknown)}")
        unknown = set(self.export) - {"best", "final", "every", "every_name"}
        if unknown:
            raise ValueError(f"export 有未知字段 {sorted(unknown)}")

    def to_dict(self) -> dict:
        return asdict(self)

    def identity(self) -> dict:
        d = self.to_dict()
        for k in self._NOT_HASHED:
            d.pop(k, None)
        return d

    def hash(self) -> str:
        blob = json.dumps({"schema": TRAIN_SCHEMA, "config": self.identity()}, sort_keys=True,
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
