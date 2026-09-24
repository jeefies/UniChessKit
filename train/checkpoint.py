"""检查点：原子写入、RNG 状态、优化器状态搬回设备。"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_atomic(obj, path) -> None:
    """先写临时文件再 ``os.replace``：写到一半被杀也不会留下半个 latest.pt。"""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                # 同时播种所有 CUDA 设备


def optimizer_state_like_params(opt) -> None:
    """``Optimizer.load_state_dict`` 只搬设备/类型、不保留内存格式；fused AdamW 要求状态与参数
    步长一致（channels_last 的卷积权重）。与参数同形的状态张量按参数的布局重建。"""
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p)
            if not state:
                continue
            for k, v in list(state.items()):
                if torch.is_tensor(v) and v.shape == p.shape and v.stride() != p.stride():
                    state[k] = torch.empty_like(p, dtype=v.dtype).copy_(v)
