"""从兄弟仓库（R / T）隔离加载模块，用于对照测试。

默认找 kit 同级目录的 ResNet / Transformer / SSM，可用环境变量 UNICHESS_R_ROOT / UNICHESS_T_ROOT /
UNICHESS_S_ROOT 覆盖。S 的包名就是 stateseq（load("S", "stateseq.gumbel")）。
改名前（core.encoding）与改名后（unichess_r.core.encoding）的模块路径都支持。
加载完即把引擎的顶层包从 sys.modules 清掉，避免 R/T 同名包互相污染（正是 _import_isolated 想解决的问题）。
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parent.parent

ROOTS = {
    "R": Path(os.environ.get("UNICHESS_R_ROOT", KIT_ROOT.parent / "ResNet")),
    "T": Path(os.environ.get("UNICHESS_T_ROOT", KIT_ROOT.parent / "Transformer")),
    "S": Path(os.environ.get("UNICHESS_S_ROOT", KIT_ROOT.parent / "SSM")),
}
PACKAGES = {"R": "unichess_r", "T": "unichess_t", "S": "stateseq"}


def load(which: str, *relative: str):
    """load("R", "core.encoding", "search.mcts") → 模块元组；仓库不存在返回 None。"""
    root = ROOTS[which]
    if not root.is_dir():
        return None
    pkg = PACKAGES[which]
    # 只有搬进包里的子包才加前缀（R 的 eval/ 等脚本目录仍在仓库根）
    names = [f"{pkg}.{r}" if (root / pkg / r.split(".")[0]).is_dir() else r for r in relative]
    tops = {n.split(".")[0] for n in names}
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in tops}
    sys.path.insert(0, str(root))
    try:
        return tuple(importlib.import_module(n) for n in names)
    finally:
        sys.path[:] = saved_path
        for k in list(sys.modules):
            if k.split(".")[0] in tops:
                del sys.modules[k]
        sys.modules.update(saved_mods)
