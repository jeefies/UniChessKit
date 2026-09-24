"""结果溯源：记录跑出这份结果的代码版本与环境。

共享 conda 环境里同时存在多个项目、kit 也可能在任务中途被更新（计划 §2 纰漏 7），
所以每份结果都写明 kit 与各引擎仓库的 git 提交和是否有未提交改动。
"""
from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import __version__

KIT_ROOT = Path(__file__).resolve().parent.parent


def git_state(path) -> Optional[dict]:
    """{'commit', 'dirty'}；不是 git 仓库或没有 git 时返回 None。"""
    if not path:
        return None
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None
        st = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                             "--untracked-files=no"],
                            capture_output=True, text=True, timeout=10)
        return {"commit": sha.stdout.strip(), "dirty": bool(st.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def collect(engine_roots: Optional[dict] = None) -> dict:
    mods = {}
    for name in ("chess", "numpy", "torch"):
        m = sys.modules.get(name)
        if m is not None:
            mods[name] = getattr(m, "__version__", "?")
    return {
        "kit_version": __version__,
        "kit_git": git_state(KIT_ROOT),
        "engines": {k: {"root": str(v), "git": git_state(v)} for k, v in (engine_roots or {}).items()
                    if v},
        "python": sys.version.split()[0],
        "modules": mods,
        "host": platform.node(),
        "argv": list(sys.argv),
        "started_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
