"""GPU 显存租约：GPU 任务启动前声明显存预算，不够就拒绝，而不是跑到一半 OOM 或挤垮训练。

同一张卡上有 Web 服务、评测 job、训练（S 的自对弈 / 训练器）共用。每个 job 在加载模型之前
申请一个 ``GpuLease(budget_mib)``：

- 租约 = 租约目录下一个被持有者加锁的文件（内容是 pid / 名字 / 预算）。持有进程退出（含被 kill）
  后锁由操作系统释放，下一个申请者发现锁能拿到就把它当失效租约清掉，不会有「死租约」卡住。
- 判定（在注册锁内完成，申请之间不会互相看不见）::

      可用 = 总显存 − 保留 − 非租约进程已用 − Σ 活着的租约 max(预算, 实际已用)

  「非租约进程」用 nvidia-smi 的逐进程显存统计；租约进程及其子进程（WorkerPool 的 spawn 子进程）
  按祖先链归到对应租约，已用超过预算时按实际计。可用 ≥ 预算才批准。
- 没有 nvidia-smi（CPU 机器、Windows 本机测试）时不做显存判定，租约照常登记。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from .locks import FileLock

DEFAULT_LEASE_DIR = Path(os.environ.get("UNICHESS_GPU_LEASE_DIR",
                                        Path.home() / ".cache" / "unichess" / "gpu_leases"))
DEFAULT_RESERVE_MIB = 512        # 给驱动 / 显示 / 碎片留的余量


class GpuBusyError(RuntimeError):
    """显存不够，租约被拒绝。"""


def _smi(args: list) -> list:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def query_nvidia_smi(gpu_index: int = 0) -> Optional[dict]:
    """{"total": MiB, "used": MiB, "procs": {pid: MiB}}；拿不到返回 None。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        rows = _smi(["--query-gpu=index,uuid,memory.total,memory.used"])
        row = next(r for r in rows if int(r[0]) == gpu_index)
        gpu_uuid, total, used = row[1], int(row[2]), int(row[3])
        procs: dict = {}
        for r in _smi(["--query-compute-apps=gpu_uuid,pid,used_memory"]):
            if r[0] == gpu_uuid and r[2].isdigit():
                procs[int(r[1])] = procs.get(int(r[1]), 0) + int(r[2])
        return {"total": total, "used": used, "procs": procs}
    except Exception:
        return None


def _parent_pid(pid: int) -> Optional[int]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # 第 2 字段 comm 可能含空格和括号：从最后一个 ')' 之后再切
    fields = stat[stat.rindex(")") + 2:].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def _ancestors(pid: int, limit: int = 64) -> list:
    chain = [pid]
    while len(chain) < limit:
        parent = _parent_pid(chain[-1])
        if not parent or parent in chain:
            break
        chain.append(parent)
    return chain


class GpuLease:
    """``with GpuLease(4096, name="match T vs R"):`` 包住整个 GPU 任务。

    ``query`` 可注入（测试用），返回格式同 ``query_nvidia_smi``；``ancestors`` 同理。
    """

    def __init__(self, budget_mib: int, name: str = "", lock_dir=None, *,
                 reserve_mib: int = DEFAULT_RESERVE_MIB,
                 query: Callable[[], Optional[dict]] = query_nvidia_smi,
                 ancestors: Callable[[int], list] = _ancestors):
        if budget_mib < 0:
            raise ValueError("budget_mib 不能为负")
        self.budget_mib = int(budget_mib)
        self.name = name
        self.lock_dir = Path(lock_dir) if lock_dir is not None else DEFAULT_LEASE_DIR
        self.reserve_mib = reserve_mib
        self.query = query
        self.ancestors = ancestors
        self._lock: Optional[FileLock] = None
        self.path: Optional[Path] = None
        self.last_check: dict = {}

    # ---------- 租约登记 ----------

    def _live_leases(self) -> list:
        """扫描租约目录：活着的返回 [{pid, name, budget_mib, ...}]，失效的顺手删掉。"""
        live = []
        for path in sorted(self.lock_dir.glob("*.lease")):
            probe = FileLock(path)
            if probe.acquire(timeout=0):          # 能拿到锁 = 持有者已退出
                probe.release()
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            try:
                info = json.loads(path.read_text(encoding="utf-8") or "{}")
            except (OSError, ValueError):
                info = {}
            info.setdefault("budget_mib", 0)
            info["path"] = str(path)
            live.append(info)
        return live

    def check(self) -> dict:
        """只计算不登记：返回 {"ok", "available", "budget", "leases", ...}（调用方须持注册锁）。"""
        leases = self._live_leases()
        info = self.query()
        result = {"ok": True, "budget": self.budget_mib, "leases": leases, "available": None}
        if info is None:
            return result
        lease_pids = {int(l["pid"]): l for l in leases if l.get("pid")}
        lease_used = {pid: 0 for pid in lease_pids}
        others = 0
        for pid, used in info["procs"].items():
            owner = next((a for a in self.ancestors(pid) if a in lease_pids), None)
            if owner is None:
                others += used
            else:
                lease_used[owner] += used
        if not info["procs"]:
            # 逐进程统计在容器 / 权限受限时可能为空：把总已用全算作外部占用（宁可保守）
            others = info["used"]
        leased = sum(max(int(l["budget_mib"]), lease_used.get(pid, 0))
                     for pid, l in lease_pids.items())
        available = info["total"] - self.reserve_mib - others - leased
        result.update(ok=available >= self.budget_mib, available=available,
                      total=info["total"], others_mib=others, leased_mib=leased)
        return result

    def acquire(self) -> "GpuLease":
        if self._lock is not None:
            raise RuntimeError("租约已持有")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock_dir / "registry.lock"):
            check = self.check()
            self.last_check = check
            if not check["ok"]:
                holders = ", ".join(f"{l.get('name') or '?'}(pid {l.get('pid')}, "
                                    f"{l.get('budget_mib')} MiB)" for l in check["leases"]) or "无"
                raise GpuBusyError(
                    f"GPU 显存不足：需要 {self.budget_mib} MiB，可用约 {check['available']} MiB"
                    f"（非租约进程占用 {check.get('others_mib')} MiB，现有租约：{holders}）。"
                    "可能有训练任务在跑，稍后再试")
            path = self.lock_dir / f"{os.getpid()}-{uuid.uuid4().hex[:8]}.lease"
            lock = FileLock(path)
            if not lock.acquire(timeout=0):
                raise RuntimeError(f"无法锁定新建的租约文件 {path}")
            lock._fh.seek(0)
            lock._fh.truncate()
            lock._fh.write(json.dumps({"pid": os.getpid(), "name": self.name,
                                       "budget_mib": self.budget_mib,
                                       "started": time.time()}).encode())
            lock._fh.flush()
            self._lock, self.path = lock, path
        return self

    def release(self) -> None:
        if self._lock is None:
            return
        path = self.path
        self._lock.release()
        self._lock, self.path = None, None
        try:
            path.unlink()
        except OSError:
            pass                                 # 下一个申请者会当失效租约清理

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
