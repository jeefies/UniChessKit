"""跨进程文件锁（Linux flock / Windows msvcrt）。

锁随持有进程退出由操作系统自动释放，所以「能拿到某个文件的锁」= 「原持有者已经不在了」，
job 存活检测与 GPU 租约的失效回收都靠这一点，不依赖心跳或 pid 复用判断。

Windows 的 msvcrt 锁是强制锁，被锁的字节别的句柄读不了；因此锁的是远离内容的一个字节
（_LOCK_OFFSET），文件正文照常可读。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_LOCK_OFFSET = 1 << 20

if os.name == "nt":
    import msvcrt

    def _lock(fh) -> bool:
        fh.seek(_LOCK_OFFSET)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
        finally:
            fh.seek(0)

    def _unlock(fh) -> None:
        fh.seek(_LOCK_OFFSET)
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            fh.seek(0)
else:
    import fcntl

    def _lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


class FileLock:
    """独占文件锁。``acquire(timeout=0)`` 为非阻塞尝试；锁文件不会被删除（删除会与加锁竞争）。"""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self, timeout: float = 0.0, poll_s: float = 0.05) -> bool:
        if self._fh is not None:
            raise RuntimeError(f"{self.path} 已由本对象持有")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        deadline = time.monotonic() + timeout
        while True:
            if _lock(fh):
                self._fh = fh
                return True
            if time.monotonic() >= deadline:
                fh.close()
                return False
            time.sleep(poll_s)

    def release(self) -> None:
        if self._fh is None:
            return
        _unlock(self._fh)
        self._fh.close()
        self._fh = None

    def __enter__(self):
        if not self.acquire(timeout=60.0):
            raise TimeoutError(f"60 秒内拿不到锁 {self.path}")
        return self

    def __exit__(self, *exc):
        self.release()


def is_locked(path) -> bool:
    """别的进程（或本进程的其他句柄）正持有该文件的锁 → True。文件不存在 → False。"""
    path = Path(path)
    if not path.exists():
        return False
    probe = FileLock(path)
    if probe.acquire(timeout=0):
        probe.release()
        return False
    return True
