import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path

from Kit.runtime import FileLock, GpuBusyError, GpuLease, is_locked


def smi(total=16000, used=0, procs=None):
    return lambda: {"total": total, "used": used, "procs": dict(procs or {})}


def own(p):
    return [p]


def _hold_lock(path, ready, release):
    lock = FileLock(path)
    lock.acquire(timeout=5)
    ready.set()
    release.wait(10)


class TestFileLock(unittest.TestCase):
    def test_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.lock"
            a, b = FileLock(p), FileLock(p)
            self.assertTrue(a.acquire())
            self.assertTrue(is_locked(p))
            self.assertFalse(b.acquire(timeout=0))
            a.release()
            self.assertFalse(is_locked(p))
            self.assertTrue(b.acquire())
            b.release()

    def test_released_when_process_dies(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.lock"
            ready, release = ctx.Event(), ctx.Event()
            proc = ctx.Process(target=_hold_lock, args=(str(p), ready, release))
            proc.start()
            self.assertTrue(ready.wait(60))
            self.assertTrue(is_locked(p))
            proc.terminate()
            proc.join(10)
            self.assertFalse(is_locked(p))


class TestGpuLease(unittest.TestCase):
    def test_no_nvidia_smi_always_granted(self):
        with tempfile.TemporaryDirectory() as d:
            with GpuLease(10 ** 6, lock_dir=d, query=lambda: None) as lease:
                self.assertTrue(lease.path.exists())
            self.assertEqual(list(Path(d).glob("*.lease")), [])

    def test_refuses_when_training_occupies_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            q = smi(16000, 13000, {999: 13000})          # 训练进程占 13 GB
            with self.assertRaises(GpuBusyError) as cm:
                GpuLease(4096, lock_dir=d, query=q, ancestors=own).acquire()
            self.assertIn("4096", str(cm.exception))
            GpuLease(2048, lock_dir=d, query=q, ancestors=own).acquire().release()

    def test_invisible_process_usage_counts_as_external(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(GpuBusyError):
                GpuLease(4096, lock_dir=d, query=smi(16000, 13000), ancestors=own).acquire()

    def test_leases_count_budget_or_actual_usage(self):
        me = os.getpid()
        with tempfile.TemporaryDirectory() as d:
            first = GpuLease(6000, lock_dir=d, query=smi(16000, 0), ancestors=own).acquire()
            try:
                # 已登记 6000：再要 9600 就超了（16000 - 512 - 6000 = 9488）
                with self.assertRaises(GpuBusyError):
                    GpuLease(9600, lock_dir=d, query=smi(16000, 0), ancestors=own).acquire()
                # 第一个租约的子进程（祖先链含本进程）实际用了 8000 > 预算 6000：按实际计，且不算外部占用
                child = {4242: 8000}

                def anc(p):
                    return [p, me] if p == 4242 else [p]
                with self.assertRaises(GpuBusyError):
                    GpuLease(7600, lock_dir=d, query=smi(16000, 8000, child),
                             ancestors=anc).acquire()
                GpuLease(7000, lock_dir=d, query=smi(16000, 8000, child),
                         ancestors=anc).acquire().release()
            finally:
                first.release()
            GpuLease(9000, lock_dir=d, query=smi(16000, 0), ancestors=own).acquire().release()

    def test_stale_lease_file_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as d:
            stale = Path(d) / "123-dead.lease"
            stale.write_text('{"pid": 123, "budget_mib": 15000}')
            GpuLease(8000, lock_dir=d, query=smi(16000, 0), ancestors=own).acquire().release()
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
