import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import chess

from unichess_kit.jobs import JobHandle
from unichess_kit.pipelines.match import MatchConfig, run_match
from unichess_kit.testing.fakes import make_fake_player_factory

KIT_ROOT = Path(__file__).resolve().parents[1]

FAKE = {"factory": "unichess_kit.testing.fakes:make_fake_player_factory",
        "kwargs": {"simulations": 8}}
RANDOM = {"factory": "unichess_kit.testing.fakes:make_random_player_factory"}


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = str(KIT_ROOT) + os.pathsep + e.get("PYTHONPATH", "")
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def wait_final(handle, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = handle.state()
        if st in ("completed", "error", "stopped", "gpu_busy", "died"):
            assert handle.wait(30), "终态已写但进程未退出"
            return st
        time.sleep(0.1)
    raise AssertionError(f"job 超时未结束：{handle.status()}")


class TestMatchObserver(unittest.TestCase):
    def test_events_match_records(self):
        events = []
        summary = run_match(MatchConfig(pairs=2, max_plies=30, concurrency=3),
                            make_a=make_fake_player_factory("a", salt="1"),
                            make_b=make_fake_player_factory("b", salt="2"),
                            observer=events.append,
                            progress=lambda rec, recs: events.append({"type": "record", **rec}))
        self.assertEqual(summary["games"], 4)
        for rec in (e for e in events if e["type"] == "record"):
            moves = [e["uci"] for e in events if e["type"] == "move" and e["game"] == rec["game"]]
            self.assertEqual(moves, rec["moves"])
            starts = [e for e in events if e["type"] == "game_start" and e["game"] == rec["game"]]
            self.assertEqual(len(starts), 1)
            self.assertEqual(starts[0]["white"], rec["white"])
        mv = next(e for e in events if e["type"] == "move")
        self.assertEqual(mv["source"], "search")
        self.assertIn(mv["side"], ("A", "B"))
        json.dumps(events)                                   # 事件可直接序列化

    def test_external_stop(self):
        done = []
        summary = run_match(MatchConfig(pairs=4, max_plies=10, concurrency=1),
                            make_a=make_fake_player_factory("a"), make_b=make_fake_player_factory("b"),
                            progress=lambda rec, recs: done.append(rec),
                            should_stop=lambda: len(done) >= 3)
        self.assertEqual(summary["games"], 3)


class TestJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lease_dir = str(self.root / "leases")

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, name, job):
        job = {"gpu_mib": 0, "lease_dir": self.lease_dir, **job}
        return JobHandle.submit(self.root / name, job, python=sys.executable, env=env())

    def test_game_job(self):
        h = self.submit("g1", {"kind": "game", "a": FAKE, "b": RANDOM,
                               "names": {"A": "fake", "B": "random"},
                               "game": {"max_plies": 40, "seed": 3, "opening": ["e2e4"]}})
        self.assertEqual(wait_final(h), "completed", (h.dir / "job.log").read_text("utf-8"))
        self.assertFalse(h.alive())
        (rec,) = h.records()
        self.assertEqual(rec["white"], "A")
        self.assertEqual(rec["opening"], ["e2e4"])
        live = h.live()["games"]["0"]
        self.assertTrue(live["done"])
        self.assertEqual(live["moves"], rec["moves"])
        self.assertEqual(len(live["details"]), len(rec["moves"]))
        board = chess.Board()
        for uci in rec["opening"] + rec["moves"]:
            board.push_uci(uci)
        st = h.status()
        self.assertEqual(st["summary"]["result"], rec["result"])
        self.assertEqual(st["games_done"], 1)

    def test_match_job_summary_per_model(self):
        h = self.submit("m1", {"kind": "match", "a": FAKE, "b": RANDOM,
                               "names": {"A": "fake", "B": "random"},
                               "match": {"pairs": 2, "max_plies": 60, "concurrency": 4}})
        self.assertEqual(wait_final(h), "completed", (h.dir / "job.log").read_text("utf-8"))
        recs = h.records()
        self.assertEqual(len(recs), 4)
        s = h.status()["summary"]
        self.assertEqual(s["a"], "fake")
        self.assertEqual(s["a_wins"] + s["b_wins"] + s["draws"], 4)
        self.assertEqual(h.live()["games"], {})             # 完成的局从快照里移除

    def test_bad_spec_reports_error(self):
        h = self.submit("e1", {"kind": "game", "a": {"factory": "no.such.module:f"}, "b": RANDOM})
        self.assertEqual(wait_final(h), "error")
        self.assertIn("no", h.status()["error"])

    def test_gpu_busy_state(self):
        # 不可能满足的预算 + 有 nvidia-smi 时才会拒绝；没有 nvidia-smi 的机器上改为验证登记成功
        import shutil
        h = self.submit("b1", {"kind": "game", "a": RANDOM, "b": RANDOM, "gpu_mib": 10 ** 7,
                               "game": {"max_plies": 4}})
        st = wait_final(h)
        if shutil.which("nvidia-smi"):
            self.assertEqual(st, "gpu_busy")
            self.assertIn("显存不足", h.status()["error"])
        else:
            self.assertEqual(st, "completed")

    @unittest.skipIf(os.name == "nt", "进程组 SIGTERM 仅 POSIX")
    def test_stop(self):
        slow = {"factory": "unichess_kit.testing.fakes:make_slow_player_factory",
                "kwargs": {"delay_s": 0.2}}
        h = self.submit("s1", {"kind": "match", "a": slow, "b": slow,
                               "match": {"pairs": 50, "max_plies": 200, "concurrency": 2}})
        self.assertEqual(h.wait_started(60), "running")
        deadline = time.monotonic() + 30
        while not any(g["moves"] for g in h.live().get("games", {}).values()):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.1)
        h.stop(grace_s=10)
        self.assertEqual(wait_final(h, 20), "stopped")
        self.assertFalse(h.alive())


if __name__ == "__main__":
    unittest.main()
