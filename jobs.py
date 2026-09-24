"""对局 job：独立进程跑一次对局任务，持有 GPU 租约，通过 job 目录里的文件对外报告。

Server（或任何调用方）把任务写成 ``<job_dir>/job.json``，再启动
``python -m Kit.jobs <job_dir>``（PYTHONPATH 前置 import 根）；之后只读文件，不与进程通信：

    job.json        任务（调用方写）
    lock            job 进程存活期间持有的文件锁 —— 能拿到锁 = 进程已退出（被 kill 也成立）
    status.json     {"state": starting|running|completed|error|stopped|gpu_busy, pid, 时间, error,
                     games_done, games_planned, summary}；原子替换写入
    live.json       正在进行的对局快照（着法、每步耗时 / 来源 / 评估）；节流原子写入
    results.jsonl   每局结果（kind=match 为 pipelines.match 的 JSONL；kind=game 为表头 + 一局）
    job.log         进程 stdout/stderr（调用方重定向）

job.json::

    {"kind": "match" | "game",
     "a": EngineSpec, "b": EngineSpec, "names": {"A": "...", "B": "..."},
     "match": MatchConfig 字段,                        # kind=match
     "game": {"max_plies": 400, "seed": 0, "opening": [], "fen": None},  # kind=game：一局，A 执白
     "gpu_mib": 4096}                                   # 0 = 不申请租约（CPU 引擎）

停止：向进程组发 SIGTERM（``JobHandle.stop``），状态记为 stopped；已完成的局都已入库。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from . import IMPORT_ROOT

STATES_FINAL = ("completed", "error", "stopped", "gpu_busy")


class JobStopped(BaseException):
    """收到 SIGTERM：从主线程任意位置跳出（BaseException，不被引擎代码的 except Exception 吞掉）。"""


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows：读者正打开目标文件
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    for _ in range(5):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (ValueError, PermissionError):
            time.sleep(0.02)
    return default


# ====================================================================== job 进程侧

class _Status:
    def __init__(self, job_dir: Path):
        self.path = job_dir / "status.json"
        self.data = {"state": "starting", "pid": os.getpid(), "started_at": time.time(),
                     "updated_at": time.time(), "error": None, "games_done": 0,
                     "games_planned": None, "summary": None}
        self.lock = threading.Lock()
        self.flush()

    def update(self, **kw) -> None:
        with self.lock:
            self.data.update(kw, updated_at=time.time())
            self.flush()

    def flush(self) -> None:
        write_json_atomic(self.path, self.data)


class _Live:
    """正在进行的对局快照。observer 在对局线程里调用，后台线程每 interval 秒落盘一次。"""

    def __init__(self, job_dir: Path, keep_finished: bool, interval: float = 0.2):
        self.path = job_dir / "live.json"
        self.keep_finished = keep_finished
        self.interval = interval
        self.games: dict = {}
        self.lock = threading.Lock()
        self.dirty = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def on_event(self, ev: dict) -> None:
        with self.lock:
            g = str(ev["game"])
            if ev["type"] == "game_start":
                self.games[g] = {"game": ev["game"], "pair": ev.get("pair"), "white": ev["white"],
                                 "opening": ev["opening"], "fen": ev.get("fen"),
                                 "moves": [], "details": [],
                                 "started_at": time.time(), "done": False}
            elif ev["type"] == "move" and g in self.games:
                game = self.games[g]
                game["moves"].append(ev["uci"])
                game["details"].append({k: ev.get(k) for k in ("side", "source", "ms", "info")})
            self.dirty = True

    def on_record(self, record: dict) -> None:
        with self.lock:
            g = str(record["game"])
            if self.keep_finished and g in self.games:
                self.games[g].update(done=True, result=record["result"],
                                     termination=record["termination"])
            else:
                self.games.pop(g, None)
            self.dirty = True

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.flush()

    def flush(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            snapshot = {"updated_at": time.time(), "games": self.games}
            write_json_atomic(self.path, snapshot)
            self.dirty = False

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.flush()


def _run_game(job: dict, job_dir: Path, live: _Live, status: _Status) -> dict:
    """kind=game：A 执白、B 执黑下一局。"""
    from .pipelines.match import GameTask, _seed, play_game
    from .registry import EngineSpec, build_player_factory
    from .rules.referee import StandardReferee
    from .runtime.batcher import Batcher, run_sync
    from .api.types import SearchBudget

    g = job.get("game", {})
    seed = int(g.get("seed", 0))
    spec_a, spec_b = EngineSpec.from_dict(job["a"]), EngineSpec.from_dict(job["b"])
    make_a = build_player_factory(spec_a)
    make_b = build_player_factory(spec_b)
    task = GameTask(game=0, pair=0, a_is_white=(g.get("a_color", "white") == "white"),
                    opening=tuple(g.get("opening", ())), seed_a=_seed(seed, 0, "A"),
                    seed_b=_seed(seed, 0, "B"), fen=g.get("fen") or None)
    status.update(state="running", games_planned=1)
    record = run_sync(play_game(task, make_a(), make_b(),
                                StandardReferee(max_plies=int(g.get("max_plies", 400))),
                                SearchBudget(), live.on_event), Batcher())
    live.on_record(record)
    with open(job_dir / "results.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        header = {"type": "header", "kind": "game", "names": job.get("names", {}),
                  "players": {"A": job["a"], "B": job["b"]}}
        for obj in (header, record):
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
    status.update(games_done=1)
    return {"a_score": record["a_score"], "result": record["result"],
            "termination": record["termination"], "plies": record["plies"]}


def _run_match(job: dict, job_dir: Path, live: _Live, status: _Status) -> dict:
    from .pipelines.match import MatchConfig, run_match
    from .registry import EngineSpec

    cfg = MatchConfig.from_dict(job["match"])
    status.update(state="running", games_planned=2 * cfg.pairs)

    def progress(record, records):
        live.on_record(record)
        status.update(games_done=len(records))

    summary = run_match(cfg, spec_a=EngineSpec.from_dict(job["a"]),
                        spec_b=EngineSpec.from_dict(job["b"]),
                        out_path=job_dir / "results.jsonl", names=job.get("names"),
                        progress=progress, observer=live.on_event)
    status.update(games_done=summary["games"])
    return summary


def run_job(job_dir) -> int:
    from .runtime.gpu import GpuBusyError, GpuLease
    from .runtime.locks import FileLock

    job_dir = Path(job_dir).resolve()
    alive = FileLock(job_dir / "lock")
    if not alive.acquire(timeout=0):
        print(f"[job] {job_dir} 已有 job 进程在跑", file=sys.stderr)
        return 4
    status = _Status(job_dir)

    def on_term(signum, frame):
        raise JobStopped()
    signal.signal(signal.SIGTERM, on_term)

    live = None
    lease = None
    try:
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        kind = job.get("kind", "match")
        if kind not in ("match", "game"):
            raise ValueError(f"未知的 job 类型 {kind!r}")
        budget = int(job.get("gpu_mib", 0))
        if budget > 0:
            lease = GpuLease(budget, name=f"job {job_dir.name}",
                             lock_dir=job.get("lease_dir")).acquire()
        live = _Live(job_dir, keep_finished=(kind == "game"))
        summary = (_run_game if kind == "game" else _run_match)(job, job_dir, live, status)
        status.update(state="completed", summary=summary, finished_at=time.time())
        return 0
    except GpuBusyError as e:
        status.update(state="gpu_busy", error=str(e), finished_at=time.time())
        return 2
    except (JobStopped, KeyboardInterrupt):
        status.update(state="stopped", finished_at=time.time())
        return 3
    except BaseException as e:  # noqa: 一切失败都要落到 status.json
        traceback.print_exc()
        status.update(state="error", error=f"{type(e).__name__}: {e}", finished_at=time.time())
        return 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if live is not None:
            live.close()
        if lease is not None:
            lease.release()
        alive.release()


# ====================================================================== 调用方侧

class JobHandle:
    """调用方（Server）对一个 job 目录的只读视图 + 启停。进程重启后可凭目录重建。"""

    def __init__(self, job_dir):
        self.dir = Path(job_dir)

    @classmethod
    def submit(cls, job_dir, job: dict, *, python: str = sys.executable,
               env: Optional[dict] = None, cwd=None) -> "JobHandle":
        """cwd 默认为 job 目录：调用方工作目录里的同名模块（如 Server 的 jobs.py / models）
        不会因 ``python -m`` 把 cwd 放进 sys.path 而遮蔽引擎的导入。"""
        job_dir = Path(job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(job_dir / "job.json", job)
        log = open(job_dir / "job.log", "ab")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True      # 独立进程组：stop 时连 worker 子进程一起结束
        env = dict(os.environ if env is None else env)
        # 子进程按包名 ``Kit.jobs`` 启动，cwd 是 job 目录，import 根只能经 PYTHONPATH 传过去
        env["PYTHONPATH"] = os.pathsep.join(
            [str(IMPORT_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        try:
            proc = subprocess.Popen([python, "-m", "Kit.jobs", str(job_dir)],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=env, cwd=str(cwd or job_dir), **kwargs)
        finally:
            log.close()
        handle = cls(job_dir)
        handle.proc = proc
        return handle

    def status(self) -> dict:
        return read_json(self.dir / "status.json", {}) or {}

    def live(self) -> dict:
        return read_json(self.dir / "live.json", {}) or {}

    def alive(self) -> bool:
        from .runtime.locks import is_locked
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            return True                      # 刚启动、尚未拿到锁
        return is_locked(self.dir / "lock")

    def state(self) -> str:
        """status.json 的状态；进程已死却没写终态（被 SIGKILL / OOM）时返回 "died"。"""
        st = self.status().get("state")
        if st in STATES_FINAL:
            return st
        if self.alive():
            return st or "starting"
        return "died"

    def wait_started(self, timeout: float = 10.0) -> str:
        """等到 job 越过启动阶段（running 或终态），返回当时的状态。"""
        deadline = time.monotonic() + timeout
        while True:
            st = self.state()
            if st != "starting" or time.monotonic() >= deadline:
                return st
            time.sleep(0.05)

    def wait(self, timeout: float = 30.0) -> bool:
        """等 job 进程完全退出（终态写出后进程还要收尾：关 worker、放锁、关日志）。"""
        deadline = time.monotonic() + timeout
        while self.alive():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def records(self) -> list:
        path = self.dir / "results.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue                     # 正在写入的半行
            if obj.get("type") == "game":
                out.append(obj)
        return out

    def stop(self, grace_s: float = 10.0) -> None:
        """SIGTERM 整个进程组，grace_s 内没退出再 SIGKILL。进程已死时什么都不做。"""
        if not self.alive():
            return
        pid = self.status().get("pid") or getattr(getattr(self, "proc", None), "pid", None)
        if not pid:
            return
        if os.name == "nt":
            proc = getattr(self, "proc", None)
            if proc is not None:
                proc.terminate()
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not self.alive():
                return
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="UniChessKit 对局 job（由调用方写好 job.json 后启动）")
    ap.add_argument("job_dir")
    args = ap.parse_args(argv)
    return run_job(args.job_dir)


if __name__ == "__main__":
    sys.exit(main())
