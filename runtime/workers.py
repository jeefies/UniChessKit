"""多进程 worker 池：生命周期必须显式。

S 的 d87371c 教训：子进程静默退出（OOM、被 kill、C 扩展段错误）时，父进程若只看
「队列空了 + 进程没了」，就会把缺了一半的结果当成完整结果。这里的规则是：

- 每个 worker 必须以一条 ``("done", wid, None)`` 或 ``("error", wid, traceback)`` 收尾；
- 进程已退出却没发过收尾消息 → 视为错误（报告退出码）；
- 任何一个 worker 出错 → 置停止事件，其余 worker 尽快结束，run() 抛 WorkerError。

worker 函数签名：``fn(wid, task, emit, stop_event)``。``emit(obj)`` 把结果发回父进程
（父进程按到达顺序回调 on_result），``stop_event.is_set()`` 为真时应尽快返回。
fn 与 task 必须可 pickle（spawn 下 fn 必须是模块顶层函数）。
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import traceback
from typing import Any, Callable, Sequence


class WorkerError(RuntimeError):
    pass


class _Emitter:
    """可 pickle 的 emit 回调（lambda 在 spawn 下无法传递，这里也不需要传递，只是保持清晰）。"""

    def __init__(self, q, wid):
        self.q, self.wid = q, wid

    def __call__(self, obj) -> None:
        self.q.put(("result", self.wid, obj))


def _entry(fn, wid, task, q, stop_event):
    try:
        fn(wid, task, _Emitter(q, wid), stop_event)
    except BaseException:  # noqa: 一切异常（含 KeyboardInterrupt / SystemExit）都报给父进程
        q.put(("error", wid, traceback.format_exc()))
        return
    q.put(("done", wid, None))


class WorkerPool:
    def __init__(self, start_method: str = "spawn", poll_s: float = 0.5):
        # 默认 spawn：CUDA 在 fork 出来的子进程里不能用；spawn 也让 Windows / Linux 行为一致
        self.ctx = mp.get_context(start_method)
        self.poll_s = poll_s

    def run(self, fn: Callable, tasks: Sequence[Any],
            on_result: Callable[[int, Any], None],
            should_stop: Callable[[], bool] = lambda: False) -> None:
        q = self.ctx.Queue()
        stop_event = self.ctx.Event()
        procs = {wid: self.ctx.Process(target=_entry, args=(fn, wid, task, q, stop_event),
                                       daemon=True)
                 for wid, task in enumerate(tasks)}
        for p in procs.values():
            p.start()
        finished: set = set()
        errors: list = []
        try:
            while len(finished) < len(procs):
                try:
                    kind, wid, payload = q.get(timeout=self.poll_s)
                except queue_mod.Empty:
                    for wid, p in procs.items():
                        if wid in finished or p.is_alive():
                            continue
                        # 退出了却可能还有消息在管道里：先排空再判定
                        self._drain(q, finished, errors, on_result)
                        if wid not in finished:
                            finished.add(wid)
                            errors.append(f"worker {wid} 静默退出（exitcode={p.exitcode}），"
                                          "未发送 done/error")
                    if errors:
                        stop_event.set()
                    continue
                self._handle(kind, wid, payload, finished, errors, on_result)
                if errors or should_stop():
                    stop_event.set()
        finally:
            stop_event.set()
            for p in procs.values():
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        if errors:
            raise WorkerError("\n".join(errors))

    def _drain(self, q, finished, errors, on_result) -> None:
        while True:
            try:
                kind, wid, payload = q.get(timeout=0.2)
            except queue_mod.Empty:
                return
            self._handle(kind, wid, payload, finished, errors, on_result)

    @staticmethod
    def _handle(kind, wid, payload, finished, errors, on_result) -> None:
        if kind == "result":
            on_result(wid, payload)
        elif kind == "done":
            finished.add(wid)
        elif kind == "error":
            finished.add(wid)
            errors.append(f"worker {wid} 出错：\n{payload}")
        else:
            raise WorkerError(f"未知消息类型 {kind!r}")
