"""多进程 worker 池：生命周期必须显式。

S 的 d87371c 教训：子进程静默退出（OOM、被 kill、C 扩展段错误）时，父进程若只看
「队列空了 + 进程没了」，就会把缺了一半的结果当成完整结果。这里的规则是：

- 每个 worker 必须以一条 ``("done", wid, None)`` 或 ``("error", wid, traceback)`` 收尾；
- 进程已退出却没发过收尾消息 → 视为错误（报告退出码）；
- 任何一个 worker 出错 → 置停止事件，其余 worker 尽快结束，run() 抛 WorkerError。
  **不能无限等其余 worker**：万一边上有个卡死的（CUDA 调用不返回、不看 stop_event），
  父进程会永远等下去，把挂死伪装成"运行中"——出错后再等 ``error_grace_s`` 秒就强杀。

2026-09-26 的新教训：**父进程被杀时 worker 不会自己退**。spawn 出来的 worker cmdline 是
``python -c from multiprocessing.spawn import spawn_main ...``，``pkill -f 'Kit match'``
匹配不到；父进程一死它们被过继给 init，还占着 GPU 显存和显存租约不放手。实测两个孤儿
worker 就吃掉 2.5G，攒到 4 个把后来的训练直接 OOM。所以每个 worker 都带一个看门狗：
父进程的管道写端一关（等价于父进程死亡）就置停止事件，宽限 10 秒后 ``os._exit(3)`` 强退。

worker 函数签名：``fn(wid, task, emit, stop_event)``。``emit(obj)`` 把结果发回父进程
（父进程按到达顺序回调 on_result），``stop_event.is_set()`` 为真时应尽快返回。
fn 与 task 必须可 pickle（spawn 下 fn 必须是模块顶层函数）。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
import traceback
from typing import Any, Callable, Optional, Sequence

GRACE_S = 10.0        # 看门狗发现父进程死后给 fn 的收尾宽限
POLL_S = 1.0          # 看门狗的轮询间隔
ERROR_GRACE_S = 300.0  # 出错（含静默退出）后等其余 worker 收尾的宽限；不无限等。
# 为什么不是 10 s：arena / 筛选赛单局 20~45 s（2400 sims）、自对弈更长，出错时另一个
# worker 通常正在下在途局。本文件的既有口径是"已开局的照常下完并入库"——宽限必须够
# 一局下完（所以取分钟级），只用来防"某个 worker 卡死 + 不看 stop_event → 父进程永远
# 等下去、把挂死伪装成运行中"。真要早报错，构造 WorkerPool 时传 error_grace_s 覆盖。


class WorkerError(RuntimeError):
    pass


class _Emitter:
    """可 pickle 的 emit 回调（lambda 在 spawn 下无法传递，这里也不需要传递，只是保持清晰）。"""

    def __init__(self, q, wid):
        self.q, self.wid = q, wid

    def __call__(self, obj) -> None:
        self.q.put(("result", self.wid, obj))


def _put(q, msg) -> None:
    """父进程可能已经没了：发不出去就别赌（队列满/管道断都会抛）。"""
    try:
        q.put(msg, timeout=5)
    except Exception:  # noqa: 父进程已死时收尾消息没有接收方，丢掉即可
        pass


def _parent_gone(alive_r, ppid: int) -> bool:
    """父进程是否已死：优先看管道的 EOF，退化看 ppid 是否变了（Linux 上会被过继给 1）。"""
    if alive_r is not None:
        try:
            if not alive_r.poll():
                return False
            try:
                alive_r.recv()          # 父进程不会发数据；抛 EOFError 就是写端关了
            except EOFError:
                return True
            except OSError:
                return True
            return False
        except OSError:
            return True
    try:
        return os.getppid() != ppid
    except OSError:
        return True


def _watch_parent(alive_r, ppid: int, stop_event, guard: threading.Event) -> None:
    """看门狗：父进程一没就置停止事件，宽限后强退。"""
    while not guard.wait(POLL_S):
        if not _parent_gone(alive_r, ppid):
            continue
        stop_event.set()                 # fn 里轮询 stop_event，会尽快收工
        if not guard.wait(GRACE_S):      # 宽限：让 fn 把当前这一局收尾
            os._exit(3)                  # 仍不退就强杀，别占着显存当孤儿
        return


def _entry(fn, wid, task, q, stop_event, alive_r=None):
    ppid = os.getppid()
    guard = threading.Event()
    threading.Thread(target=_watch_parent,
                     args=(alive_r, ppid, stop_event, guard), daemon=True).start()
    try:
        fn(wid, task, _Emitter(q, wid), stop_event)
    except BaseException:  # noqa: 一切异常（含 KeyboardInterrupt / SystemExit）都报给父进程
        _put(q, ("error", wid, traceback.format_exc()))
        return
    finally:
        guard.set()                      # 正常结束：让看门狗退出
    _put(q, ("done", wid, None))


class WorkerPool:
    def __init__(self, start_method: str = "spawn", poll_s: float = 0.5,
                 error_grace_s: float = ERROR_GRACE_S):
        # 默认 spawn：CUDA 在 fork 出来的子进程里不能用；spawn 也让 Windows / Linux 行为一致
        self.ctx = mp.get_context(start_method)
        self.poll_s = poll_s
        self.error_grace_s = error_grace_s

    def run(self, fn: Callable, tasks: Sequence[Any],
            on_result: Callable[[int, Any], None],
            should_stop: Callable[[], bool] = lambda: False) -> None:
        q = self.ctx.Queue()
        stop_event = self.ctx.Event()
        # 父进程活着凭据：写端由父进程持有且不写数据；父进程一死（含被 SIGKILL）
        # 子进程的读端就收到 EOF，看门狗据此收工。不能提前关写端，否则会被误判。
        alive_r, alive_w = self.ctx.Pipe(False)
        procs = {wid: self.ctx.Process(target=_entry,
                                       args=(fn, wid, task, q, stop_event, alive_r),
                                       daemon=True)
                 for wid, task in enumerate(tasks)}
        for p in procs.values():
            p.start()
        finished: set = set()
        errors: list = []
        grace_until: Optional[float] = None   # 出错后等其余 worker 收尾的宽限时刻
        force = False                         # 宽限已过：剩下的 worker 直接强杀
        try:
            while len(finished) < len(procs):
                if errors and grace_until is None:
                    # 出错整批停止：停止事件已置，守规矩的 worker 会尽快收尾；但万一有
                    # worker 卡死（CUDA 调用不返回）又不看 stop_event，无限等下去会把
                    # 它伪装成「还在跑」——父进程在 q.get 上空转，整批永无结论。
                    grace_until = time.monotonic() + self.error_grace_s
                if grace_until is not None and time.monotonic() >= grace_until:
                    laggards = sorted(set(procs) - finished)
                    errors.append(f"worker {laggards} 未在 {self.error_grace_s:g}s 宽限内"
                                  f"收尾（出错整批停止，强制结束）")
                    force = True
                    break
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
            join_s = 0.1 if force else 30     # 已判定卡死就别再陪等 30s
            for p in procs.values():
                p.join(timeout=join_s)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            alive_w.close()      # 放在 join 之后：提前关会让还活着的 worker 误判父进程已死
            alive_r.close()
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
