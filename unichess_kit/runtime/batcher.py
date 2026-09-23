"""协程攒批驱动。

Batcher 只做一件事：把一拍里各协程交出的 EvalRequest 按 model_key 分组，
每个模型一次 evaluate，再把结果切回给各自的协程。

**可复现性**：分组顺序 = model_key 在本拍请求里第一次出现的顺序，组内顺序 = 请求顺序，
请求顺序 = 槽位顺序。整个驱动是单线程、不看墙钟的，所以同样的输入永远拼出同样的批——
这一点对 fp16 推理很重要：批的组成不同，数值就可能差一个 ulp，进而改变 argmax。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional

from ..api.types import EvalRequest, Think


@dataclass
class BatchStats:
    ticks: int = 0              # 驱动拍数
    forwards: int = 0           # evaluate 调用次数
    positions: int = 0          # 评估的负载总数
    requests: int = 0           # 收到的 EvalRequest 数

    def as_dict(self) -> dict:
        return dict(ticks=self.ticks, forwards=self.forwards,
                    positions=self.positions, requests=self.requests)


class Batcher:
    def __init__(self):
        self.stats = BatchStats()

    def evaluate(self, requests: list) -> list:
        """评估一拍的请求，返回与 requests 同序的结果列表（每项与该请求的 payloads 等长）。"""
        self.stats.ticks += 1
        self.stats.requests += len(requests)
        groups: dict = {}
        owners: dict = {}
        for i, req in enumerate(requests):
            if not isinstance(req, EvalRequest):
                raise TypeError(f"协程必须 yield EvalRequest，收到 {type(req).__name__}")
            key = req.evaluator.model_key
            owner = owners.setdefault(key, req.evaluator)
            if owner is not req.evaluator:
                raise ValueError(f"两个不同的 evaluator 使用了同一个 model_key={key!r}；"
                                 "会被错误地拼进同一次前向")
            groups.setdefault(key, []).append(i)

        out: list = [None] * len(requests)
        for key, idxs in groups.items():
            payloads: list = []
            for i in idxs:
                payloads.extend(requests[i].payloads)
            results = list(owners[key].evaluate(payloads))
            if len(results) != len(payloads):
                raise ValueError(f"evaluator {key!r} 返回 {len(results)} 个结果，"
                                 f"应为 {len(payloads)} 个")
            self.stats.forwards += 1
            self.stats.positions += len(payloads)
            pos = 0
            for i in idxs:
                n = len(requests[i].payloads)
                out[i] = results[pos:pos + n]
                pos += n
        return out


def run_sync(think: Think, batcher: Optional[Batcher] = None):
    """单独驱动一个协程直到完成，返回它的结果（serving、测试用）。"""
    batcher = batcher or Batcher()
    try:
        req = think.send(None)
        while True:
            (result,) = batcher.evaluate([req])
            req = think.send(result)
    except StopIteration as stop:
        return stop.value


class CoroutinePool:
    """并发驱动多个协程任务，跨任务攒批。

    jobs 是 (job_id, 工厂) 的可迭代对象，工厂调用后返回一个协程；最多 concurrency 个同时在跑。
    每个任务结束时按**完成顺序**调用 on_done(job_id, 结果)。
    should_stop() 返回真后不再启动新任务，已在跑的任务照常跑完（保证每个启动的任务都有记录）。
    任何协程抛异常 → 关闭所有协程并把异常原样抛给调用方（出错整批停止，不留半截状态）。
    """

    def __init__(self, concurrency: int, batcher: Optional[Batcher] = None):
        if concurrency < 1:
            raise ValueError("concurrency 必须 >= 1")
        self.concurrency = concurrency
        self.batcher = batcher or Batcher()

    def run(self, jobs: Iterable, on_done: Callable[[Any, Any], None],
            should_stop: Callable[[], bool] = lambda: False) -> None:
        queue: Iterator = iter(jobs)
        slots: list = [None] * self.concurrency   # [job_id, gen, pending_req]
        try:
            for i in range(self.concurrency):
                self._fill(i, slots, queue, on_done, should_stop)
            while True:
                active = [i for i, s in enumerate(slots) if s is not None]
                if not active:
                    return
                results = self.batcher.evaluate([slots[i][2] for i in active])
                for i, result in zip(active, results):
                    job_id, gen, _ = slots[i]
                    try:
                        slots[i][2] = gen.send(result)
                    except StopIteration as stop:
                        slots[i] = None
                        on_done(job_id, stop.value)
                        self._fill(i, slots, queue, on_done, should_stop)
        except BaseException:
            for s in slots:
                if s is not None:
                    s[1].close()
            raise

    @staticmethod
    def _fill(i, slots, queue, on_done, should_stop) -> None:
        slots[i] = None
        while not should_stop():
            try:
                job_id, factory = next(queue)
            except StopIteration:
                return
            gen = factory()
            try:
                req = gen.send(None)
            except StopIteration as stop:       # 不需要前向的任务当场完成
                on_done(job_id, stop.value)
                continue
            slots[i] = [job_id, gen, req]
            return
