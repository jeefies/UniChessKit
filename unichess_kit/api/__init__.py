"""协议与数据类型。kit 里最稳定的一层，不依赖 torch。"""
from .protocols import BatchEvaluator, Expander, Player, RecordSink, Referee
from .types import (EvalRequest, GameStart, Leaf, MoveDecision, NodeEval, PlayerError,
                    SearchBudget, Think, Verdict)


def immediate(value):
    """把一个现成的值包装成「不 yield 的思考」，供不需要前向的实现使用。"""
    return value
    yield  # noqa: 不可达；只为让本函数成为生成器


def gather(thinks):
    """并发驱动多个思考协程，返回按输入顺序排列的结果列表。

    每一拍把各协程交出的请求按 evaluator 合并成一个 EvalRequest（组顺序 = 该 evaluator
    首次出现的顺序，组内 = 协程顺序），驱动器只看到一次更大的批——所以一次搜索内部
    互不依赖的模拟也能拼进同一次前向。任一协程抛异常时关闭其余协程并原样抛出。
    """
    thinks = list(thinks)
    results: list = [None] * len(thinks)
    pending: dict = {}
    try:
        for i, t in enumerate(thinks):
            try:
                pending[i] = t.send(None)
            except StopIteration as stop:
                results[i] = stop.value
        while pending:
            groups: dict = {}
            for i, req in pending.items():
                groups.setdefault(id(req.evaluator), (req.evaluator, []))[1].append(i)
            for evaluator, idxs in groups.values():
                reqs = [pending[i] for i in idxs]
                out = yield EvalRequest(evaluator, [p for r in reqs for p in r.payloads])
                pos = 0
                for i, r in zip(idxs, reqs):
                    n = len(r.payloads)
                    try:
                        pending[i] = thinks[i].send(out[pos:pos + n])
                    except StopIteration as stop:
                        results[i] = stop.value
                        del pending[i]
                    pos += n
        return results
    finally:
        for i in list(pending):
            thinks[i].close()


__all__ = [
    "BatchEvaluator", "Expander", "Player", "RecordSink", "Referee",
    "EvalRequest", "GameStart", "Leaf", "MoveDecision", "NodeEval", "PlayerError",
    "SearchBudget", "Think", "Verdict", "gather", "immediate",
]
