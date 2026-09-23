"""协议与数据类型。kit 里最稳定的一层，不依赖 torch。"""
from .protocols import BatchEvaluator, Expander, Player, RecordSink, Referee
from .types import (EvalRequest, GameStart, Leaf, MoveDecision, NodeEval, PlayerError,
                    SearchBudget, Think, Verdict)


def immediate(value):
    """把一个现成的值包装成「不 yield 的思考」，供不需要前向的实现使用。"""
    return value
    yield  # noqa: 不可达；只为让本函数成为生成器


__all__ = [
    "BatchEvaluator", "Expander", "Player", "RecordSink", "Referee",
    "EvalRequest", "GameStart", "Leaf", "MoveDecision", "NodeEval", "PlayerError",
    "SearchBudget", "Think", "Verdict", "immediate",
]
