"""把 T/R 的 ``evaluate_batch(boards) -> (policy, promo, wdl)`` 接入 kit。

- ``BatchFnEvaluator``：包装引擎已有的批量前向函数，负载 = chess.Board，
  结果 = 每个局面一个 (policy[4096], promo[4], wdl[3]) 元组；超过 max_batch 时分块前向。
- ``Planes19Expander``：一批叶子 → 一个 EvalRequest → NodeEval（先验 + Q=P胜-P负）。
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from ...api.types import EvalRequest, NodeEval, Think
from .encoding import priors_from_policy


class BatchFnEvaluator:
    def __init__(self, model_key: str, fn: Callable, max_batch: Optional[int] = None):
        if not model_key:
            raise ValueError("model_key 不能为空")
        if max_batch is not None and max_batch < 1:
            raise ValueError("max_batch 必须 >= 1")
        self.model_key = model_key
        self.fn = fn
        self.max_batch = max_batch

    def evaluate(self, payloads: Sequence) -> list:
        boards = list(payloads)
        step = self.max_batch or len(boards)
        out: list = []
        for lo in range(0, len(boards), step):
            chunk = boards[lo:lo + step]
            policy, promo, wdl = self.fn(chunk)
            if not (len(policy) == len(promo) == len(wdl) == len(chunk)):
                raise ValueError(f"{self.model_key}: evaluate_batch 返回的批大小与输入不符")
            out.extend(zip(policy, promo, wdl))
        return out


class Planes19Expander:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def expand(self, leaves) -> Think[list]:
        outs = yield EvalRequest(self.evaluator, tuple(leaf.board for leaf in leaves))
        evals = []
        for leaf, (policy, promo, wdl) in zip(leaves, outs):
            moves, priors = priors_from_policy(leaf.board, policy, promo)
            evals.append(NodeEval(moves=moves, priors=priors,
                                  value=float(wdl[0] - wdl[2])))
        return evals
