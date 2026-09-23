"""kit 的公共数据类型。只依赖标准库、chess 与 numpy。

协程约定（整个 kit 的核心）::

    Think = Generator[EvalRequest, Any, R]

需要网络前向的代码写成生成器：``yield EvalRequest(...)`` 交出一批待评估的负载，
驱动器（``runtime.Batcher``）把同一模型的请求跨对局拼成一次前向，再把与负载一一对应的
结果列表 ``send`` 回来；生成器 ``return`` 的值就是这次思考的结果。
不需要前向的实现（随机、UCI、残局表）写成「不 yield 的生成器」即可，驱动器同样适用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Sequence, TypeVar

import chess

R = TypeVar("R")
Think = Generator["EvalRequest", Any, R]


@dataclass(frozen=True, eq=False)
class EvalRequest:
    """一次批量评估请求。

    payloads 是同一模型的一个微批（例如一次 virtual-loss 收集到的全部叶子），
    驱动器保证 send 回来的列表与 payloads 等长、同序。
    """
    evaluator: Any          # BatchEvaluator（此处不写类型以免循环导入）
    payloads: tuple

    def __post_init__(self):
        if not isinstance(self.payloads, tuple):
            object.__setattr__(self, "payloads", tuple(self.payloads))
        if not self.payloads:
            raise ValueError("EvalRequest.payloads 不能为空：没有东西要评估就不要 yield")
        if not getattr(self.evaluator, "model_key", None):
            raise ValueError("evaluator 必须有非空的 model_key（驱动器按它分组攒批）")


@dataclass(frozen=True)
class SearchBudget:
    """单步思考预算。None 表示用 Player 自己的默认值。

    deadline 是 time.perf_counter() 刻度的墙钟截止；**带 deadline 的对局不可复现**，
    评测默认只用 simulations。
    """
    simulations: Optional[int] = None
    deadline: Optional[float] = None
    temperature: Optional[float] = None
    add_noise: bool = False


@dataclass(frozen=True)
class MoveDecision:
    move: chess.Move
    source: str = "unknown"      # tablebase / book / search / policy / random / uci
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GameStart:
    """开局信息。opening 是从起始局面出发、已经走完的开局着法（UCI）。

    起始局面默认是标准初始局面；fen 非空时从该局面开始（Server 的自定义局面对局）。
    """
    color: bool                              # 本 Player 执哪方（chess.WHITE / BLACK）
    seed: int = 0
    opening: tuple = ()
    game_id: str = ""
    fen: Optional[str] = None

    def board(self) -> chess.Board:
        b = chess.Board(self.fen) if self.fen else chess.Board()
        for uci in self.opening:
            b.push_uci(uci)
        return b


@dataclass(frozen=True)
class Verdict:
    """终局裁决。result 为 PGN 结果串；termination 用 S 的口径。"""
    result: str                              # "1-0" / "0-1" / "1/2-1/2"
    termination: str                         # checkmate / stalemate / insufficient_material /
                                             # fifty_move / threefold / truncated
    winner: Optional[bool] = None            # chess.WHITE / chess.BLACK / None

    @property
    def truncated(self) -> bool:
        return self.termination == "truncated"

    def white_score(self) -> float:
        return {"1-0": 1.0, "0-1": 0.0}.get(self.result, 0.5)


@dataclass(frozen=True)
class Leaf:
    """搜索树里待展开的一个节点。parent_handle 是父节点的引擎私有句柄
    （T/R 为 None；S 用它指向 Mamba 状态），move 是从父节点走到这里的着法。"""
    board: chess.Board
    parent_handle: Any = None
    move: Optional[chess.Move] = None


@dataclass(frozen=True, eq=False)
class NodeEval:
    """Expander 对一个叶子的评估结果。

    moves 为该局面**全部**合法着法（无合法着法时为空，由搜索按终局处理）；
    priors 与 moves 等长、和为 1；value 为当前行棋方视角的标量价值 ∈ [-1, 1]。
    logits 可选：与 moves 等长的原始 policy logits（fp32）。Gumbel 在 logits 上加噪声与 σ，
    提供它可避免 log(softmax) 的舍入误差（S 要求与原搜索逐位一致）；PUCT 不用它。
    """
    moves: Sequence[chess.Move]
    priors: Any                              # np.ndarray[float32]
    value: float
    handle: Any = None
    logits: Any = None


class PlayerError(RuntimeError):
    """Player 违反契约（非法着法、协程协议错误等）。"""
