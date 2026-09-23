"""kit 的协议（接口）。管线只依赖这里，引擎只实现这里——两边互不 import。"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import chess

from .types import GameStart, Leaf, MoveDecision, NodeEval, SearchBudget, Think, Verdict


@runtime_checkable
class BatchEvaluator(Protocol):
    """引擎的一次批量前向。

    model_key 唯一标识「一份权重 + 一套推理设置」：驱动器把相同 model_key 的请求
    拼成一次 evaluate 调用。两个不同对象用同一个 model_key 会被驱动器拒绝。
    evaluate 必须返回与 payloads 等长、同序的结果；负载与结果的类型由引擎自定。
    """
    model_key: str

    def evaluate(self, payloads: Sequence[Any]) -> Sequence[Any]: ...


@runtime_checkable
class Expander(Protocol):
    """把叶子局面变成 (合法着法, 先验, 价值)。搜索算法只通过它接触模型。

    状态型模型（S 的 Mamba cache）把状态管理封装在这里，通过 Leaf.parent_handle /
    NodeEval.handle 与搜索树交接；无状态模型（T/R）句柄恒为 None。
    """

    def expand(self, leaves: Sequence[Leaf]) -> Think[list]: ...  # -> list[NodeEval]


@runtime_checkable
class Player(Protocol):
    """对局参与者。match / serving 只依赖它。

    生命周期：new_game → (choose | observe)* → close。每局一个实例。
    - choose 收到的 board 是裁判局面的副本，Player 可以随意修改。
    - observe 在**每一步**之后（包括自己走的）都会被调用，board 为走子之后的局面副本。
    - close 必须幂等。
    """
    name: str

    def new_game(self, start: GameStart) -> Think[None]: ...

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]: ...

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]: ...

    def close(self) -> None: ...


@runtime_checkable
class Referee(Protocol):
    def verdict(self, board: chess.Board) -> Optional[Verdict]: ...


@runtime_checkable
class RecordSink(Protocol):
    """自对弈训练数据的落盘方式（格式归引擎所有）。

    每局结束时调用一次（按**完成顺序**；并发 1 时即局序）：record 为结果记录，
    board 为终局局面（带完整着法栈），decisions 为逐 ply 的 MoveDecision（训练目标在
    decision.info 里，内容由引擎的 Player 与 Sink 约定）。
    """

    def on_game_end(self, record: dict, board: chess.Board,
                    decisions: Sequence[MoveDecision]) -> None: ...
