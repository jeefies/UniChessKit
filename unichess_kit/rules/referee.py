"""终局裁决：全 kit 唯一的口径。

沿用 S 的 ``adapter.classify_final_board``：以 ``board.outcome(claim_draw=True)`` 为准。
该语义把「下一步可申和」的三次重复 / 五十步也算作终局，比严格的 ``is_repetition(3)``
早一个 ply。S 曾因两种口径混用，把 78% 的规则申和局错记成了截断（2026-09-19，gen2k）；
Server 旧版用 ``is_game_over()``（不申和）则让重复局面一直走到封顶。
**只有规则上未终局、但走满 max_plies 的才算截断**，截断按和棋计分。
"""
from __future__ import annotations

from typing import Optional

import chess

from ..api.types import Verdict

TERMINATION_REASON = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.SEVENTYFIVE_MOVES: "fifty_move",
    chess.Termination.FIFTY_MOVES: "fifty_move",
    chess.Termination.FIVEFOLD_REPETITION: "threefold",
    chess.Termination.THREEFOLD_REPETITION: "threefold",
}

TRUNCATED = Verdict(result="1/2-1/2", termination="truncated", winner=None)


def classify(board: chess.Board) -> Optional[Verdict]:
    """规则终局 → Verdict；未终局 → None。"""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    reason = TERMINATION_REASON.get(outcome.termination)
    if reason is None:  # 标准国际象棋不会出现（变体的终止原因）
        raise ValueError(f"未知终止原因 {outcome.termination} @ {board.fen()}")
    return Verdict(result=outcome.result(), termination=reason, winner=outcome.winner)


class StandardReferee:
    """规则裁决 + 封顶截断。max_plies 按整盘棋（含开局）的 ply 数计。"""

    def __init__(self, max_plies: Optional[int] = None):
        if max_plies is not None and max_plies < 1:
            raise ValueError("max_plies 必须 >= 1")
        self.max_plies = max_plies

    def verdict(self, board: chess.Board) -> Optional[Verdict]:
        v = classify(board)
        if v is not None:
            return v
        if self.max_plies is not None and len(board.move_stack) >= self.max_plies:
            return TRUNCATED
        return None
