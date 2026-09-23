"""``chess.Board.outcome`` 的快速等价实现（P4）。

python-chess 的 ``can_claim_threefold_repetition`` 每次都要逐个 pop/push 回溯可逆历史，
再对**每个合法着** push/pop 一遍——搜索里每个新节点都要判一次终局，这是 CPU 热点。
绝大多数局面的可逆窗口内根本没有重复，这里先用一个**必要条件**快速排除：

    申和三次重复 ⇒ 计数器里有某个 transposition key 出现 ≥ 2 次
               ⇒ 窗口内有两个局面的「粗键」（各类棋子位棋盘 + 白方占位 + 走子方）相同。

窗口取最近 min(halfmove_clock, len(move_stack)) 个历史局面（python-chess 回溯到最后一个不可逆
着法为止，吃子 / 兵步都不可逆，所以其窗口是它的子集），粗键比 transposition key 粗（不含
易位权与过路兵），所以粗键全不相同时原函数必返回 False；否则回退到原函数。结果逐项相同
（``tests/test_rules_fast.py`` 对随机对局与构造的重复局面逐局面对照）。
"""
from __future__ import annotations

from typing import Optional

import chess


def _coarse(s) -> tuple:
    return (s.pawns, s.knights, s.bishops, s.rooks, s.queens, s.kings, s.occupied_w, s.turn)


def may_claim_threefold(board: chess.Board) -> bool:
    """False ⇒ ``board.can_claim_threefold_repetition()`` 必为 False（True 则需原函数确认）。"""
    n = len(board.move_stack)
    h = min(board.halfmove_clock, n)
    seen = {(board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings,
             board.occupied_co[chess.WHITE], board.turn)}
    stack = board._stack
    for j in range(n - 1, n - 1 - h, -1):
        k = _coarse(stack[j])
        if k in seen:
            return True
        seen.add(k)
    return False


def outcome(board: chess.Board, claim_draw: bool = False) -> Optional[chess.Outcome]:
    """与 ``board.outcome(claim_draw=claim_draw)`` 返回值相同（判定顺序也相同）。"""
    out = board.outcome(claim_draw=False)
    if out is not None or not claim_draw:
        return out
    if board.can_claim_fifty_moves():
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    if may_claim_threefold(board) and board.can_claim_threefold_repetition():
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    return None


def is_game_over(board: chess.Board, claim_draw: bool = False) -> bool:
    return outcome(board, claim_draw) is not None


def copy_board(board: chess.Board) -> chess.Board:
    """``board.copy(stack=True)`` 的快速版：共享（不可变用法的）Move 对象，只复制两个列表。"""
    b = board.copy(stack=False)
    b.move_stack = board.move_stack.copy()
    b._stack = board._stack.copy()
    return b
