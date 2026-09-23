"""T/R 共用的 19×8×8 编码与 4096+4 着法索引（与 R core/encoding.py、core/moves.py 逐位一致）。

约定：永远从当前行棋方视角编码（轮到黑方时 mirror）。平面：
0-5 我方 PNBRQK，6-11 对方，12-15 易位权（我王翼/我后翼/对王翼/对后翼），
16 吃过路兵目标格，17 halfmove_clock/100，18 重复次数/2。
策略索引 from*64+to（行棋方视角），升变另有 4 维头（Q/R/B/N 顺序，改动会让权重错位）。
"""
from __future__ import annotations

from typing import Optional

import chess
import numpy as np

NUM_PLANES = 19
INPUT_SHAPE = (NUM_PLANES, 8, 8)
POLICY_SIZE = 64 * 64
PROMO_SIZE = 4
PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMO_TO_IDX = {p: i for i, p in enumerate(PROMO_PIECES)}
_PIECE_PLANE = {p: i for i, p in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                            chess.ROOK, chess.QUEEN, chess.KING))}


def orient(board: chess.Board) -> chess.Board:
    """轮到白方返回原棋盘（勿修改），轮到黑方返回镜像副本。"""
    return board if board.turn == chess.WHITE else board.mirror()


def orient_move(move: chess.Move, turn: bool) -> chess.Move:
    if turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square), promotion=move.promotion)


unorient_move = orient_move      # 镜像是对合的


def repetitions_of(board: chess.Board) -> int:
    return 2 if board.is_repetition(3) else (1 if board.is_repetition(2) else 0)


def encode(board: chess.Board, repetitions: Optional[int] = None) -> np.ndarray:
    if repetitions is None:
        repetitions = repetitions_of(board)
    b = orient(board)
    planes = np.zeros(INPUT_SHAPE, dtype=np.float32)
    for square, piece in b.piece_map().items():
        plane = _PIECE_PLANE[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(square, 8)
        planes[plane, row, col] = 1.0
    if b.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        planes[14] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        planes[15] = 1.0
    if b.ep_square is not None:
        row, col = divmod(b.ep_square, 8)
        planes[16, row, col] = 1.0
    planes[17] = min(b.halfmove_clock, 100) / 100.0
    planes[18] = min(repetitions, 2) / 2.0
    return planes


def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def move_to_promo_index(move: chess.Move) -> Optional[int]:
    return None if move.promotion is None else PROMO_TO_IDX[move.promotion]


def priors_from_policy(board: chess.Board, policy, promo) -> tuple:
    """4096 维策略 + 4 维升变头 → (合法着法, 归一化先验)。与 R search/mcts.py 同算法。"""
    moves = list(board.legal_moves)
    if not moves:
        return [], np.zeros(0, dtype=np.float32)
    scores = np.empty(len(moves), dtype=np.float32)
    for i, mv in enumerate(moves):
        om = orient_move(mv, board.turn)
        s = float(policy[move_to_index(om)])
        pi = move_to_promo_index(om)
        if pi is not None:
            s *= float(promo[pi])
        scores[i] = s
    total = scores.sum()
    if total <= 0:
        scores[:] = 1.0 / len(moves)
    else:
        scores /= total
    return moves, scores
