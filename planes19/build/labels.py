"""Stockfish 评分 → 训练标签（WDL 三分布、MultiPV 策略软标签）。"""
from __future__ import annotations

import math
from typing import Optional

import chess

from ..encoding import move_to_index, move_to_promo_index, orient_move

MATE_CP = 30000.0
_CP_K = 0.00368208          # Lichess 的厘兵 → 胜率 logistic 系数


def cp_to_win_prob(cp: float) -> float:
    """厘兵评分 → 行棋方胜率 [0, 1]。"""
    return 1.0 / (1.0 + math.exp(-_CP_K * cp))


def score_to_wdl(cp: Optional[float], mate: Optional[int]) -> tuple:
    """(cp, mate) → (胜, 和, 负)。和棋权重随 |cp| 增大按高斯衰减，均势时最高（约 0.55）。

    单标量价值头在国象会塌成全 0，所以用三头。
    """
    if mate is not None:
        return (1.0, 0.0, 0.0) if mate > 0 else (0.0, 0.0, 1.0)
    w_raw = cp_to_win_prob(cp)
    draw = 0.55 * math.exp(-((cp / 220.0) ** 2))
    win = (1.0 - draw) * w_raw
    loss = (1.0 - draw) * (1.0 - w_raw)
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


def softmax_policy(cands, temperature: float, top: int = 5) -> list:
    """[(索引, 升变, 行棋方厘兵), ...] → 前 top 条的 [(索引, 概率)]。

    **先截断再归一**：softmax 之后再截断，PV 多于 top 条时存下的概率和会小于 1。
    """
    best_cp = max(c[2] for c in cands)
    weights = [math.exp((c[2] - best_cp) / temperature) for c in cands]
    pairs = list(zip(cands, weights))[:top]
    sub = sum(w for _, w in pairs)
    return [(c[0], w / sub) for c, w in pairs]


def policy_from_multipv(board: chess.Board, infos: list, temperature: float = 90.0) -> tuple:
    """python-chess ``engine.analyse(multipv=k)`` 结果 → (策略软分布, 首选着法的升变索引)。

    温度单位是厘兵：90 表示差 90 厘兵的着法权重约为 1/e。
    """
    cands = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        score = info["score"].pov(board.turn)
        mate = score.mate()
        cp = MATE_CP if (mate is not None and mate > 0) else \
            -MATE_CP if mate is not None else float(score.score())
        cands.append((pv[0], cp))
    if not cands:
        return [], None
    best_cp = max(c for _, c in cands)
    weights = [math.exp((cp - best_cp) / temperature) for _, cp in cands]
    total = sum(weights)
    policy = [(move_to_index(orient_move(mv, board.turn)), w / total)
              for (mv, _), w in zip(cands, weights)]
    promo = move_to_promo_index(orient_move(cands[0][0], board.turn))
    return policy, promo
