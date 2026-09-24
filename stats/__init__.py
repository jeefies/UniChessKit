"""对局统计：Elo±误差、LOS、GSPRT（三项式 / 五项式）。

单一实现，取代 T/R 的 eval/arena.py 与 S 的伯努利 SPRT。以 R 的 eval/arena.py 为底本：

- 全胜 / 全负时 Elo 返回 ±inf，而不是一个像真值的大数（「+3600」会误导人以为测出了差距）；
- SPRT 用广义 SPRT 的正态近似（Fishtest 同款）：LLR = n (s1-s0)(2s̄-s0-s1) / (2σ²)。

**五项式（pentanomial）**：开局成对（同一开局 A/B 各执白一次）时，两局结果强相关，
按单局独立处理会高估方差、低估显著性；应以「一对」为样本（对得分 ∈ {0, ½, 1, 1½, 2}）。
match 管线默认用五项式做早停判定，三项式只作参考。

零方差（例如 20 对全胜）时正态近似失效：按 Fishtest 的做法，任何一格为 0 时每格加 1e-3。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

ELO_INF = float("inf")
_REG = 1e-3


def score_to_elo(score: float) -> float:
    """得分率 → Elo 差（逻辑斯谛模型）。"""
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def _z(confidence: float) -> float:
    # 只需要两个常用值，避免引入 scipy
    table = {0.95: 1.959963985, 0.99: 2.575829304, 0.90: 1.644853627}
    for c, z in table.items():
        if abs(confidence - c) < 1e-9:
            return z
    raise ValueError(f"不支持的置信度 {confidence}（可选 0.90 / 0.95 / 0.99）")


def _mean_var(values: Sequence[float], counts: Sequence[float]) -> tuple:
    n = float(sum(counts))
    mean = sum(v * c for v, c in zip(values, counts)) / n
    var = sum(c * (v - mean) ** 2 for v, c in zip(values, counts)) / n
    return n, mean, var


def _regularize(counts: Sequence[float]) -> list:
    counts = [float(c) for c in counts]
    if any(c == 0 for c in counts):
        counts = [c + _REG for c in counts]
    return counts


def _elo_ci(mean: float, var: float, n: float, confidence: float) -> tuple:
    se = math.sqrt(var / n) if n > 0 else ELO_INF
    z = _z(confidence)
    lo_s, hi_s = mean - z * se, mean + z * se
    elo = ELO_INF if mean >= 1.0 else (-ELO_INF if mean <= 0.0 else score_to_elo(mean))
    lo = -ELO_INF if lo_s <= 0.0 else score_to_elo(min(lo_s, 1.0))
    hi = ELO_INF if hi_s >= 1.0 else score_to_elo(max(hi_s, 0.0))
    return elo, lo, hi


# ------------------------------------------------------------------ 三项式（逐局）

TRI_VALUES = (1.0, 0.5, 0.0)          # 胜 / 和 / 负


def elo_with_error(w: int, d: int, l: int, confidence: float = 0.95) -> tuple:
    """返回 (Elo 差, 下界, 上界)。与 R eval/arena.py 同口径（不做正则化）。"""
    n = w + d + l
    if n == 0:
        return 0.0, -ELO_INF, ELO_INF
    _, mean, var = _mean_var(TRI_VALUES, (w, d, l))
    return _elo_ci(mean, var, n, confidence)


def los(w: int, d: int, l: int) -> float:
    """Likelihood of superiority：A 真的比 B 强的概率（和棋不提供信息）。"""
    if w + l == 0:
        return 0.5
    return 0.5 * (1 + math.erf((w - l) / math.sqrt(2.0 * (w + l))))


def _gsprt(values, counts, elo0: float, elo1: float) -> float:
    if sum(counts) == 0:
        return 0.0
    n, mean, var = _mean_var(values, _regularize(counts))
    if var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def sprt_llr(w: int, d: int, l: int, elo0: float, elo1: float) -> float:
    """三项式 GSPRT 的对数似然比。H0: Elo 差 = elo0；H1: Elo 差 = elo1。"""
    return _gsprt(TRI_VALUES, (w, d, l), elo0, elo1)


# ------------------------------------------------------------------ 五项式（逐对）

PENTA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)   # 对得分 / 2


def pentanomial(pair_scores: Sequence[float]) -> list:
    """对得分列表（每对 ∈ {0, 0.5, 1, 1.5, 2}）→ 五格计数 [LL, LD, DD/WL, WD, WW]。"""
    counts = [0] * 5
    for s in pair_scores:
        k = int(round(s * 2))
        if not 0 <= k <= 4 or abs(s * 2 - k) > 1e-9:
            raise ValueError(f"非法对得分 {s}")
        counts[k] += 1
    return counts


def elo_with_error_pentanomial(counts: Sequence[int], confidence: float = 0.95) -> tuple:
    n = sum(counts)
    if n == 0:
        return 0.0, -ELO_INF, ELO_INF
    _, mean, var = _mean_var(PENTA_VALUES, counts)
    return _elo_ci(mean, var, n, confidence)


def sprt_llr_pentanomial(counts: Sequence[int], elo0: float, elo1: float) -> float:
    return _gsprt(PENTA_VALUES, counts, elo0, elo1)


# ------------------------------------------------------------------ 判定

def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> tuple:
    """(下界, 上界)。LLR <= 下界 接受 H0，>= 上界 接受 H1。"""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def sprt_verdict(llr: float, alpha: float = 0.05, beta: float = 0.05) -> Optional[str]:
    """'H1'（确实更强）/ 'H0'（没有变强）/ None（继续）。"""
    lower, upper = sprt_bounds(alpha, beta)
    if llr >= upper:
        return "H1"
    if llr <= lower:
        return "H0"
    return None


def fmt_elo(x: float) -> str:
    if x == ELO_INF:
        return "+inf"
    if x == -ELO_INF:
        return "-inf"
    return f"{x:+.1f}"


__all__ = ["ELO_INF", "score_to_elo", "elo_to_score", "elo_with_error", "los", "sprt_llr",
           "pentanomial", "elo_with_error_pentanomial", "sprt_llr_pentanomial",
           "sprt_bounds", "sprt_verdict", "fmt_elo"]
