"""分片验收：抽样检查每条记录的标签是否自洽。

- 策略着法（带 promo 字段试；非首选着法逐个升变子试）在该局面合法——漏掉 promo 会把 g7f8n 误判成 g7f8；
- 策略概率之和 = 1（监督记录截断到前 5 条后仍须归一）；PGN 分片策略全 0 视为「无策略标签」；
- WDL 之和 = 1（自对弈记录可全 0）；
- 易位索引是「王走两格」写法（4*64+6 / 4*64+2），不得出现旧的 4*64+7 / 4*64+0 易位写法。
"""
from __future__ import annotations

import numpy as np

from ..encoding import index_to_move, unorient_move
from ..records import NO_PROMO, open_shard, record_policy, record_to_board, record_wdl

_E1 = 4


def check_shard(path, sample: int = 50000) -> dict:
    a = open_shard(path)
    n = min(int(sample), len(a))
    bad_legal = bad_pol = bad_wdl = bad_castle = promo_seen = no_policy = 0
    for r in a[:n]:
        b = record_to_board(r)
        pol = record_policy(r)
        promo = int(r["promo"])
        if not pol:
            no_policy += 1
        for j, (idx, _) in enumerate(pol):
            mv = unorient_move(index_to_move(idx, promo if (j == 0 and promo != NO_PROMO) else None),
                               b.turn)
            if mv not in b.legal_moves and not any(
                    unorient_move(index_to_move(idx, k), b.turn) in b.legal_moves for k in range(4)):
                bad_legal += 1
                break
            if idx in (_E1 * 64 + 7, _E1 * 64 + 0) and b.is_castling(mv):
                bad_castle += 1
        if promo != NO_PROMO:
            promo_seen += 1
        if pol and not (0.995 < sum(p for _, p in pol) < 1.005):
            bad_pol += 1
        w = record_wdl(r)
        if sum(w) and not (0.995 < sum(w) < 1.005):
            bad_wdl += 1
    ok = bad_legal == 0 and bad_pol == 0 and bad_wdl == 0 and bad_castle == 0
    return {"path": str(path), "checked": n, "total": len(a), "illegal": bad_legal,
            "policy_unnormalized": bad_pol, "wdl_unnormalized": bad_wdl,
            "castling_rook_form": bad_castle, "promo": promo_seen, "no_policy": no_policy,
            "ok": ok}
