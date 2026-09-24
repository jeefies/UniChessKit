"""T/R 监督 / 自对弈训练的损失（torch）。

三种口径，都与旧脚本逐位一致（旧 → 新对照见各函数）：

- ``r_stage1_loss``：R ``model/train.py``（stage1）。策略软 CE 可限制在合法着法上归一，
  价值软 CE，升变 ``F.cross_entropy(ignore_index=-100)``（整批无升变样本时记 0）。
- ``r_iter_loss``：R ``train_iteration_fast.py``（iteration46）。各项先转 fp32 再算，
  升变按样本掩码平均。
- ``ChessLoss``：T ``model/loss.py``（t20m / stratified / curriculum / p3 MLH）。

历史坑：掩码填充值只能是 -1e4，**绝不能用 -inf**——被掩位置的目标恰好为 0，0 * -inf = NaN。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

MASK_FILL = -1e4


def legal_from_to_mask(x: torch.Tensor) -> torch.Tensor:
    """从输入平面推出策略掩码 [N,19,8,8] → bool [N,4096]（from*64+to）。

    是合法着法集合的**超集**：起点有我方子、落点无我方子。漏掉合法着法会算错梯度，
    多留非法着法只是少省一点。标准棋下已在 716,540 个合法着法上验证零误杀
    （``tests/test_planes19_train.py`` 在随机对局上回归）。平均把 4096 维压到约 686 维。
    """
    ours = x[:, 0:6].amax(dim=1).flatten(1) > 0.5
    return (ours.unsqueeze(2) & (~ours).unsqueeze(1)).flatten(1)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None,
                       legal: Optional[torch.Tensor] = None) -> torch.Tensor:
    """软标签交叉熵（R stage1 口径）。mask [N] 样本级开关；legal [N,C] 限制 softmax 归一范围。"""
    if legal is not None:
        logits = logits.masked_fill(~legal, MASK_FILL)
    per = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    if mask is None:
        return per.mean()
    m = mask.float()
    denom = m.sum()
    if denom < 1:
        return torch.zeros((), device=logits.device, dtype=per.dtype)
    return (per * m).sum() / denom


def r_stage1_loss(policy_logits, promo_logits, wdl_logits, p_t, pr_t, w_t, *,
                  legal=None, value_weight: float = 1.0, promo_weight: float = 0.1):
    """→ (loss, {"policy", "value", "promo", "has_policy"})。须在 autocast 内调用（与旧脚本相同）。"""
    has_policy = p_t.sum(dim=1) > 1e-6
    loss_p = soft_cross_entropy(policy_logits, p_t, has_policy, legal)
    loss_w = soft_cross_entropy(wdl_logits, w_t)
    loss_pr = F.cross_entropy(promo_logits, pr_t, ignore_index=-100)
    if torch.isnan(loss_pr):                       # 这一批没有升变样本
        loss_pr = torch.zeros((), device=policy_logits.device)
    loss = loss_p + value_weight * loss_w + promo_weight * loss_pr
    return loss, {"policy": loss_p, "value": loss_w, "promo": loss_pr, "has_policy": has_policy}


def r_iter_loss(policy_logits, promo_logits, wdl_logits, p, pr, w, *, legal=None):
    """R iteration46 口径 → (loss, {"policy", "value", "promo"})。"""
    has = p.sum(1) > 0
    pl = policy_logits
    if legal is not None:
        pl = pl.float().masked_fill(~legal, MASK_FILL)
    lp = (-(p * F.log_softmax(pl.float(), dim=1)).sum(1) * has).sum() / has.sum().clamp_min(1)
    lw = -(w * F.log_softmax(wdl_logits.float(), dim=1)).sum(1).mean()
    eligible = pr != -100
    lpr = (F.cross_entropy(promo_logits.float(), pr.clamp_min(0), reduction="none") * eligible
           ).sum() / eligible.sum().clamp_min(1)
    loss = lp + lw + 0.1 * lpr
    return loss, {"policy": lp, "value": lw, "promo": lpr}


def mlh_target(x: torch.Tensor, wdl_target: torch.Tensor) -> torch.Tensor:
    """T P3 的 moves-left 伪目标：(2·子力数 + 20·(1 − |W − L|)) / 100。"""
    piece_counts = x[:, :12].sum(dim=(1, 2, 3))
    q_target = (wdl_target[:, 0] - wdl_target[:, 2]).abs()
    return (2.0 * piece_counts + 20.0 * (1.0 - q_target)) / 100.0


@dataclass
class ChessLossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    promo_loss: torch.Tensor
    wdl_loss: torch.Tensor
    mlh_loss: Optional[torch.Tensor] = None
    parts: dict = field(default_factory=dict)


class ChessLoss:
    """T 口径：策略 ``F.cross_entropy``（软目标）、有效子集上的升变 CE、WDL 软 CE、可选 MLH smooth-L1。"""

    def __init__(self, policy_weight=1.0, promo_weight=0.1, wdl_weight=1.0, mlh_weight=0.05,
                 policy_loss_type="cross_entropy"):
        if policy_loss_type not in ("cross_entropy", "kl_divergence"):
            raise ValueError(f"未知 policy_loss_type {policy_loss_type!r}")
        self.policy_weight = policy_weight
        self.promo_weight = promo_weight
        self.wdl_weight = wdl_weight
        self.mlh_weight = mlh_weight
        self.policy_loss_type = policy_loss_type

    def __call__(self, policy_logits, promo_logits, value_wdl, policy_target, promo_target,
                 wdl_target, mlh_logits=None, mlh_target=None) -> ChessLossOutput:
        if self.policy_loss_type == "kl_divergence":
            loss_policy = F.kl_div(F.log_softmax(policy_logits, dim=-1), policy_target,
                                   reduction="batchmean", log_target=False)
        else:
            loss_policy = F.cross_entropy(policy_logits, policy_target)
        valid_promo = promo_target != -100
        if valid_promo.any():
            loss_promo = F.cross_entropy(promo_logits[valid_promo], promo_target[valid_promo])
        else:
            loss_promo = torch.tensor(0.0, device=promo_logits.device, dtype=promo_logits.dtype)
        loss_wdl = F.cross_entropy(value_wdl, wdl_target)
        total = (self.policy_weight * loss_policy + self.promo_weight * loss_promo
                 + self.wdl_weight * loss_wdl)
        loss_mlh = None
        if mlh_logits is not None and mlh_target is not None:
            loss_mlh = F.smooth_l1_loss(mlh_logits, mlh_target)
            total = total + self.mlh_weight * loss_mlh
        parts = {"policy": loss_policy, "promo": loss_promo, "value": loss_wdl}
        if loss_mlh is not None:
            parts["mlh"] = loss_mlh
        return ChessLossOutput(total, loss_policy, loss_promo, loss_wdl, loss_mlh, parts)


@torch.no_grad()
def policy_metrics(policy_logits, policy_target, legal=None) -> dict:
    """top1（与标签最大概率着法一致）/ top5（落在标签前 5 内），只在有策略标签的样本上统计。"""
    logits = policy_logits.detach().float()
    if legal is not None:
        logits = logits.masked_fill(~legal, MASK_FILL)
    valid = policy_target.sum(dim=-1) > 0
    if not bool(valid.any()):
        return {}
    pred = logits.argmax(dim=-1)
    top1 = (pred[valid] == policy_target.argmax(dim=-1)[valid]).float().mean()
    top5_idx = policy_target.topk(5, dim=-1).indices
    top5 = (pred.unsqueeze(-1) == top5_idx).any(-1)[valid].float().mean()
    return {"top1": top1, "top5": top5}
