"""planes19 的损失与合法着法掩码。

关键不变量（R AGENTS 里记过的坑）：
- 掩码填充只能是 -1e4，不能用 -inf：策略标签为 0 的位置填 -inf 会算出 0 * -inf = NaN；
- ``legal_from_to_mask`` 是合法着法的**超集但绝不许漏**（漏一个 = 标签概率被算进非法类，
  softmax 之外的梯度全错）——随机对局 AIM 对照；
- R stage1 口径与 R 旧 ``model/train.py`` 的算式逐条对应（软 CE + 掩码归一 + 升变 CE 记 0）；
- R iteration 口径的升变按样本掩码平均（旧脚本把 -100 clamp 成 0 会污染，正确做法是掩码）；
- MLH 伪目标与 T ``train_p3_pretrain.py`` 的公式逐字对应。
"""
from __future__ import annotations

import unittest

import chess
import numpy as np
import torch

from Kit.planes19 import losses as L


def boards():
    b = chess.Board()
    out = [b.copy()]
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "O-O", "Nf6"]
    for san in moves:
        try:
            b.push_san(san)
        except Exception:
            break
        out.append(b.copy())
    for fen in ("8/P7/8/8/8/8/6k1/4K3 w - - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "5k2/8/8/K1Pp3r/8/8/8/8 b - - 0 1"):
        out.append(chess.Board(fen))
    return out


def planes_of(bs):
    from Kit.planes19.encoding import encode
    return torch.from_numpy(np.stack([encode(b) for b in bs])).float()


class TestLegalMask(unittest.TestCase):
    def test_mask_is_superset_of_legal_moves(self):
        for b in boards():
            x = planes_of([b])
            mask = L.legal_from_to_mask(x)
            legal_idx = set()
            for mv in b.legal_moves:
                om = mv if b.turn == chess.WHITE else chess.Move(mv.from_square ^ 56,
                                                                 mv.to_square ^ 56,
                                                                 promotion=mv.promotion)
                legal_idx.add(om.from_square * 64 + om.to_square)
            allowed = np.flatnonzero(mask[0].numpy())
            missing = legal_idx - set(int(i) for i in allowed)
            self.assertFalse(missing, f"{b.fen()} 漏了合法着法 {sorted(missing)[:5]}")

    def test_mask_shrinks_dimension(self):
        """初始局面实测约 768 个（from,to）可走（含自己的兵 / 王的全部邻格）——
        断言的是「数量级远小于 4096」而不是具体值。R stage1 的日志口径是约 690
        （legal_from_to_mask 只在 from 是我方格子且 to 不是我方格子时为真）。"""
        x = planes_of([chess.Board()])
        mask = L.legal_from_to_mask(x)
        d = float(mask.float().sum(1).mean())
        self.assertEqual(d, 768.0)
        self.assertLess(d, 4096)

    def test_no_nan_when_target_has_zeros(self):
        """历史坑：掩码填 -inf 时 0 * -inf = NaN。这里用 -1e4，目标为 0 的位置必须有限。"""
        torch.manual_seed(0)
        logits = torch.randn(64, 4096)
        target = torch.zeros(64, 4096)
        target[0, 100] = 1.0
        target[1, 200] = 0.5
        x = planes_of([chess.Board()] * 64)
        legal = L.legal_from_to_mask(x)
        loss = L.soft_cross_entropy(logits, target, mask=(target.sum(1) > 0), legal=legal)
        self.assertTrue(torch.isfinite(loss), loss)


class TestRLosses(unittest.TestCase):
    def _batch(self, n=64):
        from Kit.planes19.encoding import encode, move_to_index, orient_move
        b = chess.Board()
        bs = [b.copy() for _ in range(n)]
        x = planes_of(bs)
        p = np.zeros((n, 4096), np.float32)
        for i in range(n):
            mv = next(iter(b.legal_moves))
            p[i, move_to_index(orient_move(mv, chess.WHITE))] = 1.0
        pr = np.full(n, -100, np.int64)
        pr[0] = 0
        w = np.zeros((n, 3), np.float32)
        w[:, 0] = 0.6
        w[:, 1] = 0.3
        w[:, 2] = 0.1
        return x, torch.from_numpy(p), torch.from_numpy(pr), torch.from_numpy(w)

    def test_r_stage1_matches_reference_expression(self):
        x, p, pr, w = self._batch()
        legal = L.legal_from_to_mask(x)
        logits = torch.zeros(x.shape[0], 4096)
        promo = torch.zeros(x.shape[0], 4)
        wdl = torch.zeros(x.shape[0], 3)
        with torch.no_grad():
            loss, parts = L.r_stage1_loss(logits, promo, wdl, p, pr, w, legal=legal)
            # 手工重算（gray 的写法）
            has = p.sum(1) > 1e-6
            lp = L.soft_cross_entropy(logits.masked_fill(~legal, -1e4), p, has)
            lw = L.soft_cross_entropy(wdl, w)
            lpr = torch.nn.functional.cross_entropy(promo, pr, ignore_index=-100)
        self.assertAlmostEqual(float(loss), float(lp + lw + 0.1 * lpr), places=12)
        self.assertEqual(set(parts) - {"has_policy"}, {"policy", "value", "promo"})
        # 整批没有升变样本时 promo 记 0，不是 NaN
        pr2 = torch.full_like(pr, -100)
        with torch.no_grad():
            loss2, _ = L.r_stage1_loss(logits, promo, wdl, p, pr2, w, legal=legal)
        self.assertTrue(torch.isfinite(loss2))

    def test_r_iter_promo_masked_not_clamped(self):
        from Kit.planes19.encoding import encode, move_to_index, orient_move
        b = chess.Board()
        bs = [b.copy() for _ in range(8)]
        x = planes_of(bs)
        p = np.zeros((8, 4096), np.float32)
        for i in range(8):
            mv = next(iter(b.legal_moves))
            p[i, move_to_index(orient_move(mv, chess.WHITE))] = 1.0
        pr = torch.full((8,), -100, dtype=torch.int64)
        w = torch.zeros(8, 3)
        w[:, 0] = 0.6
        w[:, 1] = 0.3
        w[:, 2] = 0.1
        logits = torch.zeros(8, 4096)
        promo = torch.zeros(8, 4)
        wdl = torch.zeros(8, 3)
        with torch.no_grad():
            loss, parts = L.r_iter_loss(logits, promo, wdl, torch.from_numpy(p), pr, w)
        # 全部 -100：clamp 到 0 会让分母变成 8 而不是 1，掩码写法下 prom 损失为 0
        self.assertEqual(float(parts["promo"]), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_mlh_target_matches_t_formula(self):
        x = planes_of([chess.Board()])
        w = torch.tensor([[0.7, 0.2, 0.1]])
        t = L.mlh_target(x, w)
        counts = float(x[:, :12].sum())
        q = abs(0.7 - 0.1)
        self.assertAlmostEqual(float(t[0]), (2 * counts + 20 * (1 - q)) / 100, places=6)


class TestChessLoss(unittest.TestCase):
    def test_forward_and_metrics(self):
        torch.manual_seed(0)
        n = 8
        pl = torch.randn(n, 4096)
        prl = torch.randn(n, 4)
        wl = torch.randn(n, 3)
        t = torch.zeros(n, 4096)
        t[:, 0] = 0.5
        t[:, 1] = 0.5
        pr = torch.full((n,), -100, dtype=torch.int64)
        pr[0] = 2
        w = torch.full((n, 3), -1e9)
        w[:, 0] = 1.0
        out = L.ChessLoss()(pl, prl, wl, t, pr, w)
        self.assertTrue(torch.isfinite(out.total_loss))
        self.assertGreater(float(out.policy_loss), 0.0)
        self.assertIsNone(out.mlh_loss)
        # 无 MLH 时 total = policy + 0.1·promo + wdl
        self.assertAlmostEqual(
            float(out.total_loss),
            float(out.policy_loss + 0.1 * out.promo_loss + out.wdl_loss), places=6)
        out2 = L.ChessLoss(mlh_weight=0.05)(pl, prl, wl, t, pr, w, torch.zeros(n),
                                          torch.zeros(n))
        self.assertIsNotNone(out2.mlh_loss)
        self.assertIn("mlh", out2.parts)
        # MLH 目标全 0、预测全 0 → smooth_l1 = 0，总量与非 MLH 版相同
        self.assertAlmostEqual(float(out2.total_loss), float(out.total_loss), places=6)

    def test_top_metrics(self):
        torch.manual_seed(0)
        n = 16
        t = torch.zeros(n, 4096)
        t[:, 9] = 1.0
        logits = torch.full((n, 4096), -3.0)
        logits[:, 9] = 5.0
        m = L.policy_metrics(logits, t)
        self.assertEqual(float(m["top1"]), 1.0)
        self.assertEqual(float(m["top5"]), 1.0)
        # 非法位置的 logit 没训练过、是噪声：top5 只在与标签候选集合（这里只有 1 项）比较时为 0
        logits[:, 500] = 9.0
        m = L.policy_metrics(logits, t)
        self.assertEqual(float(m["top1"]), 0.0)

    def test_policy_loss_type_switches_objective(self):
        """``policy_loss_type`` 真的切换目标：kl_divergence 与 cross_entropy 数值不同。

        P4 自对弈配方用 KL（MCTS 访问分布是软目标）；默认仍是 CE。
        """
        torch.manual_seed(0)
        n, moves = 8, 4096
        pl = torch.randn(n, moves) * 3
        wl = torch.randn(n, 3)
        t = torch.zeros(n, moves)
        t[:, 0] = 0.7
        t[:, 1] = 0.3
        pr = torch.full((n,), -100, dtype=torch.int64)
        w = torch.full((n, 3), -1e9)
        w[:, 0] = 1.0
        out = L.ChessLoss()(pl, wl[:, :4], wl, t, pr, w)
        kl = L.ChessLoss(policy_loss_type="kl_divergence")(pl, wl[:, :4], wl, t, pr, w)
        self.assertTrue(torch.isfinite(out.policy_loss))
        self.assertTrue(torch.isfinite(kl.policy_loss))
        self.assertNotAlmostEqual(float(out.policy_loss), float(kl.policy_loss), places=5)
        with self.assertRaises(ValueError):
            L.ChessLoss(policy_loss_type="mse")

    def test_default_policy_loss_type_is_cross_entropy(self):
        self.assertEqual(L.ChessLoss().policy_loss_type, "cross_entropy")


if __name__ == "__main__":
    unittest.main()
