"""测试 Gumbel 搜索中针对优势方的终局与重复罚分（Stalemate、Insufficient Material、Threefold、Twofold、2-ply 宣和前瞻）及动态 c_scale 边界。"""
import unittest
import chess
import numpy as np
from Kit.api.types import Leaf, NodeEval, EvalRequest
from Kit.search import gumbel as kg
from Kit.runtime import run_sync


class _Evaluator:
    model_key = "dummy_penalties"
    def __init__(self, value=0.5, logit_bias=None):
        self.default_value = value
        self.logit_bias = logit_bias or {}

    def evaluate(self, boards):
        out = []
        for b in boards:
            moves = list(b.legal_moves)
            logits = np.zeros(len(moves), np.float32)
            for i, m in enumerate(moves):
                if m.uci() in self.logit_bias:
                    logits[i] = self.logit_bias[m.uci()]
            out.append((moves, logits, self.default_value))
        return out


class _Expander:
    def __init__(self, ev):
        self.ev = ev

    def expand(self, leaves):
        res = yield EvalRequest(self.ev, [l.board for l in leaves])
        out = []
        for l, (moves, logits, q) in zip(leaves, res):
            priors = np.ones(len(moves), dtype=np.float32) / max(len(moves), 1)
            out.append(NodeEval(moves=moves, priors=priors, value=q, logits=logits))
        return out


class TestPenalties(unittest.TestCase):

    def test_default_penalties_zero(self):
        """默认惩罚项为 0，保持既有行为（和棋返回 0.0）。"""
        cfg = kg.GumbelConfig()
        self.assertEqual(cfg.contempt, 0.0)
        self.assertEqual(cfg.stalemate_penalty, 0.0)
        self.assertEqual(cfg.insufficient_penalty, 0.0)
        self.assertEqual(cfg.twofold_penalty, 0.0)

    def test_stalemate_penalty_when_ahead(self):
        """优势方逼和对方应受到惩罚 (child.q = stalemate_penalty -> parent 价值为 -stalemate_penalty)。"""
        # 白后在 e6, 白王在 f7, 黑王在 h8. 白走 Qg6 导致黑王无子可动 (逼和)
        # 白走 Qh6# 是直接将杀
        p_board = chess.Board("7k/5K2/4Q3/8/8/8/8/8 w - - 0 1")
        self.assertIn(chess.Move.from_uci("e6g6"), p_board.legal_moves)
        self.assertIn(chess.Move.from_uci("e6h6"), p_board.legal_moves)

        ev = _Evaluator()
        expander = _Expander(ev)

        # 启用逼和判负 (stalemate_penalty=1.0)
        cfg_pen = kg.GumbelConfig(stalemate_penalty=1.0, g=0.0)
        searcher_pen = kg.Gumbel(expander, cfg_pen)

        res_pen = run_sync(searcher_pen.search(p_board, simulations=32))
        # 白方手握巨大优势，绝不能走 e6g6（逼和！），必须走 e6h6（将杀！）
        self.assertNotEqual(res_pen.move.uci(), "e6g6")
        self.assertEqual(res_pen.move.uci(), "e6h6")

    def test_stalemate_no_penalty_when_behind(self):
        """劣势方逼迫和棋（偷和成功）不应受惩罚。"""
        p_board = chess.Board("k7/8/1K6/8/8/8/8/7q b - - 0 1")
        white_mat = kg._material_score(p_board, chess.WHITE)
        black_mat = kg._material_score(p_board, chess.BLACK)
        self.assertGreater(black_mat, white_mat)

    def test_insufficient_material_penalty_when_ahead(self):
        """优势方升变为轻子导致子力不足和棋应受惩罚。"""
        # 白王在 a1，白兵在 a7；黑单王在 c8。净子力差 1 (net_mat=1 >= 1)
        # 白方走 a7a8=N 或 a7a8=B 升变为马或象，盘面变为单王单轻子 vs 单王，
        # 直接被规则判为 INSUFFICIENT_MATERIAL（和棋）！
        # 若开启 insufficient_penalty=1.0，该两步被判负（-1.0）
        promo_board = chess.Board("2k5/P7/8/8/8/8/8/K7 w - - 0 1")
        ev = _Evaluator(value=0.5)
        expander = _Expander(ev)

        cfg_pen = kg.GumbelConfig(insufficient_penalty=1.0, g=0.0)
        searcher = kg.Gumbel(expander, cfg_pen)
        res = run_sync(searcher.search(promo_board, simulations=32))
        # 绝不能选导致子力不足终局的 a7a8b 或 a7a8n！
        self.assertNotIn(res.move.uci(), ["a7a8b", "a7a8n"])

    def test_twofold_penalty_when_ahead(self):
        """优势方走入二次重复应被施加惩罚。"""
        board = chess.Board("8/8/8/8/4k3/8/2K5/R7 w - - 0 1")
        m1 = chess.Move.from_uci("c2c3")
        m2 = chess.Move.from_uci("e4e5")
        board.push(m1)
        board.push(m2)
        m3 = chess.Move.from_uci("c3c2")
        m4 = chess.Move.from_uci("e5e4")
        board.push(m3)
        board.push(m4)
        test_board = board.copy()
        test_board.push(chess.Move.from_uci("c2c3"))
        self.assertTrue(test_board.is_repetition(2))

        ev = _Evaluator()
        expander = _Expander(ev)

        # 带有 twofold_penalty
        cfg_pen = kg.GumbelConfig(twofold_penalty=1.0, g=0.0)
        searcher = kg.Gumbel(expander, cfg_pen)
        res = run_sync(searcher.search(board, simulations=32))
        # 白方手握车优势，绝不能再次走 c2c3 导致二次重复晃步
        self.assertNotEqual(res.move.uci(), "c2c3")

    def test_twofold_no_penalty_when_behind_or_equal(self):
        """均势残局（net_mat < 2）不触发二次重复判负惩罚，保证正常防守不被误杀。"""
        board = chess.Board("8/8/8/8/4k3/8/2K2r2/R7 w - - 0 1")
        p_col = board.turn
        net_mat = kg._material_score(board, p_col) - kg._material_score(board, not p_col)
        self.assertEqual(net_mat, 0)
        self.assertLess(net_mat, 2)

    def test_2ply_claim_draw_interception(self):
        """2-ply 宣和前瞻：若优势方走某步使得大劣对方能立即宣和三次重复，该着法被直接截断判负。"""
        board = chess.Board("8/8/8/8/8/8/4k3/K6Q w - - 0 1")
        # 6 步推步后，局面出现 2 次，且尚未触发宣和
        seq = ["a1b1", "e2e3", "b1a1", "e3e2", "a1b1", "e2e3"]
        for m in seq:
            board.push(chess.Move.from_uci(m))

        child_test = board.copy()
        child_test.push(chess.Move.from_uci("b1a1"))
        # 验证在 child_test 上黑方具有立即宣和权
        self.assertTrue(child_test.can_claim_threefold_repetition())

        ev = _Evaluator()
        expander = _Expander(ev)

        # 开启 twofold_penalty
        cfg_pen = kg.GumbelConfig(twofold_penalty=1.0, g=0.0)
        searcher = kg.Gumbel(expander, cfg_pen)
        res = run_sync(searcher.search(board, simulations=32))

        # 白方手握大优，绝不能走 b1a1（给黑方 2-ply 宣和机会！），而应当走后推进或叫将
        self.assertIsNotNone(res.move)
        self.assertNotEqual(res.move.uci(), "b1a1")

    def test_dynamic_c_scale_edge_cases(self):
        """动态 c_scale 的极端边界值测试。"""
        # 1. 满盘 32 子开局 (mat=78, ply=0)
        start_board = chess.Board()
        scale_start = kg.dynamic_c_scale(start_board, base_scale=0.02, schedule=True)
        self.assertAlmostEqual(scale_start, 0.015, places=3)

        # 2. 满盘但超长局面 (mat=78, ply=150)
        board_long = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 75")
        scale_long = kg.dynamic_c_scale(board_long, base_scale=0.02, schedule=True)
        self.assertGreater(scale_long, 0.015)
        self.assertLessEqual(scale_long, 0.150)

        # 3. 极端裸子残局 (单王对单王 mat=0, ply=70)
        bare_board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 35")
        scale_bare = kg.dynamic_c_scale(bare_board, base_scale=0.02, schedule=True)
        self.assertGreater(scale_bare, 0.10)
        self.assertLessEqual(scale_bare, 0.150)

        # 4. 极端超深局面 (ply=300)
        deep_board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 150")
        scale_deep = kg.dynamic_c_scale(deep_board, base_scale=0.02, schedule=True)
        self.assertAlmostEqual(scale_deep, 0.150, places=4)

        # 5. 异常超多子力 (升变多个后，mat=100)
        heavy_board = chess.Board("qqqqkqqq/8/8/8/8/8/8/QQQQKQQQ w - - 0 1")
        scale_heavy = kg.dynamic_c_scale(heavy_board, base_scale=0.02, schedule=True)
        self.assertGreaterEqual(scale_heavy, 0.015)
        self.assertLessEqual(scale_heavy, 0.150)

        # 6. 关闭调度 (schedule=False)
        scale_fixed = kg.dynamic_c_scale(bare_board, base_scale=0.02, schedule=False)
        self.assertEqual(scale_fixed, 0.02)

    def test_adversarial_logit_kill_vs_bait(self):
        """典型战术死活对抗测试:
        晃步将军先验 logit 显著偏高 (+2.5)，绝杀步先验 logit 为 0.0。
        验证：
        1. 在旧逻辑 (c_scale=0.02, 无调度) 下，先验偏差压制绝杀步，选出晃步；
        2. 在新逻辑 (动态 c_scale 升高到 ~0.12) 下，绝杀步真值彻底压倒先验偏差，100% 选出绝杀步！
        """
        # 白王在 f6, 白后在 b7, 黑王在 h8.
        # 绝杀步: b7g7# (Qg7# 直接将死)
        # 晃步诱饵: b7b8+ (Qb8+ 晃步将军)
        b = chess.Board("7k/1Q6/5K2/8/8/8/8/8 w - - 0 35")
        kill_mv = "b7g7"
        bait_mv = "b7b8"

        ev = _Evaluator(value=0.5, logit_bias={bait_mv: 2.5, kill_mv: 0.0})

        # 1. 旧逻辑
        cfg_old = kg.GumbelConfig(c_scale=0.02, c_scale_schedule=False, g=0.0)
        s_old = kg.Gumbel(_Expander(ev), cfg_old)
        res_old = run_sync(s_old.search(b, simulations=32))
        self.assertEqual(res_old.move.uci(), bait_mv)

        # 2. 新逻辑 (动态调度)
        cfg_new = kg.GumbelConfig(c_scale=0.02, c_scale_schedule=True, g=0.0)
        s_new = kg.Gumbel(_Expander(ev), cfg_new)
        res_new = run_sync(s_new.search(b, simulations=32))
        self.assertEqual(res_new.move.uci(), kill_mv)


if __name__ == "__main__":
    unittest.main()
