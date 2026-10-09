"""测试 Gumbel 搜索中针对优势方的终局与重复罚分（Stalemate、Insufficient Material、Threefold、Twofold）。"""
import unittest
import chess
import numpy as np
from Kit.api.types import Leaf, NodeEval, EvalRequest
from Kit.search import gumbel as kg
from Kit.runtime import run_sync


class _Evaluator:
    model_key = "dummy_penalties"
    def evaluate(self, boards):
        out = []
        for b in boards:
            moves = list(b.legal_moves)
            logits = np.zeros(len(moves), np.float32)
            out.append((moves, logits, 0.5))
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

    def test_twofold_penalty_when_ahead(self):
        """优势方走入二次重复应被施加惩罚。"""
        # 构造一个白方车+王 vs 单王残局
        board = chess.Board("8/8/8/8/4k3/8/2K5/R7 w - - 0 1")
        # 初始走几步制造重复局面
        m1 = chess.Move.from_uci("c2c3")
        m2 = chess.Move.from_uci("e4e5")
        board.push(m1)
        board.push(m2)
        # 退回
        m3 = chess.Move.from_uci("c3c2")
        m4 = chess.Move.from_uci("e5e4")
        board.push(m3)
        board.push(m4)
        # 此时若白方再次走 c2c3，该局面将在历史上第 2 次出现！
        test_board = board.copy()
        test_board.push(chess.Move.from_uci("c2c3"))
        self.assertTrue(test_board.is_repetition(2))

        ev = _Evaluator()
        expander = _Expander(ev)

        # 带有 twofold_penalty
        cfg_pen = kg.GumbelConfig(twofold_penalty=0.5, g=0.0)
        searcher = kg.Gumbel(expander, cfg_pen)
        res = run_sync(searcher.search(board, simulations=32))
        # 白方不应该再次走 c2c3 倒角，而应该走车或别的推进一步
        self.assertNotEqual(res.move.uci(), "c2c3")


if __name__ == "__main__":
    unittest.main()
