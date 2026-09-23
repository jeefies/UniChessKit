import unittest

import chess
import numpy as np

from unichess_kit.contrib import planes19 as p19
from unichess_kit.testing import ExpanderContract, FakePlanesEvaluator, contract_boards
from tests import _siblings


def sample_boards():
    boards = [b.copy() for b in contract_boards()]
    rep = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"):
        rep.push_san(san)
        boards.append(rep.copy())                 # 重复次数 0/1/2 都覆盖
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 57 60")
    boards += [b, b.mirror()]
    return boards


class TestEncoding(unittest.TestCase):
    def test_shape_and_orientation(self):
        x = p19.encode(chess.Board())
        self.assertEqual(x.shape, p19.INPUT_SHAPE)
        self.assertEqual(x.dtype, np.float32)
        b = chess.Board()
        b.push_san("e4")
        # 黑方走时镜像：黑方的兵在「我方兵」平面第 6 行（行棋方视角的第 2 横排）
        self.assertEqual(p19.encode(b)[0, 1].sum(), 8)

    def test_repetition_plane(self):
        b = chess.Board()
        vals = []
        for san in ("Nf3", "Nf6", "Ng1", "Ng8") * 2:
            b.push_san(san)
            vals.append(float(p19.encode(b)[18, 0, 0]))
        self.assertEqual(vals[3], 0.5)
        self.assertEqual(vals[7], 1.0)

    def test_priors(self):
        b = chess.Board("8/P7/8/8/8/8/6k1/4K3 w - - 0 1")
        policy = np.zeros(p19.POLICY_SIZE, dtype=np.float32)
        policy[p19.move_to_index(chess.Move.from_uci("a7a8"))] = 1.0
        promo = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        moves, pri = p19.priors_from_policy(b, policy, promo)
        got = {m.uci(): float(p) for m, p in zip(moves, pri) if p > 0}
        self.assertAlmostEqual(got["a7a8n"], 0.4, places=6)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=6)
        moves, pri = p19.priors_from_policy(b, np.zeros(p19.POLICY_SIZE), promo)
        self.assertTrue(np.allclose(pri, 1.0 / len(moves)))        # 全零 → 均匀

    def test_black_move_index_is_mirrored(self):
        b = chess.Board()
        b.push_san("e4")
        om = p19.orient_move(chess.Move.from_uci("e7e5"), b.turn)
        self.assertEqual(om.uci(), "e2e4")


class TestParityWithEngines(unittest.TestCase):
    """kit 的 planes19 与 R / T 各自的 core.encoding、core.moves 逐位一致。"""

    def check(self, which):
        mods = _siblings.load(which, "core.encoding", "core.moves")
        if mods is None:
            self.skipTest(f"找不到 {which} 仓库")
        enc, moves = mods
        for b in sample_boards():
            np.testing.assert_array_equal(p19.encode(b), enc.encode(b), err_msg=f"{which} {b.fen()}")
            for m in b.legal_moves:
                om = p19.orient_move(m, b.turn)
                self.assertEqual(om, enc.orient_move(m, b.turn))
                self.assertEqual(p19.move_to_index(om), moves.move_to_index(om))
                self.assertEqual(p19.move_to_promo_index(om), moves.move_to_promo_index(om))
        self.assertEqual(tuple(moves.PROMO_PIECES), p19.PROMO_PIECES)

    def test_r(self):
        self.check("R")

    def test_t(self):
        self.check("T")

    def test_priors_match_r(self):
        mods = _siblings.load("R", "search.mcts")
        if mods is None:
            self.skipTest("找不到 R 仓库")
        ev = FakePlanesEvaluator()
        for b in sample_boards():
            policy, promo, _ = ev._one(b)
            km, kp = p19.priors_from_policy(b, policy, promo)
            rm, rp = mods[0].priors_from_policy(b, policy, promo)
            self.assertEqual(km, rm)
            np.testing.assert_array_equal(kp, rp)


class TestExpander(ExpanderContract, unittest.TestCase):
    def make_expander(self):
        return p19.Planes19Expander(FakePlanesEvaluator())


class TestBatchFnEvaluator(unittest.TestCase):
    def test_chunking(self):
        fake = FakePlanesEvaluator()
        sizes = []

        def fn(boards):
            sizes.append(len(boards))
            return fake.evaluate_batch(boards)

        ev = p19.BatchFnEvaluator("k", fn, max_batch=3)
        boards = sample_boards()[:7]
        out = ev.evaluate(boards)
        self.assertEqual(sizes, [3, 3, 1])
        self.assertEqual(len(out), 7)
        for b, (pol, pro, wdl) in zip(boards, out):
            np.testing.assert_array_equal(pol, fake._one(b)[0])

    def test_bad_batch_size(self):
        ev = p19.BatchFnEvaluator("k", lambda bs: ([0], [0], [0]))
        with self.assertRaises(ValueError):
            ev.evaluate([chess.Board(), chess.Board()])


class TestFactory(unittest.TestCase):
    def test_players_share_evaluator(self):
        ev = FakePlanesEvaluator()
        f = p19.make_search_player_factory("s", ev, simulations=8, batch_size=4)
        a, b = f(), f()
        self.assertIsNot(a, b)
        self.assertIs(a.expander.evaluator, ev)
        self.assertIs(f.evaluator, ev)
        self.assertEqual(a.puct_cfg.simulations, 8)

    def test_missing_resources_raise(self):
        ev = FakePlanesEvaluator()
        with self.assertRaises(FileNotFoundError):
            p19.make_search_player_factory("s", ev, syzygy_path="/no/such/dir")
        with self.assertRaises(FileNotFoundError):
            p19.make_search_player_factory("s", ev, book_path="/no/such/book.bin")

    def test_unknown_puct_option_rejected(self):
        with self.assertRaises(TypeError):
            p19.make_search_player_factory("s", FakePlanesEvaluator(), c_puct=1.0)


if __name__ == "__main__":
    unittest.main()
