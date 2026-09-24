import time
import unittest

import chess
import numpy as np

from Kit.api import EvalRequest
from Kit.planes19 import Planes19Expander
from Kit.rules import TablebaseOracle
from Kit.runtime import run_sync
from Kit.search import PUCT, PUCTConfig
from Kit.testing import FakePlanesEvaluator
from Kit.tests import golden
from Kit.tests.test_rules import FakeTablebase

POSITIONS = (
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",   # Qxf7# 可杀
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",                                # 底线杀 Rd8#
    "8/P7/8/8/8/8/6k1/4K3 w - - 0 1",                                       # 升变
    "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R b KQkq - 3 8",   # 中局，黑方走
)


def make(sims=64, batch=16, salt="", oracle=None, seed=0, **kw):
    ev = FakePlanesEvaluator("fake", salt)
    return PUCT(Planes19Expander(ev), PUCTConfig(simulations=sims, batch_size=batch, **kw),
                oracle=oracle, rng=np.random.default_rng(seed)), ev


class TestPUCT(unittest.TestCase):
    def test_visit_accounting(self):
        for fen in POSITIONS:
            puct, _ = make(sims=100)
            root = run_sync(puct.search(chess.Board(fen)))
            self.assertEqual(int(root.N.sum()), 100, fen)
            self.assertEqual(puct.last_metrics["simulations"], 100)
            self.assertEqual(float(root.VL.sum()), 0.0)         # virtual loss 全部撤销

    def test_finds_mate_in_one(self):
        for fen, mate in ((POSITIONS[1], "f3f7"), (POSITIONS[2], "d1d8")):
            puct, _ = make(sims=200)
            mv, _ = run_sync(puct.best_move(chess.Board(fen)))
            self.assertEqual(mv.uci(), mate)

    def test_terminal_sims_consume_budget(self):
        """搜到杀棋后下探全撞终局节点：必须计入预算，否则空转（R 实测访问数涨到 7661）。"""
        puct, _ = make(sims=300, batch=32)
        root = run_sync(puct.search(chess.Board(POSITIONS[2])))
        self.assertEqual(int(root.N.sum()), 300)

    def test_batches_respect_batch_size_and_one_request_per_batch(self):
        ev = FakePlanesEvaluator("fake")
        puct = PUCT(Planes19Expander(ev), PUCTConfig(batch_size=16))
        gen = puct.search(chess.Board(), simulations=80)
        sizes = []
        try:
            req = gen.send(None)
            while True:
                self.assertIsInstance(req, EvalRequest)
                sizes.append(len(req.payloads))
                req = gen.send(ev.evaluate(req.payloads))
        except StopIteration:
            pass
        self.assertEqual(sizes[0], 1)                          # 根节点
        self.assertTrue(all(1 <= s <= 16 for s in sizes[1:]))
        self.assertGreater(max(sizes), 8)

    def test_root_min_visits(self):
        puct, _ = make(sims=60)
        root = run_sync(puct.search(chess.Board()))
        self.assertTrue((root.N >= 1).all())

    def test_advance_root_and_reuse(self):
        puct, _ = make(sims=100)
        board = chess.Board()
        mv, root = run_sync(puct.best_move(board))
        board.push(mv)
        child = PUCT.advance_root(root, mv)
        self.assertIsNotNone(child)
        before = int(child.N.sum())
        root2 = run_sync(puct.search(board, simulations=50, root=child))
        self.assertTrue(puct.last_metrics["reused_root"])
        self.assertEqual(int(root2.N.sum()), before + 50)
        self.assertIsNone(PUCT.advance_root(root, chess.Move.from_uci("a2a5")))

    def test_deadline_in_past(self):
        puct, _ = make(sims=100)
        root = run_sync(puct.search(chess.Board(), deadline=time.perf_counter() - 1))
        self.assertTrue(puct.last_metrics["stopped_early"])
        self.assertEqual(int(root.N.sum()), 0)
        mv, _ = run_sync(puct.best_move(chess.Board(), deadline=time.perf_counter() - 1))
        self.assertIn(mv, chess.Board().legal_moves)

    def test_noise_seeded(self):
        a, _ = make(sims=50, seed=3)
        b, _ = make(sims=50, seed=3)
        ra = run_sync(a.search(chess.Board(), add_noise=True))
        rb = run_sync(b.search(chess.Board(), add_noise=True))
        np.testing.assert_array_equal(ra.P, rb.P)
        np.testing.assert_array_equal(ra.N, rb.N)

    def test_temperature_sampling_seeded(self):
        moves = set()
        for seed in range(6):
            puct, _ = make(sims=60, seed=seed)
            mv, _ = run_sync(puct.best_move(chess.Board(), temperature=1.0))
            moves.add(mv)
        self.assertGreater(len(moves), 1)

    def test_root_top_k_limits_sampling(self):
        for k in (1, 2, 3):
            picked = set()
            for seed in range(20):
                puct, _ = make(sims=60, seed=seed, root_top_k=k)
                mv, root = run_sync(puct.best_move(chess.Board(), temperature=1.0))
                top = [root.moves[i] for i in np.argsort(-root.N, kind="stable")[:k]]
                self.assertIn(mv, top)
                picked.add(mv)
            if k == 1:
                self.assertEqual(len(picked), 1)
            else:
                self.assertGreater(len(picked), 1)

    def test_tablebase_root_fallback(self):
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        oracle = TablebaseOracle(FakeTablebase({"a1a2": (-2, 7)}, default=(0, 3), root=(2, 8)))
        puct, ev = make(sims=50, oracle=oracle)
        mv, root = run_sync(puct.best_move(board))
        self.assertEqual(root.terminal_value, 1.0)
        self.assertEqual(mv.uci(), "a1a2")
        self.assertEqual(ev.calls, [])                          # 根定值后不需要网络

    def test_tablebase_missing_child_dtz_research_without_tb(self):
        class NoDtz(FakeTablebase):
            def probe_wdl(self, board):
                if board.move_stack:
                    raise KeyError("child table missing")
                return 2
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        oracle = TablebaseOracle(NoDtz({}, default=(0, 0), root=(2, 8)))
        puct, ev = make(sims=40, oracle=oracle)
        mv, _ = run_sync(puct.best_move(board))
        self.assertIn(mv, board.legal_moves)
        self.assertGreater(len(ev.calls), 0)                    # 回退为无残局表的搜索


class TestParityWithR(unittest.TestCase):
    """kit PUCT 与旧 R ``search/mcts.py`` 的冻结整树（tests/fixtures/r_mcts）逐节点一致。"""

    @classmethod
    def setUpClass(cls):
        cls.g = golden.load("r_mcts")

    def make(self, sims, batch, seed=0, **kw):
        return PUCT(Planes19Expander(FakePlanesEvaluator("k")),
                    PUCTConfig(simulations=sims, batch_size=batch, **kw),
                    rng=np.random.default_rng(seed))

    def test_same_tree_and_move(self):
        self.assertEqual(len(self.g["cases"]), len(POSITIONS) * 3)
        for case in self.g["cases"]:
            k = self.make(case["sims"], case["batch"])
            km, kroot = run_sync(k.best_move(chess.Board(case["fen"])))
            self.assertEqual(km.uci(), case["move"], case["fen"])
            golden.assert_tree(self, kroot, case["tree"])
            for key, v in case["metrics"].items():
                self.assertEqual(k.last_metrics[key], v, (case["fen"], key))

    def test_same_with_noise_and_temperature(self):
        c = self.g["noise"]
        k = self.make(c["sims"], c["batch"], seed=c["seed"], temperature=c["temperature"])
        km, kroot = run_sync(k.best_move(chess.Board(c["fen"]), add_noise=True))
        self.assertEqual(km.uci(), c["move"])
        golden.assert_tree(self, kroot, c["tree"])

    def test_same_with_tree_reuse_over_a_game(self):
        c = self.g["reuse"]
        k = self.make(c["sims"], c["batch"])
        board, kroot = chess.Board(), None
        for step in c["steps"]:
            km, kroot = run_sync(k.best_move(board.copy(), root=kroot))
            self.assertEqual(km.uci(), step["move"], board.fen())
            golden.assert_tree(self, kroot, step["tree"])
            board.push(km)
            kroot = PUCT.advance_root(kroot, km)


if __name__ == "__main__":
    unittest.main()
