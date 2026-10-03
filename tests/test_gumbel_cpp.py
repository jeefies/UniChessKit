"""C++ Gumbel 节点级算术 vs Python 参考实现（数值等价契约，非逐位一致）。

契约（puct_native.cpp 顶部同款）：决策一致率 ≥ 99.9%、π′ max|Δ| ≤ 1e-5、
softmax max|Δ| ≤ 1e-6、top-m 顺序逐项相同。逐位不可达的原因：numpy 的 fp32 exp
（SIMD 多项式）与 BLAS sdot 跟 libm / 手写累加在最后 1 ULP 上不同，而 Gumbel 热路径
每步都要 softmax。
"""
from __future__ import annotations

import ctypes
import unittest

import chess
import numpy as np

from Kit.search import gumbel as G
from Kit.search import native

F32 = ctypes.POINTER(ctypes.c_float)
I64 = ctypes.POINTER(ctypes.c_int64)
I32 = ctypes.POINTER(ctypes.c_int32)


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def i32(a):
    return np.ascontiguousarray(a, dtype=np.int32)


class TestGumbelCppArithmetic(unittest.TestCase):
    N_NODES = 3000

    @classmethod
    def setUpClass(cls):
        cls.lib = native.lib()
        cls.c_visit, cls.c_scale = G.C_VISIT, G.C_SCALE

    def _nodes(self, seed=0):
        """随机节点 battery：覆盖全未访问 / 全 1 次 / 随机访问 / 平局。"""
        rng = np.random.default_rng(seed)
        nodes = []
        for t in range(self.N_NODES):
            n = int(rng.integers(1, 48))
            logits = f32(rng.normal(0, 3.0, n))
            if t % 7 == 0:
                logits[:] = 0.0                      # 平局
            N = np.zeros(n, np.int64)
            mode = t % 5
            if mode == 0:
                pass                                  # 全未访问
            elif mode == 1:
                N[:] = 1                              # 全访问 1 次
            else:
                k = int(rng.integers(0, n + 1))
                if k:
                    N[rng.choice(n, size=k, replace=False)] = rng.integers(1, 60, size=k)
            QSUM = f32(rng.normal(0, 0.5, n) * np.maximum(N, 1))
            q = float(rng.normal(0, 0.4))
            nodes.append(G.Node(legal=np.arange(n, dtype=np.int64), logits=logits,
                                q=q, n=N.copy(), q_sum=QSUM.copy()))
        return nodes

    def test_select_action_agreement(self):
        out = ctypes.c_int()
        bad = 0
        for node in self._nodes(1):
            want = G.select_action(node, self.c_visit, self.c_scale)
            lg, nn, qs = f32(node.logits), i64(node.n), f32(node.q_sum)
            rc = self.lib.kg_select_action(lg.ctypes.data_as(F32), len(lg),
                                           nn.ctypes.data_as(I64), qs.ctypes.data_as(F32),
                                           ctypes.c_double(node.q),
                                           ctypes.c_double(self.c_visit),
                                           ctypes.c_double(self.c_scale), ctypes.byref(out))
            self.assertEqual(rc, 0)
            if out.value != want:
                bad += 1
        rate = 1.0 - bad / self.N_NODES
        self.assertGreaterEqual(rate, 0.999, f"决策一致率 {rate:.4f}（{bad}/{self.N_NODES} 不同）")

    def test_pi_prime_close(self):
        worst = 0.0
        for node in self._nodes(2):
            m = len(node.logits)
            want = G.pi_prime(node, self.c_visit, self.c_scale)
            got = np.zeros(m, np.float32)
            lg, nn, qs = f32(node.logits), i64(node.n), f32(node.q_sum)
            rc = self.lib.kg_pi_prime(lg.ctypes.data_as(F32), m, nn.ctypes.data_as(I64),
                                      qs.ctypes.data_as(F32), ctypes.c_double(node.q),
                                      ctypes.c_double(self.c_visit),
                                      ctypes.c_double(self.c_scale),
                                      got.ctypes.data_as(F32))
            self.assertEqual(rc, 0)
            worst = max(worst, float(np.abs(got - want).max()))
        self.assertLessEqual(worst, 1e-5, f"π′ max|Δ| = {worst:.3e}")

    def test_softmax_close(self):
        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(500):
            n = int(rng.integers(1, 64))
            x = f32(rng.normal(0, 4.0, n))
            want = G.softmax(x)
            got = np.zeros(n, np.float32)
            rc = self.lib.kg_softmax(x.ctypes.data_as(F32), n, got.ctypes.data_as(F32))
            self.assertEqual(rc, 0)
            worst = max(worst, float(np.abs(got - want).max()))
        self.assertLessEqual(worst, 1e-6, f"softmax max|Δ| = {worst:.3e}")

    def test_topm_matches_stable_argsort(self):
        rng = np.random.default_rng(4)
        for _ in range(500):
            n = int(rng.integers(1, 48))
            m0 = int(rng.integers(1, 20))
            logits = f32(rng.normal(0, 2.0, n))
            noise = f32(rng.random(n))
            m = min(m0, n)
            want = np.argsort(-(noise + logits), kind="stable")[:m].tolist()
            out = np.zeros(m, np.int32)
            rc = self.lib.kg_topm(logits.ctypes.data_as(F32), n,
                                  noise.ctypes.data_as(F32), m0, out.ctypes.data_as(I32))
            self.assertEqual(rc, m)
            self.assertEqual(out.tolist(), want)

    def test_edges(self):
        # 单候选
        lg = f32(np.zeros(1)); nn = i64(np.zeros(1)); qs = f32(np.zeros(1))
        out = ctypes.c_int()
        rc = self.lib.kg_select_action(lg.ctypes.data_as(F32), 1, nn.ctypes.data_as(I64),
                                       qs.ctypes.data_as(F32), ctypes.c_double(0.0),
                                       ctypes.c_double(self.c_visit),
                                       ctypes.c_double(self.c_scale), ctypes.byref(out))
        self.assertEqual(rc, 0)
        self.assertEqual(out.value, 0)
        # 全平局全未访问：argmax 取第一个
        n = 8
        node = G.Node(legal=np.arange(n, dtype=np.int64), logits=np.zeros(n, np.float32),
                      q=0.0, n=np.zeros(n, np.int64), q_sum=np.zeros(n, np.float32))
        self.assertEqual(G.select_action(node, self.c_visit, self.c_scale), 0)
        lg, nn, qs = f32(node.logits), i64(node.n), f32(node.q_sum)
        rc = self.lib.kg_select_action(lg.ctypes.data_as(F32), n, nn.ctypes.data_as(I64),
                                       qs.ctypes.data_as(F32), ctypes.c_double(0.0),
                                       ctypes.c_double(self.c_visit),
                                       ctypes.c_double(self.c_scale), ctypes.byref(out))
        self.assertEqual(rc, 0)
        self.assertEqual(out.value, 0)


class TestGumbelCppSearchParity(unittest.TestCase):
    """同一假模型 + 同一噪声：C++ 与 Python 搜索的决策/π′/统计一致（数值等价）。"""

    FENS = (
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P7/8/8/8/8/6kp/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 140 120",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    )

    def _run_pair(self, board, sims=64, m0=8, seed=5):
        from Kit.planes19 import BatchFnEvaluator, Planes19Expander
        from Kit.runtime import run_sync
        from Kit.search.gumbel import Gumbel, GumbelConfig
        from Kit.search.gumbel_cpp import GumbelCpp
        from Kit.testing import FakePlanes19Model

        n_legal = len(list(board.legal_moves))
        u = np.random.default_rng(seed).random(n_legal).astype(np.float32)

        class _R:
            def random(self, n, dtype=None):
                return u

        fake = FakePlanes19Model(salt="gcpp")
        py = Gumbel(Planes19Expander(BatchFnEvaluator("fk:py", fake.evaluate_batch)),
                    GumbelConfig(simulations=sims, m0=m0, g=1.0))
        a = run_sync(py.search(board, rng=_R()))
        cpp = GumbelCpp(BatchFnEvaluator("fk:cpp", fake.evaluate_planes),
                        GumbelConfig(simulations=sims, m0=m0, g=1.0))
        b = run_sync(cpp.search(board, rng=_R(), simulations=sims))
        return a, b

    def test_action_and_stats(self):
        for fen in self.FENS:
            a, b = self._run_pair(chess.Board(fen))
            self.assertEqual(a.stats["sims_used"], b.stats["sims_used"], fen)
            if "rounds" in a.stats:
                self.assertIn("rounds", b.stats, fen)
                for k in ("sims_used", "rounds", "n_nodes", "n_terminal", "max_depth"):
                    self.assertEqual(a.stats[k], b.stats[k], f"{fen} {k}")
            self.assertEqual(a.move, b.move, fen)

    def test_pi_prime_close(self):
        cfg = G.GumbelConfig()
        worst = 0.0
        for fen in self.FENS:
            a, b = self._run_pair(chess.Board(fen))
            if a.root is None:
                self.assertIsNone(b.root, fen)
                continue
            _, pa = a.pi_prime(cfg)
            _, pb = b.pi_prime(cfg)
            self.assertEqual(len(pa), len(pb), fen)
            if len(pa):
                worst = max(worst, float(np.abs(pa - pb).max()))
        self.assertLessEqual(worst, 2e-5, f"π′ max|Δ| = {worst:.3e}")

    def test_terminal_root(self):
        # 被将死局面：两侧都应无着法（Python 走 fast_outcome 早退，C++ 在 kg_begin 里判）
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertIsNotNone(board.outcome(claim_draw=True))
        a, b = self._run_pair(board, sims=8)
        self.assertIsNone(a.move)
        self.assertIsNone(b.move)


if __name__ == "__main__":
    unittest.main()