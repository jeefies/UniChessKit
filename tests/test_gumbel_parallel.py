"""P4：``api.gather`` 与 Gumbel 轮内并发模拟（``GumbelConfig.parallel``）。

并发只改变请求的拼批方式，不改变搜索：用与批无关的假评估器，并发与串行的选着、根访问数、
Q 累加、π′、展开直方图逐位相同，而驱动拍数显著减少。
"""
from __future__ import annotations

import unittest

import chess
import numpy as np

from Kit.api import EvalRequest, gather
from Kit.planes19 import Planes19Expander
from Kit.runtime import Batcher, run_sync
from Kit.search.gumbel import Gumbel, GumbelConfig
from Kit.testing.fakes import FakePlanesEvaluator


class _Echo:
    model_key = "echo"

    def __init__(self):
        self.calls = []

    def evaluate(self, payloads):
        self.calls.append(list(payloads))
        return [p * 10 for p in payloads]


def _asks(ev, values):
    """依次各请求一次，返回结果列表。"""
    out = []
    for v in values:
        (r,) = yield EvalRequest(ev, [v])
        out.append(r)
    return out


class GatherTest(unittest.TestCase):
    def test_merges_per_tick_and_keeps_order(self):
        ev = _Echo()
        res = run_sync(gather([_asks(ev, [1, 2, 3]), _asks(ev, [4]), _asks(ev, [])]))
        self.assertEqual(res, [[10, 20, 30], [40], []])
        self.assertEqual(ev.calls, [[1, 4], [2], [3]])

    def test_multi_payload_requests_are_split_back(self):
        ev = _Echo()

        def two():
            r = yield EvalRequest(ev, [5, 6])
            return list(r)

        res = run_sync(gather([two(), _asks(ev, [7])]))
        self.assertEqual(res, [[50, 60], [70]])
        self.assertEqual(ev.calls, [[5, 6, 7]])

    def test_distinct_evaluators_are_not_mixed(self):
        a, b = _Echo(), _Echo()
        b.model_key = "echo-b"
        batcher = Batcher()
        res = run_sync(gather([_asks(a, [1]), _asks(b, [2]), _asks(a, [3])]), batcher)
        self.assertEqual(res, [[10], [20], [30]])
        self.assertEqual(a.calls, [[1, 3]])
        self.assertEqual(b.calls, [[2]])

    def test_error_closes_siblings(self):
        ev = _Echo()
        closed = []

        def victim():
            try:
                yield EvalRequest(ev, [1])
                yield EvalRequest(ev, [2])
            finally:
                closed.append(True)

        def boom():
            yield EvalRequest(ev, [3])
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            run_sync(gather([victim(), boom()]))
        self.assertEqual(closed, [True])


def _search(board, parallel, sims=64, m0=16, g=0.0, seed=7):
    ev = FakePlanesEvaluator()
    gumbel = Gumbel(Planes19Expander(ev), GumbelConfig(simulations=sims, m0=m0, g=g,
                                                         parallel=parallel))
    batcher = Batcher()
    res = run_sync(gumbel.search(board, rng=np.random.default_rng(seed)), batcher)
    return res, batcher.stats, ev


class ParallelSearchTest(unittest.TestCase):
    BOARDS = [
        chess.Board(),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
        chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),          # 一步杀
        chess.Board("8/8/4k3/8/2P5/8/4K3/8 w - - 0 1"),                # 合法着 < m0
    ]

    def test_identical_to_serial(self):
        for board in self.BOARDS:
            for sims, g in ((64, 0.0), (256, 1.0), (37, 1.0)):
                with self.subTest(fen=board.fen(), sims=sims, g=g):
                    a, sa, _ = _search(board, False, sims=sims, g=g)
                    b, sb, _ = _search(board, True, sims=sims, g=g)
                    self.assertEqual(a.move, b.move)
                    np.testing.assert_array_equal(a.root.n, b.root.n)
                    self.assertEqual(a.root.q_sum.tobytes(), b.root.q_sum.tobytes())
                    self.assertEqual(a.pi_prime(GumbelConfig())[1].tobytes(),
                                     b.pi_prime(GumbelConfig())[1].tobytes())
                    for k in ("sims_used", "n_nodes", "n_terminal", "max_depth", "expand_hist",
                              "survivors_per_round", "qmin", "qmax"):
                        self.assertEqual(a.stats[k], b.stats[k], k)
                    self.assertEqual(sa.positions, sb.positions)
                    self.assertLessEqual(sb.ticks, sa.ticks)

    def test_fewer_ticks_bigger_batches(self):
        a, sa, _ = _search(chess.Board(), False, sims=256)
        b, sb, ev = _search(chess.Board(), True, sims=256)
        # 256 模拟 / m0=16：串行每次模拟一拍；并发每轮 ⌈预算/候选⌉ 拍 = 4+8+16+32；另加根评估 1 拍
        self.assertEqual(sa.ticks, 1 + 256)
        self.assertEqual(sb.ticks, 1 + 60)
        self.assertEqual(max(ev.calls), 16)


if __name__ == "__main__":
    unittest.main()
