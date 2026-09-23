import unittest

from unichess_kit.api import EvalRequest, immediate
from unichess_kit.runtime import Batcher, CoroutinePool, WorkerError, WorkerPool, run_sync
from unichess_kit.testing import fakes


class Echo:
    """evaluate(x) = x * factor，记录每次调用的负载。"""

    def __init__(self, key, factor=1):
        self.model_key = key
        self.factor = factor
        self.calls = []

    def evaluate(self, payloads):
        self.calls.append(list(payloads))
        return [p * self.factor for p in payloads]


def asker(ev, values):
    """每拍 yield 一个请求，返回所有结果。"""
    got = []
    for v in values:
        out = yield EvalRequest(ev, v)
        got.append(list(out))
    return got


class TestEvalRequest(unittest.TestCase):
    def test_rejects_empty_and_keyless(self):
        with self.assertRaises(ValueError):
            EvalRequest(Echo("a"), [])
        with self.assertRaises(ValueError):
            EvalRequest(Echo(""), [1])

    def test_payloads_become_tuple(self):
        self.assertEqual(EvalRequest(Echo("a"), [1, 2]).payloads, (1, 2))


class TestBatcher(unittest.TestCase):
    def test_groups_by_model_one_forward_each(self):
        a, b = Echo("a", 10), Echo("b", 100)
        batcher = Batcher()
        out = batcher.evaluate([EvalRequest(a, [1, 2]), EvalRequest(b, [3]),
                                EvalRequest(a, [4])])
        self.assertEqual(out, [[10, 20], [300], [40]])
        self.assertEqual(a.calls, [[1, 2, 4]])        # 一个模型只前向一次，按请求顺序拼接
        self.assertEqual(b.calls, [[3]])
        self.assertEqual(batcher.stats.forwards, 2)
        self.assertEqual(batcher.stats.positions, 4)

    def test_same_key_different_objects_rejected(self):
        with self.assertRaises(ValueError):
            Batcher().evaluate([EvalRequest(Echo("a"), [1]), EvalRequest(Echo("a"), [2])])

    def test_wrong_output_length_rejected(self):
        class Bad(Echo):
            def evaluate(self, payloads):
                return [0]
        with self.assertRaises(ValueError):
            Batcher().evaluate([EvalRequest(Bad("x"), [1, 2])])

    def test_non_request_rejected(self):
        def bad():
            yield "not a request"
        with self.assertRaises(TypeError):
            run_sync(bad())

    def test_run_sync(self):
        self.assertEqual(run_sync(asker(Echo("a", 2), [[1], [2, 3]])), [[2], [4, 6]])
        self.assertEqual(run_sync(immediate(5)), 5)


class TestCoroutinePool(unittest.TestCase):
    def test_all_jobs_complete_with_refill(self):
        ev = Echo("a")
        jobs = [(i, (lambda i=i: asker(ev, [[i]] * (i % 3 + 1)))) for i in range(7)]
        done = {}
        CoroutinePool(3).run(jobs, lambda j, r: done.__setitem__(j, r))
        self.assertEqual(sorted(done), list(range(7)))
        for i, r in done.items():
            self.assertEqual(r, [[i]] * (i % 3 + 1))
        self.assertTrue(all(len(c) <= 3 for c in ev.calls))   # 每拍至多 3 个并发任务

    def test_cross_job_batching(self):
        ev = Echo("a")
        jobs = [(i, (lambda i=i: asker(ev, [[i, i]]))) for i in range(4)]
        CoroutinePool(4).run(jobs, lambda j, r: None)
        self.assertEqual(ev.calls, [[0, 0, 1, 1, 2, 2, 3, 3]])

    def test_immediate_jobs(self):
        done = []
        CoroutinePool(2).run([(i, (lambda i=i: immediate(i))) for i in range(5)],
                             lambda j, r: done.append(r))
        self.assertEqual(done, [0, 1, 2, 3, 4])

    def test_should_stop_lets_running_jobs_finish(self):
        ev = Echo("a")
        done = []
        jobs = [(i, (lambda i=i: asker(ev, [[i]] * 3))) for i in range(10)]
        CoroutinePool(2).run(jobs, lambda j, r: done.append(j), should_stop=lambda: len(done) >= 1)
        self.assertEqual(sorted(done), [0, 1])     # 两个已启动的任务都完成，之后不再开新任务

    def test_exception_propagates_and_closes_others(self):
        ev = Echo("a")
        closed = []

        def good():
            try:
                while True:
                    yield EvalRequest(ev, [1])
            finally:
                closed.append(True)

        def bad():
            yield EvalRequest(ev, [1])
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            CoroutinePool(2).run([(0, good), (1, bad)], lambda j, r: None)
        self.assertEqual(closed, [True])

    def test_deterministic(self):
        def run():
            ev = Echo("a")
            order = []
            jobs = [(i, (lambda i=i: asker(ev, [[i]] * (7 * i % 5 + 1)))) for i in range(12)]
            CoroutinePool(4).run(jobs, lambda j, r: order.append(j))
            return order, ev.calls
        self.assertEqual(run(), run())


class TestWorkerPool(unittest.TestCase):
    def test_results_and_done(self):
        got = []
        WorkerPool().run(fakes.ok_worker, [[1, 2], [3]], lambda w, x: got.append(x))
        self.assertEqual(sorted(got), [1, 2, 3])

    def test_error_reported(self):
        with self.assertRaisesRegex(WorkerError, "故意失败"):
            WorkerPool().run(fakes.error_worker, [None], lambda w, x: None)

    def test_silent_exit_is_error(self):
        got = []
        with self.assertRaisesRegex(WorkerError, "静默退出"):
            WorkerPool(poll_s=0.2).run(fakes.crash_worker, [None], lambda w, x: got.append(x))


if __name__ == "__main__":
    unittest.main()
