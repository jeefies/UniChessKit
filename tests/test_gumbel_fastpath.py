"""select_action 快路径与参考写法（逐函数组合）逐位一致：随机节点，含未访问、全访问、
并列 logits / q、极端 q、float64 logits、同一节点多次选择（π 缓存复用）。"""

from __future__ import annotations

import unittest

import numpy as np

from Kit.search.gumbel import Node, _select_action_ref, select_action


def _rand_node(rng: np.random.Generator) -> Node:
    k = int(rng.integers(1, 40))
    kind = int(rng.integers(0, 4))
    if kind == 0:          # 并列 logits（量化到少数几个值）
        logits = rng.integers(-2, 3, k).astype(np.float32)
    elif kind == 1:        # 大幅度 logits（含 -3e4 这类非法着占位）
        logits = np.where(rng.random(k) < 0.2, np.float32(-3e4),
                          rng.normal(0, 8, k)).astype(np.float32)
    else:
        logits = rng.normal(0, 2, k).astype(np.float32)
    if rng.random() < 0.1:
        logits = logits.astype(np.float64)
    node = Node(legal=rng.permutation(1936)[:k].astype(np.int64), logits=logits,
                q=float(rng.uniform(-1, 1)))
    mode = int(rng.integers(0, 4))
    if mode == 0:
        return node        # 未分配 n
    n = np.zeros(k, np.int64)
    if mode >= 2:
        mask = rng.random(k) < (0.3 if mode == 2 else 1.0)
        n[mask] = rng.integers(1, 60, int(mask.sum()))
    q_vals = rng.choice(np.array([-1.0, 0.0, 1.0, 0.5], np.float32), k) if rng.random() < 0.3 \
        else rng.uniform(-1, 1, k).astype(np.float32)
    node.n = n
    node.q_sum = (q_vals * n).astype(np.float32)
    return node


class TestSelectActionFastPath(unittest.TestCase):
    def test_bitwise_equal_to_reference(self):
        rng = np.random.default_rng(20260924)
        for i in range(4000):
            node = _rand_node(rng)
            for cv, cs in ((50.0, 0.1), (50.0, 1.0), (10.0, 0.3)):
                self.assertEqual(select_action(node, cv, cs), _select_action_ref(node, cv, cs),
                                 f"case {i} c_visit={cv} c_scale={cs}")

    def test_cache_reused_across_visits(self):
        """同一节点边选边记访问（搜索里的真实用法），每步与参考一致。"""
        rng = np.random.default_rng(7)
        for _ in range(50):
            node = _rand_node(rng)
            node.n = np.zeros(0, np.int64)
            node.q_sum = np.zeros(0, np.float32)
            for _step in range(80):
                a = select_action(node)
                self.assertEqual(a, _select_action_ref(node))
                e = int(np.flatnonzero(node.legal == a)[0])
                node.record_child(e, float(rng.uniform(-1, 1)))
            self.assertIsNotNone(node.pi_cache)


if __name__ == "__main__":
    unittest.main()
