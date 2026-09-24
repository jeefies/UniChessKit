"""Gumbel 节点级算术与顺序减半的单测：原样沿用 S 的 tests/test_gumbel.py（A 组 §2.8），
只换了 import——kit 与 S 的函数同名同签名，S 的回归测试即是 kit 的回归测试。
改这里时同步 S 的原文件。
"""

from __future__ import annotations

import unittest

import numpy as np

from Kit.search.gumbel import (
    C_SCALE,
    C_VISIT,
    EPS,
    M0,
    N_SIMS,
    Node,
    completed_q,
    export_pi_prime,
    gumbel_topm,
    improved_policy,
    normalize_q,
    order_halving,
    pi_prime,
    policy_probs,
    qtransform_completed,
    select_action,
    sigma,
    softmax,
    v_mix,
)


# ------------------------- 辅助 -------------------------

def _make_node(legal, logits, q=0.0, n=None, q_sum=None, terminal=False):
    legal = np.asarray(legal, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    n = np.asarray(n, dtype=np.int64) if n is not None else np.zeros(len(legal), dtype=np.int64)
    q_sum = np.asarray(q_sum, dtype=np.float32) if q_sum is not None else np.zeros(len(legal), dtype=np.float32)
    return Node(legal=legal, logits=logits, q=float(q), n=n, q_sum=q_sum, terminal=terminal)


def _expand_stub(root, action):
    """固定 1 层深度的 expand stub：所有子节点都是终局，q = −0.3（子视角）。"""
    child_q = -0.3
    return Node(
        legal=np.array([], dtype=np.int64),
        logits=np.array([], dtype=np.float32),
        q=float(child_q),
        terminal=True,
    )


def _expand_depth2(root, action):
    """两层深度的 expand stub：子节点再展开一个终局叶子，子视角 q = +0.6。"""
    child = Node(
        legal=np.array([], dtype=np.int64),
        logits=np.array([], dtype=np.float32),
        q=0.6,
        terminal=True,
    )
    return child


_CHAIN_DEPTH_CUTOFF = 5


def _expand_chain(node, action):
    """单分支变长链 stub：每层恰有 1 个合法着，直到 depth=_CHAIN_DEPTH_CUTOFF 才终局。

    终局 q 由链首动作（node.path[0]，根展开时即 action 本身）决定，构造成奇数深度
    （根视角 = (-1)^depth * 终局 q），用于端到端验证：
    (a) 递归下探真的随同一候选的模拟预算变深（而不是恒定 2 层）；
    (b) 根节点 completedQ 符号正确——链 0 终局 q=-1.0 应让根视角变为 +1.0（更优），
        链 1 终局 q=+1.0 应让根视角变为 -1.0（更差）；旧符号 bug 会使二者颠倒。
    """
    depth = node.depth + 1
    chain_id = node.path[0] if node.path else action
    path = node.path + (action,)
    if depth >= _CHAIN_DEPTH_CUTOFF:
        terminal_q = -1.0 if chain_id == 0 else 1.0
        return Node(legal=np.array([], dtype=np.int64), logits=np.array([], dtype=np.float32),
                    q=terminal_q, terminal=True, depth=depth, path=path)
    return Node(legal=np.array([100 + depth], dtype=np.int64), logits=np.array([0.0], dtype=np.float32),
                q=0.0, depth=depth, path=path)


# ------------------------- A#2 Gumbel 正确性 -------------------------

class GumbelCorrectnessTest(unittest.TestCase):
    """已知真值的 bandit / 小树验证。"""

    def test_v_mix_zero_visits_fallback(self):
        node = _make_node([0, 1], [0.0, 0.0], q=0.5, n=[0, 0])
        self.assertAlmostEqual(v_mix(node), 0.5)

    def test_v_mix_single_visited(self):
        node = _make_node([0, 1], [0.0, 0.0], q=0.5, n=[1, 0], q_sum=[0.8, 0.0])
        vm = v_mix(node)
        self.assertTrue(np.isfinite(vm))

    def test_v_mix_eps_denominator(self):
        node = _make_node([0], [0.0], q=0.0, n=[0])
        vm = v_mix(node)
        self.assertAlmostEqual(vm, 0.0)

    def test_sigma_zero_q(self):
        s = sigma(np.array([0.0], dtype=np.float32), n_max=0)
        self.assertAlmostEqual(float(s.item()), 0.0)

    def test_completed_q_all_visited(self):
        node = _make_node([0, 1], [0.0, 0.0], q=0.0, n=[1, 1], q_sum=[0.5, -0.5])
        cq = completed_q(node)
        self.assertAlmostEqual(float(cq[0]), 0.5)
        self.assertAlmostEqual(float(cq[1]), -0.5)

    def test_order_halving_budget_exact(self):
        root = _make_node([0, 1, 2, 3], [0.1, 0.0, -0.1, -0.2], q=0.0)
        res = order_halving(root, _expand_stub, n_sims=64, m0=4, g=0.0, seed=42)
        self.assertTrue(res["budget_check"], msg=f"budget_check={res['budget_check']}, sims={res['sims_used']}")
        self.assertEqual(res["sims_used"], 64)

    def test_gumbel_topm_g_zero(self):
        node = _make_node([0, 1, 2], [0.5, 0.1, 0.0], q=0.0)
        top = gumbel_topm(node, m0=2, g=0.0, rng=np.random.default_rng(0))
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0], 0)

    def test_pi_prime_sum(self):
        node = _make_node([0, 1], [0.0, 0.0], q=0.0, n=[1, 1], q_sum=[0.3, -0.3])
        pp = pi_prime(node)
        s = float(pp.sum())
        self.assertAlmostEqual(s, 1.0, places=5)


# ------------------------- A#3 目标支持集不变量 -------------------------

class InvariantTest(unittest.TestCase):
    def test_same_completed_q_preserves_pi(self):
        legal = np.array([0, 1, 2], dtype=np.int64)
        logits = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        n = np.array([1, 1, 1], dtype=np.int64)
        q_sum = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        node = Node(legal=legal, logits=logits, q=0.0, n=n, q_sum=q_sum)
        pi = policy_probs(node)
        pp = pi_prime(node)
        np.testing.assert_allclose(pp, pi, atol=1e-5)

    def test_illegal_actions_zero_prob(self):
        legal = np.array([0, 2], dtype=np.int64)
        logits = np.array([0.5, 0.0], dtype=np.float32)
        node = Node(legal=legal, logits=logits, q=0.0)
        full = np.full(5, -3e4, dtype=np.float32)
        full[legal] = logits
        node2 = Node(legal=np.array([0, 1, 2, 3, 4], dtype=np.int64), logits=full, q=0.0)
        pp = pi_prime(node2)
        mask = np.isin(np.arange(5), legal)
        zero_mask = ~mask
        if np.any(zero_mask):
            self.assertTrue(np.all(pp[zero_mask] == 0.0))


# ------------------------- A#4 预算守恒与边界 -------------------------

class BudgetBoundaryTest(unittest.TestCase):
    def test_budget_exact_16(self):
        root = _make_node(list(range(16)), [0.1] * 16, q=0.0)
        res = order_halving(root, _expand_stub, n_sims=64, m0=16, g=0.0, seed=1)
        self.assertTrue(res["budget_check"])
        self.assertEqual(res["sims_used"], 64)

    def test_m_smaller_than_m0(self):
        root = _make_node([0, 1, 2], [0.5, 0.2, 0.1], q=0.0)
        res = order_halving(root, _expand_stub, n_sims=32, m0=16, g=0.0, seed=2)
        self.assertTrue(res["budget_check"])
        self.assertEqual(res["sims_used"], 32)

    def test_single_legal(self):
        root = _make_node([0], [0.0], q=0.0)
        res = order_halving(root, _expand_stub, n_sims=32, m0=16, g=0.0, seed=3)
        self.assertTrue(res["budget_check"])
        self.assertEqual(res["sims_used"], 32)

    def test_non_power_of_two_legal(self):
        root = _make_node([0, 1, 2], [0.5, 0.2, 0.1], q=0.0)
        res = order_halving(root, _expand_stub, n_sims=10, m0=16, g=0.0, seed=4)
        self.assertTrue(res["budget_check"])
        self.assertEqual(res["sims_used"], 10)


# ------------------------- A#2 补充：递归深化 + 端到端符号回归 -------------------------

class RecursiveDepthAndSignTest(unittest.TestCase):
    """回归测试：修复前 do_sim() 固定 2 层深度、非终局分支回传值少取负一次。"""

    def test_depth_grows_with_budget(self):
        root = _make_node([0, 1], [0.0, 0.0], q=0.0)
        res = order_halving(root, _expand_chain, n_sims=64, m0=2, g=0.0, seed=0)
        # 旧实现每次模拟都新建一个从未复用的深度2叶子：64 次模拟 ~= 2(child) + 64(fresh leaf) 个节点。
        # 新实现每条候选链共享持久子树，触达终局后不再新建节点：至多 2 * _CHAIN_DEPTH_CUTOFF 个。
        self.assertLessEqual(res["n_nodes"], 2 * _CHAIN_DEPTH_CUTOFF)
        # 至少真正探到了 2 层以上（否则退化为旧的固定深度）；tree 含 root 本身，故用全体节点的最大 depth。
        deepest = max(node.depth for node in res["tree"])
        self.assertGreaterEqual(deepest, 3)

    def test_root_perspective_sign_correct(self):
        """链 0（终局 q=-1 在奇数深度）应被判定优于链 1（终局 q=+1），根选择应为动作 0。

        旧符号 bug 下非终局分支的回传值方向相反，会使根节点误选动作 1。
        """
        root = _make_node([0, 1], [0.0, 0.0], q=0.0)
        res = order_halving(root, _expand_chain, n_sims=64, m0=2, g=0.0, seed=1)
        self.assertEqual(res["action"], 0)
        cq = completed_q(root)
        self.assertGreater(cq[0], 0.0)
        self.assertLess(cq[1], 0.0)


# ------------------------- A#5 g=0 确定性 -------------------------

class GZeroDeterminismTest(unittest.TestCase):
    def test_g_zero_selects_top_logits(self):
        root = _make_node([0, 1, 2, 3], [1.0, 0.8, 0.5, 0.2], q=0.0)
        res = order_halving(root, _expand_stub, n_sims=64, m0=4, g=0.0, seed=42)
        self.assertEqual(res["action"], 0)

    def test_g_nonzero_varies(self):
        node = _make_node([0, 1], [0.0, 0.0], q=0.0)
        top1 = gumbel_topm(node, m0=2, g=1.0, rng=np.random.default_rng(7))
        top2 = gumbel_topm(node, m0=2, g=1.0, rng=np.random.default_rng(8))
        self.assertNotEqual(top1[0][1], top2[0][1])


# ------------------------- A#6 软 CE 数值安全 -------------------------

class SoftCESafetyTest(unittest.TestCase):
    def test_pi_prime_finite_with_extreme_logits(self):
        legal = np.array([0, 1], dtype=np.int64)
        logits = np.array([-3e4, 0.0], dtype=np.float32)
        node = Node(legal=legal, logits=logits, q=0.0)
        ids, pp = pi_prime(node)
        self.assertTrue(np.all(np.isfinite(pp)))
        self.assertAlmostEqual(float(pp.sum()), 1.0, places=5)

    def test_terminal_node_empty(self):
        node = Node(legal=np.array([], dtype=np.int64), logits=np.array([], dtype=np.float32),
                    q=0.0, terminal=True)
        pp = pi_prime(node)
        self.assertEqual(len(pp), 0)


# ------------------------- A#7 生命周期 -------------------------

class LifecycleTest(unittest.TestCase):
    def test_root_structural_immutable(self):
        root = _make_node([0, 1], [0.1, -0.2], q=0.3)
        legal_before = root.legal.copy()
        logits_before = root.logits.copy()
        q_before = root.q
        terminal_before = root.terminal
        order_halving(root, _expand_stub, n_sims=32, m0=2, g=0.0, seed=9)
        np.testing.assert_array_equal(root.legal, legal_before)
        np.testing.assert_array_equal(root.logits, logits_before)
        self.assertAlmostEqual(root.q, q_before)
        self.assertEqual(root.terminal, terminal_before)


if __name__ == "__main__":
    unittest.main()


class PerNodeQNormalizationTest(unittest.TestCase):
    """回归：σ 的 q̂ 必须用**本节点自身 completed Q 集合**的量程归一（2026-09-19 规格变更）。

    旧实现用整棵树的 qbox（且以父视角 root.q 起算、用子视角 child.q 扩展），
    分母被其他节点撑大后会压缩价值差异在打分中的权重，可能改变选择。
    """

    def _node(self):
        # 两个合法着：policy 强烈偏向 a0（ℓ 差 7），但 Q 偏向 a1（差 0.20）
        node = Node(legal=np.array([0, 1], np.int64),
                    logits=np.array([0.0, -7.0], np.float32), q=0.0)
        node.n = np.array([1, 1], np.int64)
        node.q_sum = np.array([0.05, 0.25], np.float32)
        return node

    def test_uses_own_completed_q_span(self):
        node = self._node()
        cq = completed_q(node)
        np.testing.assert_allclose(cq, [0.05, 0.25], atol=1e-6)

        s = qtransform_completed(node, c_visit=50.0, c_scale=1.0)
        # 本节点量程 = 0.20 → q̂ = [0, 1] → σ 展幅 = (50+1)*1.0 = 51
        self.assertAlmostEqual(float(s[0]), 0.0, places=4)
        self.assertAlmostEqual(float(s[1]), 51.0, places=3)

        # σ 展幅 51 > ℓ 差 7 ⇒ π′ 必须偏向 Q 更高的 a1
        # 本用例的算术钉在 c_scale=1.0（展幅 51）上，必须显式传参：
        # 否则会随模块默认值（现为 0.1，展幅 5.1 < ℓ 差 7）漂移而失败。
        pp = pi_prime(node, c_visit=50.0, c_scale=1.0)
        self.assertGreater(float(pp[1]), float(pp[0]))
        self.assertEqual(int(np.argmax(pp)), 1)

    def test_wide_foreign_span_does_not_leak_in(self):
        """把无关的大量程塞进全树 qbox 不应再影响本节点打分。"""
        node = self._node()
        s_before = qtransform_completed(node)
        # 旧实现下这类"其他节点的极端 q"会把分母撑到 ~2.0，使 σ 差降到 ~6 < ℓ 差 7
        node_far = Node(legal=np.array([0], np.int64),
                        logits=np.array([0.0], np.float32), q=-1.0)
        _ = completed_q(node_far)
        s_after = qtransform_completed(node)
        np.testing.assert_allclose(s_before, s_after, atol=1e-6)

    def test_transform_shared_by_selection_and_target(self):
        """非根选择与 π′ 导出必须来自同一个变换。"""
        node = self._node()
        a = select_action(node)
        ids, pp = export_pi_prime(node)
        pi_imp = improved_policy(node)
        np.testing.assert_allclose(pp, pi_imp, atol=1e-6)
        frac = node.n.astype(np.float32) / np.float32(1 + node.n_total)
        self.assertEqual(a, int(node.legal[np.argmax(pi_imp - frac)]))

    def test_constant_completed_q_gives_flat_sigma(self):
        """所有 completedQ 相同 ⇒ σ 全相等 ⇒ π′ = π（不变量，零量程不得除零）。"""
        node = Node(legal=np.array([0, 1, 2], np.int64),
                    logits=np.array([1.0, 0.0, -2.0], np.float32), q=0.3)
        pp = pi_prime(node)
        np.testing.assert_allclose(pp, softmax(node.logits), atol=1e-5)
        self.assertTrue(np.all(np.isfinite(pp)))


class TestScaleIsPlumbedThrough(unittest.TestCase):
    """回归：export_pi_prime 曾无参数、恒用 C_SCALE=1.0。

    若搜索按 c_scale=0.1 分配访问、选择动作，而训练目标仍按 1.0 写出，
    尺度对照实验就会被这条缺口污染。断言：
        π′_written ≈ softmax(ℓ + qtransform_completed(Q; 配置尺度))
    在**非默认**尺度下同样成立。
    """

    @staticmethod
    def _node():
        node = Node(legal=np.array([0, 1, 2], np.int64),
                    logits=np.array([0.0, 0.1, -0.2], np.float32), q=0.0)
        node.n = np.array([8, 4, 2], np.int64)
        node.q_sum = np.array([0.9 * 8, 0.6 * 4, 0.1 * 2], np.float32)
        return node

    @staticmethod
    def _entropy(p):
        p = np.asarray(p, np.float64)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def test_export_matches_configured_scale(self):
        node = self._node()
        for c_scale in (1.0, 0.1, 0.03):
            ids, probs = export_pi_prime(node, C_VISIT, c_scale)
            expect = softmax(node.logits + qtransform_completed(node, C_VISIT, c_scale))
            np.testing.assert_array_equal(ids, node.legal)
            np.testing.assert_allclose(probs, expect, rtol=1e-5, atol=1e-7)

    def test_different_scales_give_different_targets(self):
        """不同尺度必须写出不同目标，否则说明参数没传到底（旧实现会全部相同）。"""
        node = self._node()
        _, p1 = export_pi_prime(node, C_VISIT, 1.0)
        _, p01 = export_pi_prime(node, C_VISIT, 0.1)
        self.assertFalse(np.allclose(p1, p01, atol=1e-4))
        self.assertGreater(self._entropy(p01), self._entropy(p1))
