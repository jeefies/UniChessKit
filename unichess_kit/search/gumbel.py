"""Gumbel 顺序减半搜索（Danihelka et al., ICLR 2022），协程版。

移植自 S ``stateseq/gumbel.py``（规格：S ``docs/stage-b-implementation.md`` §2.2【锁定】），
分两层：

1. **节点级算术 + 调度**（``Node`` / ``qtransform_completed`` / ``select_action`` /
   ``gumbel_topm`` / ``order_halving_gen``）：与 S 的函数同名、同签名、同浮点次序，
   S 的 ``tests/test_gumbel.py`` 换个 import 就能原样跑（kit 的 tests/test_gumbel.py 即是），
   与 S 的逐位一致性由 tests/test_gumbel_parity.py 对照。
   ``order_halving_gen`` 把 S 里三份手抄的顺序减半（同步版 ``order_halving``、arena 与
   自对弈的 ``_order_halving_gen``）合成一份：展开回调 ``expand(node, action)`` 是生成器，
   同步调用走 ``order_halving``。
2. **``Gumbel``**：接 kit 的 ``Expander``，动作键 = 边序号（``legal = arange(n)``，着法在
   ``node.moves``），终局在搜索侧判定（``claim_draw`` 口径，不耗前向）。

S 教训（均保留，勿回退）：
- 每层回传恰好取负一次（旧版非终局分支少取负，根 Q 符号系统性反转）；
- σ 的 q̂ 用**本节点自身** completed Q 集合的量程归一（旧版用全树 qbox，视角混用）；
- 根评分 / 非根选择 / π′ 导出共用 ``qtransform_completed``，尺度必须与搜索同一份配置；
- 预算均分、余数前置，总和恰好 = n_sims；唯一候选时剩余预算全部投入。

价值约定：``Node.q`` 为该节点行棋方视角 q = P(win) − P(loss)；父节点上动作价值 = −子 q。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import chess
import numpy as np

from ..api import gather
from ..api.types import Leaf, NodeEval, Think
from ..rules.fast import copy_board
from ..rules.fast import outcome as fast_outcome

# ---- S 锁定超参（§2.2 / §3）----
C_VISIT = 50.0          # σ 的访问数偏置
C_SCALE = 0.1           # σ 的价值缩放（S design-deviations.md §9.3）
EPS = 1e-8              # 分母保护
NEG_LOGIT = -3e4        # 非法动作的有限大负数（S 训练侧软 CE 数值安全）
N_SIMS = 256            # 根节点顺序减半总预算
M0 = 16                 # 根节点候选数上界


# ---------------- 基础数值 ----------------

def softmax(x: np.ndarray) -> np.ndarray:
    """稳定 softmax，fp32 输出，行和 = 1。"""
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x, dtype=np.float32)
    return e / e.sum(dtype=np.float64).astype(np.float32)


def sigma(q_hat, n_max, c_visit: float = C_VISIT, c_scale: float = C_SCALE) -> np.ndarray:
    """σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂。"""
    return (c_visit + np.asarray(n_max, dtype=np.float32)) * c_scale * np.asarray(q_hat, np.float32)


def normalize_q(q, q_min: float, q_max: float) -> np.ndarray:
    """min−max 归一：q̂ = (q − q_min) / (q_max − q_min + eps)。q_min/q_max 必须来自本节点。"""
    span = float(q_max) - float(q_min) + EPS
    return (np.asarray(q, np.float32) - np.float32(q_min)) / np.float32(span)


# ---------------- 节点 ----------------

@dataclass
class Node:
    """搜索树节点。所有数组维度对齐 ``legal``（动作键；``Gumbel`` 里是边序号）。

    terminal=True 时 legal 为空、q 为规则真值（行棋方 −1 负 / 0 和 / +1 胜）。
    moves / line / handle 只有 ``Gumbel`` 用：合法着法、从根到本节点的着法序列、Expander 句柄。
    """

    legal: np.ndarray
    logits: np.ndarray
    q: float
    depth: int = 0
    action: Optional[int] = None
    path: tuple = ()
    terminal: bool = False
    n: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    q_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    root_key: int = 0
    children: dict = field(default_factory=dict)
    moves: Optional[list] = None
    line: tuple = ()
    handle: Any = None

    @property
    def is_terminal(self) -> bool:
        return bool(self.terminal) or self.legal.size == 0

    @property
    def n_total(self) -> int:
        return int(self.n.sum()) if self.n.size else 0

    @property
    def n_max(self) -> int:
        return int(self.n.max()) if self.n.size else 0

    def record_child(self, edge_idx: int, child_value: float) -> None:
        """累加一条边的访问：child_value 已是本节点行棋方视角。"""
        if self.n.size == 0:
            n = len(self.legal)
            self.n = np.zeros(n, np.int64)
            self.q_sum = np.zeros(n, np.float32)
        self.n[edge_idx] += 1
        self.q_sum[edge_idx] += np.float32(child_value)

    def q_edge(self, edge_idx: int) -> Optional[float]:
        if self.n.size == 0 or self.n[edge_idx] == 0:
            return None
        return float(self.q_sum[edge_idx] / self.n[edge_idx])


# ---------------- completed Q 与策略 ----------------

def policy_probs(node: Node) -> np.ndarray:
    if node.terminal or node.logits.size == 0:
        return np.zeros(0, np.float32)
    return softmax(node.logits)


def v_mix(node: Node, c_visit: float = C_VISIT) -> float:
    """v_mix = (v̂ + ΣN · Σ_{N>0} π·q / (Σ_{N>0} π + ε)) / (1 + ΣN)；无访问退化为 v̂。"""
    v_hat = float(node.q)
    if node.n.size == 0 or node.n_total == 0:
        return v_hat
    pi = policy_probs(node)
    visited = np.flatnonzero(node.n > 0)
    num = float(np.dot(pi[visited], node.q_sum[visited] / node.n[visited].astype(np.float32)))
    den = float(pi[visited].sum()) + EPS
    n_tot = float(node.n_total)
    return (v_hat + n_tot * (num / den)) / (1.0 + n_tot)


def completed_q(node: Node) -> np.ndarray:
    """completedQ(a) = q(a) 若 N(a)>0，否则 v_mix。"""
    if node.terminal or node.legal.size == 0:
        return np.zeros(0, np.float32)
    vm = v_mix(node)
    out = np.full(len(node.legal), vm, np.float32)
    if node.n.size:
        visited = np.flatnonzero(node.n > 0)
        out[visited] = (node.q_sum[visited] / node.n[visited].astype(np.float32))
    return out


def qtransform_completed(node: Node, c_visit: float = C_VISIT,
                         c_scale: float = C_SCALE) -> np.ndarray:
    """唯一的 Q→打分变换：σ(q̂)，q̂ 为本节点 completed Q 集合的 min−max 归一。"""
    cq = completed_q(node)
    if cq.size == 0:
        return cq
    q_hat = normalize_q(cq, float(cq.min()), float(cq.max()))
    return sigma(q_hat, node.n_max, c_visit, c_scale)


def improved_policy(node: Node, c_visit: float = C_VISIT,
                    c_scale: float = C_SCALE) -> np.ndarray:
    """π_imp = softmax(ℓ + σ(completedQ))。"""
    if node.terminal or node.logits.size == 0:
        return np.zeros(0, np.float32)
    return softmax(node.logits + qtransform_completed(node, c_visit, c_scale))


def pi_prime(node: Node, c_visit: float = C_VISIT, c_scale: float = C_SCALE) -> np.ndarray:
    """训练目标 π′ = softmax(ℓ + σ(completedQ))，在全部合法着上（与 π_imp 同式）。"""
    return improved_policy(node, c_visit, c_scale)


def select_action(node: Node, c_visit: float = C_VISIT, c_scale: float = C_SCALE) -> int:
    """非根确定性选择：a* = argmax[π_imp − N/(1+ΣN)]。"""
    pi_imp = improved_policy(node, c_visit, c_scale)
    if node.n.size == 0:
        return int(node.legal[np.argmax(pi_imp)])
    frac = node.n.astype(np.float32) / np.float32(1 + node.n_total)
    return int(node.legal[np.argmax(pi_imp - frac)])


def gumbel_topm(node: Node, m0: int = M0, rng: Optional[np.random.Generator] = None,
                g: float = 1.0) -> list:
    """根候选：g·Gumbel(0,1) + ℓ 取 top-m（g=0 关噪声，且不消耗 rng）。→ [(action, noise)]。"""
    if node.terminal or node.logits.size == 0:
        return []
    rng = rng if rng is not None else np.random.default_rng()
    m = min(m0, len(node.legal))
    noise = np.zeros(len(node.legal), np.float32)
    if g != 0.0:
        u = rng.random(len(node.legal), dtype=np.float32)
        noise = -np.log(-np.log(u + 1e-20), dtype=np.float32).astype(np.float32)
        noise = (g * noise)
    score = noise + node.logits.astype(np.float32)
    order = np.argsort(-score, kind="stable")[:m]
    return [(int(node.legal[i]), float(noise[i])) for i in order]


def export_pi_prime(node: Node, c_visit: float = C_VISIT,
                    c_scale: float = C_SCALE) -> tuple:
    """→ (动作键 int64, π′ fp32)，支持集 = 全部合法着。尺度必须与搜索用的同一份配置。"""
    return node.legal.astype(np.int64), pi_prime(node, c_visit, c_scale)


# ---------------- 顺序减半调度 ----------------

@dataclass
class _Candidate:
    action: int
    noise: float
    child: Optional[Node] = None


def _n_rounds(m: int) -> int:
    """⌈log₂ m⌉；m=1 时仍需 1 轮把预算全部投入该唯一候选。"""
    return max(1, math.ceil(math.log2(max(m, 2)))) if m >= 2 else 1


def _hist_add(hist: list, depth: int) -> None:
    if depth >= len(hist):
        hist.extend([0] * (depth + 1 - len(hist)))
    hist[depth] += 1


def order_halving_gen(root: Node, expand, n_sims: int = N_SIMS, m0: int = M0, g: float = 1.0,
                      rng: Optional[np.random.Generator] = None, c_visit: float = C_VISIT,
                      c_scale: float = C_SCALE, qmin: Optional[float] = None,
                      qmax: Optional[float] = None, parallel: bool = False) -> Think[dict]:
    """根节点顺序减半（协程）。``expand(parent, action)`` 为生成器，return 子 Node（不得为 None）。

    parallel=True：同一轮里各候选的第 j 次模拟并发执行（``gather`` 合并成一次请求）。
    候选子树互不相交、根上各记各的边，轮内的根打分只在轮末做，所以结果与串行**逐位相同**
    （前提是 expand 的结果不依赖拼批——S 的前向随批大小有 ~1e-5 差异，因此只在并发 1 下
    与串行逐位一致）；一步搜索的串行拍数从 n_sims 降到 Σ⌈预算/候选数⌉（256/16 时 60）。

    返回 dict：action, noise, sims_used, rounds, budget_check, survivors_per_round, qmin, qmax,
    n_nodes, n_terminal, max_depth, expand_hist（按新节点深度计数）, tree。
    qmin/qmax 仅作诊断（全树见过的 q 范围），不参与打分。
    """
    if root.terminal or root.legal.size == 0:
        return {"action": None, "noise": 0.0, "sims_used": 0, "rounds": 0,
                "budget_check": True, "survivors_per_round": [], "qmin": None, "qmax": None,
                "n_nodes": 0, "n_terminal": 0, "max_depth": 0, "expand_hist": [], "tree": None}
    rng = rng if rng is not None else np.random.default_rng(0)
    cands = gumbel_topm(root, m0=m0, rng=rng, g=g)
    rounds = _n_rounds(len(cands))
    surv = [_Candidate(action=a, noise=ns) for a, ns in cands]
    base, rem = divmod(n_sims, rounds)
    budget_per_round = [base + (1 if i < rem else 0) for i in range(rounds)]

    st = {"qmin": float(root.q if qmin is None or qmax is None else qmin),
          "qmax": float(root.q if qmin is None or qmax is None else qmax),
          "n_nodes": 0, "n_terminal": 0, "max_depth": 0}
    hist: list = []
    tree: list = [root]

    def _expand(node: Node, action: int) -> Think[Node]:
        _hist_add(hist, node.depth + 1)
        child = yield from expand(node, action)
        if child is None:
            raise ValueError(f"expand 返回 None：动作 {action} 无法展开")
        st["n_nodes"] += 1
        tree.append(child)
        if child.is_terminal:
            st["n_terminal"] += 1
        if child.depth > st["max_depth"]:
            st["max_depth"] = child.depth
        if child.q < st["qmin"]:
            st["qmin"] = child.q
        if child.q > st["qmax"]:
            st["qmax"] = child.q
        return child

    def _simulate(node: Node) -> Think[float]:
        """一次下探：返回该节点自身行棋方视角的价值（每层恰好取负一次）。"""
        if node.is_terminal:
            return float(node.q)
        a = select_action(node, c_visit, c_scale)
        edge_idx = int(np.flatnonzero(node.legal == a)[0])
        child = node.children.get(int(a))
        if child is None:
            child = yield from _expand(node, a)
            node.children[int(a)] = child
            val = -float(child.q)
        else:
            val = -(yield from _simulate(child))
        node.record_child(edge_idx, val)
        return val

    def _sim_root(c: _Candidate) -> Think[None]:
        if c.child is None:
            c.child = yield from _expand(root, c.action)
            val = -float(c.child.q)
        elif c.child.is_terminal:
            val = -float(c.child.q)
        else:
            val = -(yield from _simulate(c.child))
        root.record_child(int(np.flatnonzero(root.legal == c.action)[0]), val)

    survivors_log: list = []
    sims_used = 0
    for r, budget in enumerate(budget_per_round):
        if len(surv) == 1:
            budget = sum(budget_per_round[r:])       # 唯一候选：剩余预算全部投入（预算守恒）
        per_base, per_rem = divmod(budget, len(surv))
        ks = [per_base + (1 if i < per_rem else 0) for i in range(len(surv))]
        if parallel:
            for j in range(max(ks)):
                batch = [c for c, k in zip(surv, ks) if j < k]
                if len(batch) == 1:
                    yield from _sim_root(batch[0])
                else:
                    yield from gather([_sim_root(c) for c in batch])
        else:
            for c, k in zip(surv, ks):
                for _ in range(k):
                    yield from _sim_root(c)
        sims_used += sum(ks)
        l_root = {int(a): float(x) for a, x in zip(root.legal, root.logits)}
        s_root = qtransform_completed(root, c_visit, c_scale)
        s_map = {int(a): float(x) for a, x in zip(root.legal, s_root)}
        scored = sorted(((c.noise + l_root[c.action] + s_map[c.action], c) for c in surv),
                        key=lambda t: -t[0])
        surv = [c for _, c in scored[:max(1, (len(surv) + 1) // 2)]]
        survivors_log.append(len(surv))

    return {"action": int(surv[0].action), "noise": float(surv[0].noise),
            "sims_used": int(sims_used), "rounds": rounds, "budget_check": sims_used == n_sims,
            "survivors_per_round": survivors_log, "qmin": st["qmin"], "qmax": st["qmax"],
            "n_nodes": st["n_nodes"], "n_terminal": st["n_terminal"],
            "max_depth": st["max_depth"], "expand_hist": hist, "tree": tree}


def order_halving(root: Node, expand, n_sims: int = N_SIMS, m0: int = M0, g: float = 1.0,
                  seed=0, qmin: Optional[float] = None, qmax: Optional[float] = None,
                  c_visit: float = C_VISIT, c_scale: float = C_SCALE) -> dict:
    """同步版（S ``stateseq.gumbel.order_halving`` 的签名）：``expand(node, action) -> Node``。"""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def expand_gen(node, action):
        return expand(node, action)
        yield  # noqa: unreachable —— 让它成为不 yield 的生成器

    gen = order_halving_gen(root, expand_gen, n_sims=n_sims, m0=m0, g=g, rng=rng,
                            c_visit=c_visit, c_scale=c_scale, qmin=qmin, qmax=qmax)
    try:
        gen.send(None)
    except StopIteration as stop:
        return stop.value
    raise RuntimeError("同步 order_halving 的 expand 不应产生评估请求")


# ---------------- 接 kit Expander 的着法级搜索 ----------------

@dataclass
class GumbelConfig:
    simulations: int = N_SIMS
    m0: int = M0
    g: float = 1.0                  # 根 Gumbel 噪声尺度；评测 / arena 用 0（确定性）
    c_visit: float = C_VISIT
    c_scale: float = C_SCALE
    claim_draw: bool = True         # 搜索内把可申和（三次重复 / 五十步）当终局（S 口径）
    parallel: bool = True           # 轮内各候选并发模拟（见 order_halving_gen；False = 原串行次序）


@dataclass
class GumbelResult:
    move: Optional[chess.Move]
    root: Optional[Node]
    stats: dict

    def pi_prime(self, cfg: GumbelConfig) -> tuple:
        """→ (着法列表, π′ fp32)，按根的合法着顺序。"""
        if self.root is None or self.root.is_terminal:
            return [], np.zeros(0, np.float32)
        _, probs = export_pi_prime(self.root, cfg.c_visit, cfg.c_scale)
        return list(self.root.moves), probs


def terminal_q(board: chess.Board, claim_draw: bool = True) -> float:
    """终局真值（行棋方视角）：被将死 −1，和棋 0。"""
    return _outcome_q(board, fast_outcome(board, claim_draw))


def _outcome_q(board: chess.Board, outcome) -> float:
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if board.turn == outcome.winner else -1.0


def node_from_eval(ev: NodeEval, *, depth: int = 0, action: Optional[int] = None,
                   path: tuple = (), line: tuple = ()) -> Node:
    """NodeEval → Node。优先用 ev.logits（S 要逐位一致）；只有先验时取 log。"""
    moves = list(ev.moves)
    n = len(moves)
    if ev.logits is not None:
        logits = np.asarray(ev.logits, dtype=np.float32)
    else:
        with np.errstate(divide="ignore"):
            logits = np.log(np.asarray(ev.priors, dtype=np.float32)).astype(np.float32)
        logits = np.maximum(logits, np.float32(NEG_LOGIT))
    if logits.shape != (n,):
        raise ValueError(f"logits 形状 {logits.shape} 与合法着数 {n} 不符")
    return Node(legal=np.arange(n, dtype=np.int64), logits=logits, q=float(ev.value),
                depth=depth, action=action, path=path, terminal=n == 0,
                moves=moves, line=line, handle=ev.handle)


class Gumbel:
    """着法级 Gumbel 搜索：一次模拟展开一个新节点（顺序减半天然串行，每拍 1 个叶子）。

    ``search(board, root=...)``：root 为已评估好的根（S 的根评估来自对局推进时那次前向），
    缺省则先经 Expander 评估根。不支持 deadline：顺序减半的预算必须事先定死。
    """

    def __init__(self, expander, cfg: Optional[GumbelConfig] = None):
        self.expander = expander
        self.cfg = cfg or GumbelConfig()

    def search(self, board: chess.Board, *, root: Optional[NodeEval] = None,
               rng: Optional[np.random.Generator] = None, simulations: Optional[int] = None,
               g: Optional[float] = None) -> Think[GumbelResult]:
        cfg = self.cfg
        if fast_outcome(board, cfg.claim_draw) is not None:
            return GumbelResult(None, None, {"sims_used": 0})
        if root is None:
            (root,) = yield from self.expander.expand([Leaf(board=board)])
        root_node = node_from_eval(root)
        if root_node.is_terminal:
            return GumbelResult(None, root_node, {"sims_used": 0})
        root_board = copy_board(board)

        def expand(parent: Node, action: int) -> Think[Node]:
            mv = parent.moves[action]
            line = parent.line + (mv,)
            depth, path = parent.depth + 1, parent.path + (action,)
            child_board = copy_board(root_board)
            for m in line:
                child_board.push(m)
            outcome = fast_outcome(child_board, cfg.claim_draw)
            if outcome is not None:
                return Node(legal=np.zeros(0, np.int64), logits=np.zeros(0, np.float32),
                            q=_outcome_q(child_board, outcome), depth=depth,
                            action=action, path=path, terminal=True, moves=[], line=line)
            (ev,) = yield from self.expander.expand(
                [Leaf(board=child_board, parent_handle=parent.handle, move=mv)])
            return node_from_eval(ev, depth=depth, action=action, path=path, line=line)

        res = yield from order_halving_gen(
            root_node, expand, n_sims=simulations or cfg.simulations, m0=cfg.m0,
            g=cfg.g if g is None else g, rng=rng, c_visit=cfg.c_visit, c_scale=cfg.c_scale,
            parallel=cfg.parallel)
        res.pop("tree", None)
        return GumbelResult(root_node.moves[res["action"]], root_node, res)
