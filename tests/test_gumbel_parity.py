"""kit Gumbel 与旧 S ``stateseq.gumbel``（冻结为 tests/fixtures/s_gumbel）的逐位对照，以及着法级 ``Gumbel`` 的行为测试。

1. 节点级：同一棵确定性随机树上，kit 的 ``order_halving`` 选着、访问数、Q 累加、
   π′ 与旧 S 实现的冻结结果逐字节一致（g=0/1、多组预算 / m0 / c_scale）。
2. 着法级：``Gumbel`` 接 Expander 在真实棋盘上搜索，与「S arena 口径的同步 order_halving +
   路径重放 expand」逐位一致——即 S 接入 kit 后搜索结果不变的前提。
3. 行为：一步杀、终局根、只给先验时退化为 log 先验、展开深度直方图。
"""
from __future__ import annotations

import unittest
import zlib

import chess
import numpy as np

from Kit.tests import golden
from Kit.api import EvalRequest
from Kit.api.types import NodeEval
from Kit.runtime import run_sync
from Kit.search import gumbel as kg



# ---------------- 确定性随机世界 ----------------

def _world(path: tuple) -> tuple:
    """path → (合法键, logits, q, 终局)。键故意取非连续值，证明算法不依赖键 = 下标。"""
    rng = np.random.default_rng(zlib.crc32(repr(path).encode()))
    if len(path) >= 2 and (len(path) >= 9 or rng.random() < 0.08):
        return np.zeros(0, np.int64), np.zeros(0, np.float32), float(rng.choice([-1.0, 0.0, 1.0])), True
    n = int(rng.integers(1, 30))
    keys = (1000 + 7 * np.arange(n) + int(rng.integers(0, 5))).astype(np.int64)
    logits = (rng.standard_normal(n) * 2.0).astype(np.float32)
    q = float(np.float32(rng.uniform(-1, 1)))
    return keys, logits, q, False


def _mk(mod, path, depth):
    keys, logits, q, term = _world(path)
    return mod.Node(legal=keys, logits=logits, q=q, depth=depth,
                    action=path[-1] if path else None, path=path, terminal=term)


def _expand_for(mod):
    return lambda node, action: _mk(mod, node.path + (int(action),), node.depth + 1)


CASES = [(g, n_sims, m0, cs, root_seed)
         for g in (0.0, 1.0)
         for n_sims, m0 in ((1, 16), (7, 4), (16, 16), (64, 1), (64, 16), (256, 16))
         for cs in (0.1, 1.0)
         for root_seed in (0, 1, 2)]


def _hexs(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [float.hex(float(x)) for x in v.ravel()]
    return float.hex(float(v))


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return int(v) if isinstance(v, np.integer) else v


class TestNodeLevelParity(unittest.TestCase):
    """与旧 S ``stateseq.gumbel`` 的冻结结果（tests/fixtures/s_gumbel）逐位一致。"""

    def test_order_halving_bitwise(self):
        gold = golden.load("s_gumbel")
        checked = 0
        for g, n_sims, m0, cs, root_seed in CASES:
            with self.subTest(g=g, n_sims=n_sims, m0=m0, c_scale=cs, root=root_seed):
                rk = _mk(kg, ("root", root_seed), 0)
                if rk.is_terminal:
                    continue
                rec = gold[checked]
                checked += 1
                self.assertEqual((rec["g"], rec["n_sims"], rec["m0"], rec["c_scale"], rec["root_seed"]),
                                 (g, n_sims, m0, cs, root_seed))
                b = kg.order_halving(rk, _expand_for(kg), n_sims=n_sims, m0=m0, g=g,
                                     seed=np.random.default_rng(root_seed), c_visit=50.0, c_scale=cs)
                for k in ("action", "sims_used", "rounds", "budget_check", "survivors_per_round",
                          "n_nodes", "n_terminal"):
                    self.assertEqual(_plain(b[k]), rec[k], k)
                for k in ("noise", "qmin", "qmax"):
                    self.assertEqual(_hexs(b[k]), rec[k], k)
                self.assertEqual(np.asarray(rk.n).tolist(), rec["n"])
                self.assertEqual(_hexs(np.asarray(rk.q_sum)), rec["q_sum"])
                ib, pb = kg.export_pi_prime(rk, 50.0, cs)
                self.assertEqual(np.asarray(ib).tolist(), rec["pi_ids"])
                self.assertEqual(_hexs(np.asarray(pb)), rec["pi"])
                self.assertEqual(sum(b["expand_hist"]), b["n_nodes"])
                self.assertEqual(len(b["tree"]), b["n_nodes"] + 1)
        self.assertEqual(checked, len(gold))


# ---------------- 着法级：Expander + 真实棋盘 ----------------

class _BoardEval:
    """局面 → (合法着法, logits, q)，由 FEN 确定性派生。记录评估次数与批大小。"""

    model_key = "fake-gumbel"

    def __init__(self, flat: bool = False):
        self.flat = flat
        self.calls = 0

    def one(self, board: chess.Board):
        moves = list(board.legal_moves)
        rng = np.random.default_rng(zlib.crc32(board.fen().encode()))
        logits = (np.zeros(len(moves)) if self.flat else rng.standard_normal(len(moves)) * 1.5)
        return moves, logits.astype(np.float32), 0.0 if self.flat else float(np.float32(rng.uniform(-0.8, 0.8)))

    def evaluate(self, payloads):
        self.calls += len(payloads)
        return [self.one(b) for b in payloads]


class _Expander:
    def __init__(self, ev: _BoardEval, give_logits: bool = True):
        self.ev = ev
        self.give_logits = give_logits
        self.handles = []

    def expand(self, leaves):
        res = yield EvalRequest(self.ev, [leaf.board for leaf in leaves])
        out = []
        for leaf, (moves, logits, q) in zip(leaves, res):
            self.handles.append(leaf.parent_handle)
            priors = kg.softmax(logits) if len(moves) else np.zeros(0, np.float32)
            out.append(NodeEval(moves=moves, priors=priors, value=q,
                                handle=(leaf.move.uci() if leaf.move else "root"),
                                logits=logits if self.give_logits else None))
        return out


def _s_style_search(mod, board: chess.Board, ev: _BoardEval, *, n_sims, m0, g, seed, c_scale):
    """S arena 口径：动作键 = 着法 id（这里用 uci 的 crc），展开时从根重放 path 再判终局。"""
    def ids_of(b):
        return [zlib.crc32(m.uci().encode()) & 0x7FFFFFFF for m in b.legal_moves]

    def resolve(b, a):
        return next(m for m in b.legal_moves if zlib.crc32(m.uci().encode()) & 0x7FFFFFFF == a)

    def expand(node, action):
        b = board.copy()
        for a in node.path:
            b.push(resolve(b, a))
        b.push(resolve(b, action))
        path = node.path + (action,)
        if b.is_game_over(claim_draw=True):
            return mod.Node(np.zeros(0, np.int64), np.zeros(0, np.float32), kg.terminal_q(b),
                            depth=node.depth + 1, action=action, path=path, terminal=True)
        _, logits, q = ev.one(b)
        return mod.Node(np.array(ids_of(b), np.int64), logits, q, depth=node.depth + 1,
                        action=action, path=path)

    _, logits, q = ev.one(board)
    root = mod.Node(np.array(ids_of(board), np.int64), logits, q)
    res = mod.order_halving(root, expand, n_sims=n_sims, m0=m0, g=g,
                            seed=np.random.default_rng(seed), c_visit=50.0, c_scale=c_scale)
    return resolve(board, res["action"]), root, res


POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",   # Qxf7# 一步杀
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",                                 # Rd8# 底线杀
    "8/8/8/4k3/8/8/4K3/4R3 w - - 98 120",                                    # 五十步边缘
]


class TestMoveLevelGumbel(unittest.TestCase):
    def test_matches_s_style_reference(self):
        for fen in POSITIONS:
            for g, seed in ((0.0, 0), (1.0, 3)):
                with self.subTest(fen=fen, g=g):
                    board = chess.Board(fen)
                    ev = _BoardEval()
                    mv_ref, root_ref, res_ref = _s_style_search(
                        kg, board, ev, n_sims=48, m0=8, g=g, seed=seed, c_scale=0.1)
                    search = kg.Gumbel(_Expander(ev), kg.GumbelConfig(simulations=48, m0=8, g=g))
                    out = run_sync(search.search(board, rng=np.random.default_rng(seed)))
                    self.assertEqual(out.move, mv_ref)
                    self.assertEqual(out.root.n.tobytes(), root_ref.n.tobytes())
                    self.assertEqual(out.root.q_sum.tobytes(), root_ref.q_sum.tobytes())
                    moves, probs = out.pi_prime(search.cfg)
                    _, probs_ref = (kg).export_pi_prime(root_ref, 50.0, 0.1)
                    self.assertEqual(moves, list(board.legal_moves))
                    self.assertEqual(probs.tobytes(), probs_ref.tobytes())
                    for k in ("sims_used", "n_nodes", "n_terminal"):
                        self.assertEqual(out.stats[k], res_ref[k], k)
                    self.assertEqual(out.stats["max_depth"], max(x.depth for x in res_ref["tree"]))

    def test_finds_mate_in_one_and_counts_forwards(self):
        board = chess.Board(POSITIONS[2])
        ev = _BoardEval(flat=True)
        exp = _Expander(ev)
        search = kg.Gumbel(exp, kg.GumbelConfig(simulations=64, m0=64, g=0.0))
        out = run_sync(search.search(board))
        self.assertEqual(out.move, chess.Move.from_uci("d1d8"))
        # 终局子节点在搜索侧判定，不耗前向：前向数 = 根 1 次 + 非终局展开
        self.assertEqual(ev.calls, 1 + out.stats["n_nodes"] - out.stats["n_terminal"])
        self.assertGreaterEqual(out.stats["n_terminal"], 1)
        self.assertEqual(sum(out.stats["expand_hist"]), out.stats["n_nodes"])
        self.assertEqual(out.stats["sims_used"], 64)
        # 句柄链：根的子节点拿到根句柄，更深的节点拿到父着法
        self.assertIn("root", exp.handles)

    def test_root_eval_supplied_skips_root_forward(self):
        board = chess.Board()
        ev = _BoardEval()
        moves, logits, q = ev.one(board)
        root = NodeEval(moves=moves, priors=kg.softmax(logits), value=q, handle="mine", logits=logits)
        search = kg.Gumbel(_Expander(ev), kg.GumbelConfig(simulations=16, m0=4, g=0.0))
        out = run_sync(search.search(board, root=root))
        self.assertEqual(ev.calls, out.stats["n_nodes"] - out.stats["n_terminal"])
        self.assertIn(out.move, board.legal_moves)
        self.assertEqual(out.root.handle, "mine")

    def test_terminal_root(self):
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")    # 黑被将死
        out = run_sync(kg.Gumbel(_Expander(_BoardEval())).search(board))
        self.assertIsNone(out.move)

    def test_priors_only_fallback(self):
        board = chess.Board()
        ev = _BoardEval()
        a = run_sync(kg.Gumbel(_Expander(ev, give_logits=True),
                               kg.GumbelConfig(simulations=32, m0=8, g=0.0)).search(board))
        b = run_sync(kg.Gumbel(_Expander(ev, give_logits=False),
                               kg.GumbelConfig(simulations=32, m0=8, g=0.0)).search(board))
        self.assertIn(b.move, board.legal_moves)
        # log(softmax(ℓ)) 与 ℓ 只差常数，决策应一致（数值上不保证逐位）
        self.assertEqual(a.move, b.move)
        self.assertEqual(a.root.n.tolist(), b.root.n.tolist())


if __name__ == "__main__":
    unittest.main()
