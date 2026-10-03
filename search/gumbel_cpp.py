"""C++ Gumbel 顺序减半搜索（``_native/puct_native.cpp`` 的 ``kg_*`` 段）。

与 ``gumbel.Gumbel`` 数值等价（非逐位，契约见 cpp 顶部）：同样经 ``EvalRequest`` 前向，
但调度在 C++ 里按拍（wave）批量 —— 一轮里每个存活候选各下探一次 = 一拍，一拍的叶子
一次前向（候选子树互不相交）。

与 Python 版的差异（有意为之）：
- 不支持外部预评估根（``root=``）与 deadline：顺序减半的预算必须事先定死；
- 搜索树在 C++ 里，返回的 ``GumbelResult.root`` 是**重建**的 Python ``Node``（供
  ``pi_prime()`` 与测试读根统计），不含整棵树。
"""
from __future__ import annotations

import ctypes
from typing import Optional

import chess
import numpy as np

from ..api.types import EvalRequest, Think
from ..rules.fast import outcome as fast_outcome
from . import native
from .gumbel import GumbelConfig, GumbelResult, Node
from .puct_cpp import code_move, move_code

_PLANES_SHAPE = (19, 8, 8)


class GumbelCpp:
    """planes_evaluator: 负载为 (19,8,8) float32 编码的 BatchEvaluator。"""

    def __init__(self, planes_evaluator, cfg: Optional[GumbelConfig] = None):
        if int(np.__version__.split(".")[0]) < 2:
            raise RuntimeError("C++ Gumbel 按 numpy 2（NEP 50）的类型提升对齐 Python，需要 numpy>=2")
        self.cfg = cfg or GumbelConfig()
        self.evaluator = planes_evaluator
        self._lib = native.lib()
        c = self.cfg
        self._ctx = self._lib.kg_ctx_new(float(c.c_visit), float(c.c_scale), int(c.m0),
                                         int(bool(c.claim_draw)))
        if not self._ctx:
            raise RuntimeError(f"C++ Gumbel 初始化失败：{native.last_error()}")
        self._cap = max(1, int(c.m0))
        self._planes = np.empty((self._cap,) + _PLANES_SHAPE, dtype=np.float32)
        self._info = (ctypes.c_int * 6)()
        self.last_stats: dict = {}

    def __del__(self):
        ctx, self._ctx = getattr(self, "_ctx", None), None
        if ctx:
            self._lib.kg_ctx_free(ctx)

    # ---------- 与 C++ 的交接 ----------

    def _load_root(self, board: chess.Board) -> None:
        r = board.root()
        bbs = (ctypes.c_uint64 * 8)(r.pawns, r.knights, r.bishops, r.rooks, r.queens, r.kings,
                                    r.occupied_co[chess.WHITE], r.occupied_co[chess.BLACK])
        codes = np.fromiter((move_code(m) for m in board.move_stack), dtype=np.uint16,
                            count=len(board.move_stack))
        native.check(self._lib.kg_set_root(self._ctx, bbs, int(r.turn),
                                           r.clean_castling_rights(),
                                           -1 if r.ep_square is None else r.ep_square,
                                           r.halfmove_clock,
                                           codes.ctypes.data_as(native._u16p), len(codes)),
                     "载入根局面")

    def _evaluate(self, planes: np.ndarray, n: int) -> Think[tuple]:
        outs = yield EvalRequest(self.evaluator, tuple(planes[:n]))
        if len(outs) != n:
            raise ValueError(f"评估器返回 {len(outs)} 个结果，应为 {n} 个")
        policy = np.ascontiguousarray(np.stack([o[0] for o in outs]), dtype=np.float32)
        promo = np.ascontiguousarray(np.stack([o[1] for o in outs]), dtype=np.float32)
        wdl = np.ascontiguousarray(np.stack([o[2] for o in outs]), dtype=np.float32)
        if policy.shape != (n, 4096) or promo.shape != (n, 4) or wdl.shape != (n, 3):
            raise ValueError(f"评估器输出形状不对：{policy.shape} {promo.shape} {wdl.shape}")
        return policy, promo, wdl

    def _root_node(self) -> Node:
        n_max = 256
        moves = np.zeros(n_max, dtype=np.uint16)
        logits = np.zeros(n_max, dtype=np.float32)
        N = np.zeros(n_max, dtype=np.int64)
        qsum = np.zeros(n_max, dtype=np.float32)
        q = ctypes.c_double()
        n = native.check(self._lib.kg_root_export(
            self._ctx, moves.ctypes.data_as(native._u16p), logits.ctypes.data_as(native._f32p),
            N.ctypes.data_as(native._i64p), qsum.ctypes.data_as(native._f32p), ctypes.byref(q)),
            "导出根")
        return Node(legal=np.arange(n, dtype=np.int64), logits=logits[:n].copy(), q=float(q.value),
                    n=N[:n].copy(), q_sum=qsum[:n].copy(),
                    moves=[code_move(int(c)) for c in moves[:n]], terminal=False)

    # ---------- 对外 ----------

    def search(self, board: chess.Board, *, root: Optional[Node] = None,
               rng: Optional[np.random.Generator] = None,
               simulations: Optional[int] = None,
               g: Optional[float] = None) -> Think[GumbelResult]:
        if root is not None:
            raise ValueError("GumbelCpp 不支持外部预评估根")
        cfg = self.cfg
        sims = int(simulations or cfg.simulations)
        gv = cfg.g if g is None else float(g)
        L = self._lib
        if fast_outcome(board, cfg.claim_draw) is not None:
            self.last_stats = {"sims_used": 0}
            return GumbelResult(None, None, dict(self.last_stats))
        self._load_root(board)
        planes = np.empty((1,) + _PLANES_SHAPE, dtype=np.float32)
        native.check(L.kg_encode_root(self._ctx, planes.ctypes.data_as(native._f32p)),
                     "编码根局面")
        policy, promo, wdl = yield from self._evaluate(planes, 1)
        n_moves = native.check(L.kg_expand_root(self._ctx,
                                                policy.ctypes.data_as(native._f32p),
                                                promo.ctypes.data_as(native._f32p),
                                                wdl.ctypes.data_as(native._f32p)),
                               "展开根节点")
        if n_moves == 0:
            self.last_stats = {"sims_used": 0}
            return GumbelResult(None, self._root_node(), dict(self.last_stats))
        # 噪声：复刻 gumbel_topm（rng.random(fp32) → -log(-log(u+1e-20)) → *g）
        if rng is None:
            rng = np.random.default_rng()
        noise = np.zeros(n_moves, dtype=np.float32)
        if gv != 0.0:
            u = rng.random(n_moves, dtype=np.float32)
            noise = -np.log(-np.log(u + 1e-20), dtype=np.float32).astype(np.float32)
            noise = (gv * noise)
        rounds = native.check(L.kg_begin(self._ctx, noise.ctypes.data_as(native._f32p), sims),
                              "开始搜索")
        while True:
            n = native.check(L.kg_collect(self._ctx, self._cap,
                                          self._planes.ctypes.data_as(native._f32p),
                                          ctypes.cast(self._info, native._intp)), "收集叶子")
            if n == 0:
                break
            policy, promo, wdl = yield from self._evaluate(self._planes, n)
            native.check(L.kg_apply(self._ctx, policy.ctypes.data_as(native._f32p),
                                    promo.ctypes.data_as(native._f32p),
                                    wdl.ctypes.data_as(native._f32p), n), "回填叶子")
        info = self._info
        self.last_stats = {"sims_used": int(info[3]), "rounds": int(info[4]),
                           "n_nodes": int(info[0]), "n_terminal": int(info[1]),
                           "max_depth": int(info[2])}
        node = self._root_node()
        action = int(info[5])
        if action < 0:
            return GumbelResult(None, node, dict(self.last_stats))
        mv = list(board.legal_moves)[action]
        return GumbelResult(mv, node, dict(self.last_stats))