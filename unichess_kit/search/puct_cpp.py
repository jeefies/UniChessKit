"""PUCT 的 C++ 版：树、下探、棋规、编码都在 C++（``_native/puct_native.cpp``），Python 只做调度。

与 ``PUCT`` 逐位一致（同一评估器下整棵树的 N / W / VL / P 完全相同，tests/test_puct_cpp.py 对照），
对外接口也相同（search / best_move / advance_root / principal_variation / last_metrics），
SearchPlayer 可以直接替换。差别只在模型接口：

- 叶子不再构造 chess.Board，而是由 C++ 直接写出 19×8×8 编码；评估器的负载是
  ``np.ndarray (19, 8, 8) float32``，结果仍是 (policy[4096], promo[4], wdl[3])。
  T/R 的 ``evaluate_planes`` 就是 ``evaluate_batch`` 去掉编码那一步。
- 残局表仍由 Python 的 oracle 探测：C++ 遇到子力 <= oracle.max_pieces 的叶子会暂停，
  把路径交回 Python，在原棋盘（含走子栈）上重放后探测，再回填。

步进式 API（collect / apply）调用 C++ 期间释放 GIL。
"""
from __future__ import annotations

import ctypes
import time
from typing import Optional

import chess
import numpy as np

from ..api.types import EvalRequest, Think
from . import native
from .puct import DEADLINE_PROBE_SIMS, PUCT, PUCTConfig

_PLANES_SHAPE = (19, 8, 8)
_MAX_PATH = 1024


def move_code(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


_MOVE_CACHE: dict = {}


def code_move(code: int) -> chess.Move:
    mv = _MOVE_CACHE.get(code)
    if mv is None:
        promo = (code >> 12) & 7
        mv = chess.Move(code & 63, (code >> 6) & 63, promotion=promo or None)
        _MOVE_CACHE[code] = mv
    return mv


def _ptr(arr: np.ndarray, typ):
    return arr.ctypes.data_as(typ)


class CppNode:
    """C++ 树节点的句柄（持有 shared_ptr：句柄活着，子树就活着）。属性与 puct.Node 同名同型，均为快照。"""

    __slots__ = ("_h", "_lib")

    def __init__(self, handle):
        if not handle:
            raise RuntimeError(f"C++ PUCT 节点句柄为空：{native.last_error()}")
        self._lib = native.lib()
        self._h = handle

    @classmethod
    def new(cls) -> "CppNode":
        return cls(native.lib().kp_node_new())

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            self._lib.kp_node_free(h)

    def _state(self):
        e, t = ctypes.c_int(), ctypes.c_int()
        v, s = ctypes.c_double(), ctypes.c_int64()
        n = self._lib.kp_node_state(self._h, ctypes.byref(e), ctypes.byref(t), ctypes.byref(v),
                                    ctypes.byref(s))
        return n, bool(e.value), (v.value if t.value else None), int(s.value)

    _ARRAYS = {"moves": (np.uint16, native._u16p), "P": (np.float32, native._f32p),
               "N": (np.int32, native._i32p), "W": (np.float32, native._f32p),
               "VL": (np.float32, native._f32p)}

    def _array(self, which: str) -> np.ndarray:
        dtype, ptype = self._ARRAYS[which]
        out = np.zeros(self._state()[0], dtype=dtype)
        args = {k: None for k in self._ARRAYS}
        args[which] = _ptr(out, ptype)
        self._lib.kp_node_arrays(self._h, args["moves"], args["P"], args["N"], args["W"],
                                 args["VL"])
        return out

    @property
    def expanded(self) -> bool:
        return self._state()[1]

    @property
    def terminal_value(self) -> Optional[float]:
        return self._state()[2]

    @property
    def sum_N(self) -> int:
        return self._state()[3]

    @property
    def moves(self) -> list:
        return [code_move(int(c)) for c in self._array("moves")]

    @property
    def P(self) -> np.ndarray:
        return self._array("P")

    @property
    def N(self) -> np.ndarray:
        return self._array("N")

    @property
    def W(self) -> np.ndarray:
        return self._array("W")

    @property
    def VL(self) -> np.ndarray:
        return self._array("VL")

    @property
    def children(self) -> list:
        n = self._state()[0]
        out = []
        for i in range(n):
            h = self._lib.kp_node_child(self._h, i)
            out.append(CppNode(h) if h else None)
        return out


class PUCTCpp(PUCT):
    """planes_evaluator: 负载为 (19,8,8) float32 编码的 BatchEvaluator；其余参数同 PUCT。

    expander 只在 ``SearchPlayer`` 的网络直出兜底里用到，这里不需要。
    """

    def __init__(self, planes_evaluator, cfg: Optional[PUCTConfig] = None, oracle=None,
                 rng: Optional[np.random.Generator] = None, expander=None):
        if int(np.__version__.split(".")[0]) < 2:
            raise RuntimeError("C++ PUCT 按 numpy 2（NEP 50）的类型提升逐位对齐 Python PUCT，需要 numpy>=2")
        super().__init__(expander, cfg, oracle, rng)
        self.evaluator = planes_evaluator
        self._lib = native.lib()
        c = self.cfg
        max_pieces = -1 if oracle is None else int(getattr(oracle, "max_pieces", 64))
        self._ctx = self._lib.kp_ctx_new(float(c.c_puct_base), float(c.c_puct_init),
                                         float(c.fpu_reduction), float(c.virtual_loss),
                                         int(bool(c.claim_draw)), int(c.max_collision),
                                         int(c.root_min_visits), max_pieces)
        if not self._ctx:
            raise RuntimeError(f"C++ PUCT 初始化失败：{native.last_error()}")
        self._info = (ctypes.c_int * 4)()
        self._path = np.zeros(_MAX_PATH, dtype=np.uint16)

    def __del__(self):
        ctx, self._ctx = getattr(self, "_ctx", None), None
        if ctx:
            self._lib.kp_ctx_free(ctx)

    def _without_oracle(self) -> "PUCTCpp":
        return PUCTCpp(self.evaluator, self.cfg, oracle=None, rng=self.rng, expander=self.expander)

    # ---------- 与 C++ 的交接 ----------

    def _load_root(self, board: chess.Board) -> None:
        r = board.root()
        bbs = (ctypes.c_uint64 * 8)(r.pawns, r.knights, r.bishops, r.rooks, r.queens, r.kings,
                                    r.occupied_co[chess.WHITE], r.occupied_co[chess.BLACK])
        codes = np.fromiter((move_code(m) for m in board.move_stack), dtype=np.uint16,
                            count=len(board.move_stack))
        rc = self._lib.kp_set_root(self._ctx, bbs, int(r.turn), r.clean_castling_rights(),
                                   -1 if r.ep_square is None else r.ep_square, r.halfmove_clock,
                                   _ptr(codes, native._u16p), len(codes))
        native.check(rc, "载入根局面")
        # 核对 C++ 重放出的局面：接错局面不会报错，只会静默下另一盘棋
        buf = ctypes.create_string_buffer(128)
        native.check(self._lib.kp_root_fen(self._ctx, buf, 128), "读取根局面")
        want = board.fen(en_passant="fen").rsplit(" ", 1)[0]
        got = buf.value.decode().rsplit(" ", 1)[0]
        if got != want:
            raise RuntimeError(f"C++ PUCT 根局面与棋盘不符：{got!r} != {want!r}")

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

    def _collect_cpp(self, board: chess.Board, want: int):
        planes = np.empty((want,) + _PLANES_SHAPE, dtype=np.float32)
        pp = _ptr(planes, native._f32p)
        info = self._info
        while True:
            n = self._lib.kp_collect(self._ctx, want, pp, info, _ptr(self._path, native._u16p),
                                     _MAX_PATH)
            if n != -1:
                native.check(n, "收集叶子")
                return planes, n, info[0], info[1], info[2]
            probe = board.copy(stack=True)
            for code in self._path[:info[3]]:
                probe.push(code_move(int(code)))
            value = self.oracle.exact_value(probe)
            native.check(self._lib.kp_probe_answer(self._ctx, int(value is not None),
                                                   0.0 if value is None else float(value)),
                         "回填残局表")

    # ---------- 对外 ----------

    def search(self, board: chess.Board, simulations: Optional[int] = None,
               add_noise: bool = False, root: Optional[CppNode] = None,
               deadline: Optional[float] = None) -> Think[CppNode]:
        root_reused = root is not None and root.expanded
        t_enter = time.perf_counter()
        m = self.last_metrics = {"simulations": 0, "stopped_early": False,
                                 "network_positions": 0, "network_batches": 0, "max_depth": 0,
                                 "collisions": 0, "reused_root": root_reused}
        cfg = self.cfg
        sims = simulations or cfg.simulations
        root = root if root is not None else CppNode.new()
        L, ctx = self._lib, self._ctx
        self._load_root(board)
        native.check(L.kp_begin(ctx, root._h), "开始搜索")
        try:
            if not root_reused:
                exact = self.exact_value(board)
                if exact is not None:
                    L.kp_node_set_terminal(root._h, float(exact))
                    return root
                planes = np.empty((1,) + _PLANES_SHAPE, dtype=np.float32)
                native.check(L.kp_encode_root(ctx, _ptr(planes, native._f32p)), "编码根局面")
                policy, promo, wdl = yield from self._evaluate(planes, 1)
                m["network_positions"] += 1
                m["network_batches"] += 1
                n_moves = native.check(
                    L.kp_expand_root(ctx, _ptr(policy, native._f32p), _ptr(promo, native._f32p),
                                     _ptr(wdl, native._f32p)), "展开根节点")
                if n_moves == 0:
                    return root

            if add_noise:
                P = root.P
                if len(P) > 1:
                    noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(P))
                    newP = np.ascontiguousarray(((1 - cfg.dirichlet_eps) * P
                                                 + cfg.dirichlet_eps * noise).astype(np.float32))
                    native.check(L.kp_node_set_P(root._h, _ptr(newP, native._f32p), len(newP)),
                                 "写入根噪声")

            done = 0
            t0 = time.perf_counter()
            t_root = t0 - t_enter if not root_reused else 0.0
            stopped_early = False
            while done < sims:
                want = min(cfg.batch_size, sims - done)
                if deadline is not None:
                    now = time.perf_counter()
                    if now >= deadline:
                        stopped_early = True
                        break
                    left = deadline - now
                    if done > 0:
                        want = max(1, min(want, int(done / max(now - t0, 1e-9) * left)))
                    elif self._sim_rate:
                        want = max(1, min(want, int(self._sim_rate * left)))
                    elif t_root > 1e-4:
                        want = min(want, max(1, int(left / t_root)))
                        want = min(want, DEADLINE_PROBE_SIMS)
                    else:
                        want = min(want, DEADLINE_PROBE_SIMS)
                planes, n, terminal_sims, depth, collisions = self._collect_cpp(board, want)
                m["collisions"] = collisions
                if n == 0 and terminal_sims == 0:
                    break
                if n:
                    m["network_positions"] += n
                    m["network_batches"] += 1
                    m["max_depth"] = max(m["max_depth"], depth)
                    policy, promo, wdl = yield from self._evaluate(planes, n)
                    native.check(L.kp_apply(ctx, _ptr(policy, native._f32p),
                                            _ptr(promo, native._f32p), _ptr(wdl, native._f32p), n),
                                 "回传叶子结果")
                done += n + terminal_sims
            m["simulations"] = done
            m["stopped_early"] = stopped_early
            spent = time.perf_counter() - t0
            if done > 0 and spent > 1e-6:
                self._sim_rate = done / spent
            return root
        finally:
            L.kp_begin(ctx, None)

    @staticmethod
    def advance_root(root: Optional[CppNode], move: chess.Move) -> Optional[CppNode]:
        if root is None:
            return None
        h = root._lib.kp_node_advance(root._h, move_code(move))
        return CppNode(h) if h else None

    @staticmethod
    def principal_variation(root: CppNode, max_len: int = 24) -> list:
        buf = np.zeros(max_len, dtype=np.uint16)
        n = root._lib.kp_node_pv(root._h, max_len, _ptr(buf, native._u16p))
        return [code_move(int(c)) for c in buf[:n]]
