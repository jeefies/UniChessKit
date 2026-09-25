"""PUCT 蒙特卡洛树搜索（协程版）。

逐行移植自 R ``search/mcts.py``（R 的版本修过最多的坑），改动只有一处：
网络前向从「直接调用 evaluator」改为 ``yield from expander.expand(leaves)``，
于是同一棵树的一批叶子仍是一个微批，而多盘棋的微批又能被 Batcher 拼成一次前向。
与 R 的逐位一致性由 tests/test_puct_parity.py 对照（设置 UNICHESS_R_ROOT 时运行）。

保留的设计与教训（细节见 R 的注释）：
- 子节点统计量用 numpy 数组，PUCT 向量化；c 随访问数按 c_puct_base / c_puct_init 增长。
- virtual loss 同时计为一次访问和一次负价值，否则待定路径看起来仍是中性，反复碰撞。
- FPU：未访问子节点取「父节点已探明价值 - fpu_reduction」。
- 根节点每个合法着法至少访问 root_min_visits 次（否则 P=1e-4 的一步杀可能一次都不访问）。
- 终局 / 残局表结算的下探也算一次模拟（否则搜到杀棋后循环空转，访问数失控）。
- 同一批里撞上待定叶子：只撤销本次路径的 virtual loss，继续找别的叶子（不提前送批）。
- 带墙钟时，批量大小本身受剩余时间约束（只在批尾比对 deadline 会把整步时间烧光）。

价值约定：全部是**当前行棋方视角**，每上升一层取负。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np

from ..api.types import Leaf, Think

DEADLINE_PROBE_SIMS = 8


@dataclass
class PUCTConfig:
    simulations: int = 800
    batch_size: int = 128           # 一次收集多少叶子再批量推理
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.8
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.2
    temperature: float = 0.0        # 0 = 取访问数最大者
    virtual_loss: float = 1.0
    claim_draw: bool = False        # 搜索内是否把「可申和」当作和棋终局（R 默认关）
    max_collision: int = 8          # 一批里同一叶子最多被选中几次
    root_min_visits: int = 1
    root_top_k: int = 0             # 温度采样只在访问数前 K 的根着法里进行（0 = 不限）
    # 走到本局（含搜索路径）已出现过的局面时按和棋（值 0）结算，不再下探（AlphaZero 口径）。
    # **默认关**：打开会改变搜索结果，会把 tests/golden/r_mcts 的冻结整树对拍打挂
    # （那个 fixture 里确实有重复路径）。自对弈生成配置里显式打开。
    repetition_draw: bool = False


class Node:
    __slots__ = ("moves", "P", "N", "W", "VL", "children",
                 "expanded", "terminal_value", "sum_N", "handle")

    def __init__(self):
        self.moves: list = []
        self.P = np.zeros(0, dtype=np.float32)
        self.N = np.zeros(0, dtype=np.int32)
        self.W = np.zeros(0, dtype=np.float32)
        self.VL = np.zeros(0, dtype=np.float32)
        self.children: list = []
        self.expanded = False
        self.terminal_value: Optional[float] = None
        self.sum_N = 0
        self.handle = None          # Expander 的私有句柄（S 的状态；T/R 为 None）

    def expand(self, moves, priors) -> None:
        n = len(moves)
        self.moves = list(moves)
        self.P = np.asarray(priors).astype(np.float32)
        self.N = np.zeros(n, dtype=np.int32)
        self.W = np.zeros(n, dtype=np.float32)
        self.VL = np.zeros(n, dtype=np.float32)
        self.children = [None] * n
        self.expanded = True

    def best_child(self, cfg: PUCTConfig) -> int:
        total = max(self.sum_N, 1)
        c = math.log((1 + total + cfg.c_puct_base) / cfg.c_puct_base) + cfg.c_puct_init
        denom = self.N + self.VL
        q = np.zeros_like(self.W)
        visited = denom > 0
        q[visited] = (self.W[visited] - cfg.virtual_loss * self.VL[visited]) / denom[visited]
        if visited.any():
            parent_q = float((self.W[visited] - cfg.virtual_loss * self.VL[visited]).sum()
                             / denom[visited].sum())
        else:
            parent_q = 0.0
        q[~visited] = parent_q - cfg.fpu_reduction
        u = c * self.P * math.sqrt(total) / (1.0 + denom)
        return int(np.argmax(q + u))


class PUCT:
    """expander: api.Expander；oracle: 可选的 rules.TablebaseOracle；rng: numpy Generator。"""

    def __init__(self, expander, cfg: Optional[PUCTConfig] = None, oracle=None,
                 rng: Optional[np.random.Generator] = None):
        self.expander = expander
        self.cfg = cfg or PUCTConfig()
        self.oracle = oracle
        self.rng = rng if rng is not None else np.random.default_rng()
        self._sim_rate: Optional[float] = None
        self.last_metrics: dict = {}

    # ---------- 终局 / 残局表 ----------

    def exact_value(self, board: chess.Board) -> Optional[float]:
        if board.is_checkmate():
            return -1.0
        if self.cfg.claim_draw and (board.is_repetition(3) or board.is_fifty_moves()):
            return 0.0
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            return 0.0
        if self.oracle is not None:
            return self.oracle.exact_value(board)
        return None

    # ---------- 一次批量迭代 ----------

    def _collect(self, root: Node, root_board: chess.Board, want: int):
        leaves = []
        terminal_sims = 0
        pending = set()
        spins = 0
        cfg = self.cfg
        while (len(leaves) + terminal_sims) < want and spins < cfg.max_collision * want:
            node = root
            board = root_board.copy(stack=True)
            path: list = []
            first = True
            while node.expanded and node.terminal_value is None:
                if not node.moves:
                    break
                if first and cfg.root_min_visits > 0:
                    unvisited = np.flatnonzero(node.N + node.VL < cfg.root_min_visits)
                    i = int(unvisited[0]) if unvisited.size else node.best_child(cfg)
                else:
                    i = node.best_child(cfg)
                first = False
                node.VL[i] += cfg.virtual_loss
                path.append((node, i))
                board.push(node.moves[i])
                child = node.children[i]
                if child is None:
                    child = Node()
                    node.children[i] = child
                node = child
                if (cfg.repetition_draw and node.terminal_value is None
                        and board.is_repetition(2)):
                    # 这一步走到本局（含搜索路径）里已出现过的局面：按和棋结算。
                    # AlphaZero 口径。没有它，搜索把重复当普通节点一路下探，
                    # 两个副本自对弈会互相镜像进三次重复循环（详见
                    # Kit/players/search_player.py::_avoid_repetition 的实测记录）。
                    node.expanded = True
                    node.terminal_value = 0.0
                    break                           # 交给下面按终局回填

            if node.terminal_value is not None:
                self._backup(path, node.terminal_value)
                terminal_sims += 1
                spins += 1
                continue
            if node.expanded:
                self._backup(path, 0.0)
                terminal_sims += 1
                spins += 1
                continue
            exact = self.exact_value(board)
            if exact is not None:
                node.expanded = True
                node.terminal_value = exact
                self._backup(path, exact)
                terminal_sims += 1
                spins += 1
                continue
            if id(node) in pending:
                self.last_metrics["collisions"] += 1
                for parent, edge in path:
                    parent.VL[edge] -= cfg.virtual_loss
                spins += 1
                continue
            pending.add(id(node))
            leaves.append((node, board, path))
            spins += 1
        return leaves, terminal_sims

    def _backup(self, path, value: float) -> None:
        v = value
        for node, i in reversed(path):
            v = -v
            node.N[i] += 1
            node.W[i] += v
            node.VL[i] -= self.cfg.virtual_loss
            node.sum_N += 1

    def _evaluate_and_expand(self, leaves) -> Think[None]:
        if not leaves:
            return
        m = self.last_metrics
        m["network_positions"] += len(leaves)
        m["network_batches"] += 1
        m["max_depth"] = max(m["max_depth"], max(len(p) for _, _, p in leaves))
        requests = [Leaf(board=b, parent_handle=p[-1][0].handle if p else None,
                         move=p[-1][0].moves[p[-1][1]] if p else None)
                    for _, b, p in leaves]
        evals = yield from self.expander.expand(requests)
        if len(evals) != len(leaves):
            raise ValueError(f"expander 返回 {len(evals)} 个结果，应为 {len(leaves)} 个")
        for (node, board, path), ev in zip(leaves, evals):
            node.handle = ev.handle
            if not ev.moves:
                node.expanded = True
                node.terminal_value = 0.0
                self._backup(path, 0.0)
                continue
            node.expand(ev.moves, ev.priors)
            self._backup(path, float(ev.value))

    # ---------- 对外 ----------

    def search(self, board: chess.Board, simulations: Optional[int] = None,
               add_noise: bool = False, root: Optional[Node] = None,
               deadline: Optional[float] = None) -> Think[Node]:
        root_reused = root is not None and root.expanded
        t_enter = time.perf_counter()
        self.last_metrics = {"simulations": 0, "stopped_early": False, "network_positions": 0,
                             "network_batches": 0, "max_depth": 0, "collisions": 0,
                             "reused_root": root_reused}
        cfg = self.cfg
        sims = simulations or cfg.simulations
        root = root or Node()

        if not root_reused:
            exact = self.exact_value(board)
            if exact is not None:
                root.expanded = True
                root.terminal_value = exact
                return root
            (ev,) = yield from self.expander.expand([Leaf(board=board)])
            self.last_metrics["network_positions"] += 1
            self.last_metrics["network_batches"] += 1
            root.handle = ev.handle
            if not ev.moves:
                root.expanded = True
                root.terminal_value = 0.0
                return root
            root.expand(ev.moves, ev.priors)

        if add_noise and len(root.moves) > 1:
            noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(root.moves))
            root.P = ((1 - cfg.dirichlet_eps) * root.P
                      + cfg.dirichlet_eps * noise).astype(np.float32)

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
            leaves, terminal_sims = self._collect(root, board, want)
            if not leaves and terminal_sims == 0:
                break
            yield from self._evaluate_and_expand(leaves)
            done += len(leaves) + terminal_sims
        self.last_metrics["simulations"] = done
        self.last_metrics["stopped_early"] = stopped_early
        spent = time.perf_counter() - t0
        if done > 0 and spent > 1e-6:
            self._sim_rate = done / spent
        return root

    @staticmethod
    def advance_root(root: Optional[Node], move: chess.Move) -> Optional[Node]:
        """实际走子后复用对应子树；该着法没被搜到则返回 None。"""
        if root is None or not root.expanded:
            return None
        for index, candidate in enumerate(root.moves):
            if candidate == move:
                child = root.children[index]
                if child is not None:
                    child.VL.fill(0.0)
                return child
        return None

    def best_move(self, board: chess.Board, simulations: Optional[int] = None,
                  temperature: Optional[float] = None, add_noise: bool = False,
                  root: Optional[Node] = None,
                  deadline: Optional[float] = None) -> Think[tuple]:
        root = yield from self.search(board, simulations, add_noise=add_noise, root=root,
                                      deadline=deadline)
        if not root.moves:
            legal = list(board.legal_moves)
            if not legal:
                raise ValueError("无合法走法")
            if self.oracle is not None and not board.is_game_over():
                mv = self._tablebase_root_move(board, legal)
                if mv is not None:
                    return mv, root
                # 根节点的残局表缺子节点 DTZ：绝不随便挑一步，改为关掉残局表重搜
                fallback = self._without_oracle()
                result = yield from fallback.best_move(board, simulations, temperature,
                                                       add_noise)
                self.last_metrics = fallback.last_metrics
                return result
            return legal[0], root

        t = self.cfg.temperature if temperature is None else temperature
        if t <= 0:
            i = int(np.argmax(root.N))
        else:
            # 只在访问数前 K 的着法里采样：T 实测不限范围的温度采样会选到排名很差的着法
            k = int(self.cfg.root_top_k)
            idx = (np.argsort(-root.N, kind="stable")[:k] if k > 0
                   else np.arange(len(root.N)))
            counts = root.N[idx].astype(np.float64) ** (1.0 / t)
            s = counts.sum()
            i = (int(np.argmax(root.N)) if s <= 0 or k == 1
                 else int(idx[self.rng.choice(len(counts), p=counts / s)]))
        return root.moves[i], root

    def _without_oracle(self) -> "PUCT":
        return PUCT(self.expander, self.cfg, oracle=None, rng=self.rng)

    def _tablebase_root_move(self, board: chess.Board, legal) -> Optional[chess.Move]:
        """根节点已由残局表定值时的选着（R search/mcts.py best_move 的回退分支）。"""
        ranked = []
        for move in legal:
            child = board.copy(stack=True)
            child.push(move)
            value = self.exact_value(child)
            if value is None:
                continue
            try:
                dtz = self.oracle.dtz(child)
            except Exception:
                dtz = 0 if child.is_game_over() else 1000
            # 键 (对手 WDL 取负, 赢棋时清零优先, 有向 DTZ)，取 max
            zeroing = 1 if (value < 0 and board.is_zeroing(move)) else 0
            ranked.append((-value, zeroing, -dtz if value < 0 else dtz, move))
        if ranked and (len(ranked) == len(legal) or max(r[0] for r in ranked) == 1):
            return max(ranked, key=lambda r: r[:3])[3]
        return None

    @staticmethod
    def visit_policy(root: Node) -> list:
        total = int(root.N.sum())
        if total <= 0:
            return []
        return [(mv, int(n) / total) for mv, n in zip(root.moves, root.N)]

    @staticmethod
    def root_value(root: Node) -> float:
        denom = int(root.N.sum())
        if denom <= 0:
            return 0.0 if root.terminal_value is None else root.terminal_value
        return float(root.W.sum() / denom)

    @staticmethod
    def principal_variation(root: Node, max_len: int = 24) -> list:
        node, pv = root, []
        while node is not None and node.moves and len(pv) < max_len:
            j = int(np.argmax(node.N))
            if node.N[j] <= 0:
                break
            pv.append(node.moves[j])
            node = node.children[j]
        return pv
