"""T/R 的 Gumbel 自对弈 Player（走 ``GumbelCpp``）。

约定（照抄 S 的 ``SsmSelfPlayer``，见 SSM/AGENTS.md §8）：
- 走**顺序减半最终幸存着**（``res.move``，含根 Gumbel 噪声）；``temperature>0`` 时改为
  从 π′ 采样（多样性旋钮，评测/arena 用 0 = 确定性）；
- **训练目标 = π′**（全部合法着上的 ``softmax(ℓ+σ(completedQ))``），放进
  ``MoveDecision.info["visits"]``（``(uci, prob)`` 对）——sink 会按同一个归一化路径落
  ``visit_prob``，语义即"策略目标"。注意：Gumbel 下方根的 N 只有 m0 个候选有访问，
  **不能**再拿 N 当目标（会比 PUCT 的访问分布更稀疏、更差）；
- 开局 ply（``start.book``）照常搜索产出 π′ 但走 book 着法（flags 机制依赖它）；
- ``avoid_repetition``：与 ``SearchPlayer`` 同规则（按根 N 降序找第一个非重复着法；
  自对弈同一 Player 执双方时必须有，否则镜像进三次重复）。
"""
from __future__ import annotations

from typing import Optional

import chess
import numpy as np

from ..api import immediate
from ..api.types import GameStart, MoveDecision, PlayerError, SearchBudget, Think
from ..search.gumbel import C_SCALE, C_VISIT, GumbelConfig
from ..search.gumbel_cpp import GumbelCpp
from .search_player import open_polyglot


class GumbelPlayer:
    """``search`` 为 ``GumbelCpp``（或任何同签名的 Gumbel 搜索）。"""

    def __init__(self, name: str, search, *, book=None, book_plies: int = 10,
                 avoid_repetition: bool = True, temperature: float = 0.0):
        self.name = name
        self.search = search
        self.book = book
        self.book_plies = book_plies
        self.avoid_repetition = avoid_repetition
        self.temperature = temperature
        self.rng = np.random.default_rng(0)
        self.record_visits = False
        self.last_info: dict = {}
        self._game_book: list = []
        self._book_len = 0
        self._ply = 0

    # ---------- Player 协议 ----------

    def new_game(self, start: GameStart) -> Think[None]:
        seed = int(start.seed)
        if getattr(start, "index", 0):
            seed = int(np.random.SeedSequence(seed, spawn_key=(int(start.index),))
                       .generate_state(1, dtype=np.uint32)[0])
        self.rng = np.random.default_rng(seed)
        self.record_visits = start.both_sides
        self._game_book = [chess.Move.from_uci(u) for u in start.book]
        self._book_len = len(self._game_book)
        self._ply = 0
        self.last_info = {}
        return immediate(None)

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]:
        return immediate(None)

    def close(self) -> None:
        pass

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]:
        self._ply = len(board.move_stack)
        sims = self.search.cfg.simulations if budget.simulations is None else budget.simulations
        res = yield from self.search.search(board, rng=self.rng, simulations=sims)
        moves = list(getattr(res.root, "moves", None) or [])
        probs = np.zeros(0, dtype=np.float32)
        if res.root is not None:
            _, probs = res.pi_prime(self.search.cfg)
        if res.move is None:
            raise PlayerError(f"{self.name}: 搜索无着法（{board.fen()}）")
        info = {"sims": int(res.stats.get("sims_used", 0))}
        mv = res.move
        temp = self.temperature if budget.temperature is None else budget.temperature
        if temp > 0 and probs.size:
            p = np.power(probs, 1.0 / float(temp), dtype=np.float32)
            total = float(p.sum())
            if total > 0:
                mv = moves[int(self.rng.choice(len(moves), p=(p / total).astype(np.float64)))]
        if self.avoid_repetition:
            mv = self._avoid_repetition(board, mv, moves, res.root)
        if self._ply < self._book_len:               # 开局库：走库着，但 π′ 已由搜索产出
            bmv = self._game_book[self._ply]
            if bmv not in board.legal_moves:
                raise PlayerError(f"{self.name}: 开局库着法 {bmv.uci()} 在 {board.fen()} 不合法")
            self._finish_info(info, moves, probs, res.root, bmv)
            return MoveDecision(bmv, "book", dict(self.last_info))
        self._finish_info(info, moves, probs, res.root, mv)
        return MoveDecision(mv, "search", dict(self.last_info))

    # ---------- 内部 ----------

    def _finish_info(self, info, moves, probs, root, mv) -> None:
        if self.record_visits and moves and probs.size:
            info["visits"] = [(m.uci(), float(p)) for m, p in zip(moves, probs) if p > 0.0]
        if root is not None:
            try:
                j = moves.index(mv)
                if root.n[j] > 0:
                    info["q"] = float(root.q_sum[j]) / float(root.n[j])
                info["n"] = int(root.n[j])
            except (ValueError, IndexError):
                pass
        self.last_info = info

    def _avoid_repetition(self, board: chess.Board, mv: chess.Move, moves, root):
        """与 SearchPlayer 同规则：按根 N 降序取第一个不重复的着法。"""
        if not self.avoid_repetition or root is None or not moves:
            return mv
        probe = board.copy(stack=True)
        probe.push(mv)
        if not probe.is_repetition(2):
            return mv
        order = sorted(range(len(moves)), key=lambda j: -int(root.n[j]))
        for i in order:
            if int(root.n[i]) <= 0:
                break
            cand = moves[i]
            if cand == mv:
                continue
            probe = board.copy(stack=True)
            probe.push(cand)
            if not probe.is_repetition(2):
                return cand
        return mv


def make_gumbel_player_factory(name: str, planes_evaluator, *, simulations: int = 800,
                               m0: int = 16, g: float = 1.0, temperature: float = 0.0,
                               c_visit: float = C_VISIT, c_scale: float = C_SCALE,
                               claim_draw: bool = True, book_path: Optional[str] = None,
                               book_plies: int = 10,
                               avoid_repetition: bool = True):
    """返回无参 PlayerFactory（每个对局新建一个 GumbelPlayer + GumbelCpp 上下文）。"""
    book = None
    if book_path:
        book = open_polyglot(book_path)
        if book is None:
            raise FileNotFoundError(f"{name}: 开局书不存在：{book_path}")
    cfg = GumbelConfig(simulations=simulations, m0=m0, g=g, c_visit=c_visit,
                       c_scale=c_scale, claim_draw=claim_draw)

    def factory():
        return GumbelPlayer(name, GumbelCpp(planes_evaluator, cfg), book=book,
                            book_plies=book_plies, avoid_repetition=avoid_repetition,
                            temperature=temperature)
    factory.planes_evaluator = planes_evaluator
    factory.search_impl = "gumbel_cpp"
    return factory