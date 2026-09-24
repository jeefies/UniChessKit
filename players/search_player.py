"""SearchPlayer：残局表 → 开局书 → 搜索 → 网络直出 的出招链。

对应 R ``engine/engine.py:UniChessEngine.play``，把原先写死在引擎里的优先级链、树复用、
随机数都收进来，引擎只需提供 Expander。

与 R 的差别（有意为之）：
- 随机数全部由 GameStart.seed 派生（R 的 MCTS 用未播种的 default_rng，评测无法复现）；
- 每局一个实例，树不会跨局残留（R 靠调用方记得 reset_search）。
"""
from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import chess
import numpy as np

from ..api import immediate
from ..api.types import GameStart, Leaf, MoveDecision, SearchBudget, Think
from ..search.puct import PUCT, PUCTConfig

_ROOT_MAX_SKIP = 4     # 跳过这么多手以上就重建：重放的收益追不上失配的风险


def open_polyglot(path):
    """打开 Polyglot 开局书；路径为空或不存在返回 None。"""
    if not path or not Path(path).exists():
        return None
    import chess.polyglot
    return chess.polyglot.open_reader(str(path))


class SearchPlayer:
    def __init__(self, name: str, expander, *, simulations: int = 0,
                 puct: Optional[PUCTConfig] = None, oracle=None, book=None,
                 book_plies: int = 10, temperature: float = 0.0, reuse_tree: bool = True,
                 planes_evaluator=None):
        self.name = name
        self.expander = expander
        self.simulations = simulations
        self.puct_cfg = replace(puct or PUCTConfig(), temperature=temperature)
        self.oracle = oracle
        self.book = book
        self.book_plies = book_plies
        self.temperature = temperature
        self.reuse_tree = reuse_tree
        self.planes_evaluator = planes_evaluator    # 给出时用 C++ PUCT（search/puct_cpp.py）
        self.new_game_called = False
        self.record_visits = False
        self._reset(0)

    # ---------- 生命周期 ----------

    def _reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if self.planes_evaluator is not None:
            from ..search.puct_cpp import PUCTCpp
            self.search = PUCTCpp(self.planes_evaluator, self.puct_cfg, oracle=self.oracle,
                                  rng=self.np_rng, expander=self.expander)
        else:
            self.search = PUCT(self.expander, self.puct_cfg, oracle=self.oracle, rng=self.np_rng)
        self._root, self._root_ply, self._root_epd = None, -1, None
        self.last_info: dict = {}

    def new_game(self, start: GameStart) -> Think[None]:
        self._reset(start.seed)
        self.record_visits = start.both_sides      # 自对弈：决策里带根访问分布（训练目标）
        self.new_game_called = True
        return immediate(None)

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]:
        return immediate(None)       # 树的推进在 choose 里按 EPD 核对后进行（见 _take_root）

    def close(self) -> None:
        self._root = None

    # ---------- 出招 ----------

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]:
        temperature = self.temperature if budget.temperature is None else budget.temperature
        mv = self._from_tablebase(board)
        if mv is not None:
            return MoveDecision(mv, "tablebase")
        mv = self._from_book(board)
        if mv is not None:
            return MoveDecision(mv, "book")
        sims = self.simulations if budget.simulations is None else budget.simulations
        if sims > 0:
            mv = yield from self._from_search(board, sims, temperature, budget)
            if mv is not None and mv in board.legal_moves:
                return MoveDecision(mv, "search", dict(self.last_info))
        mv = yield from self._from_policy(board, temperature)
        return MoveDecision(mv, "policy", dict(self.last_info))

    def _from_tablebase(self, board: chess.Board) -> Optional[chess.Move]:
        if self.oracle is None:
            return None
        mv = self.oracle.best_move(board)
        return mv if mv is not None and mv in board.legal_moves else None

    def _from_book(self, board: chess.Board) -> Optional[chess.Move]:
        if self.book is None or board.fullmove_number * 2 > self.book_plies:
            return None
        try:
            entries = list(self.book.find_all(board))
        except Exception:
            return None
        if not entries:
            return None
        weights = [max(e.weight, 1) for e in entries]
        mv = self.rng.choices([e.move for e in entries], weights=weights)[0]
        return mv if mv in board.legal_moves else None

    def _from_search(self, board, sims, temperature, budget) -> Think:
        reused = self._take_root(board) if self.reuse_tree else None
        try:
            mv, root = yield from self.search.best_move(
                board, simulations=sims, temperature=temperature,
                add_noise=budget.add_noise, root=reused, deadline=budget.deadline)
        except Exception:
            if reused is None:
                raise
            print("info string 搜索树复用失败，本步从零重建", file=sys.stderr)
            mv, root = yield from self.search.best_move(
                board, simulations=sims, temperature=temperature,
                add_noise=budget.add_noise, deadline=budget.deadline)
        self._root, self._root_ply, self._root_epd = root, len(board.move_stack), board.epd()
        self.last_info = self._search_info(root)
        return mv

    def _from_policy(self, board, temperature) -> Think:
        (ev,) = yield from self.expander.expand([Leaf(board=board)])
        if not ev.moves:
            raise ValueError(f"{self.name}: 局面没有合法着法 {board.fen()}")
        scores = np.asarray(ev.priors, dtype=np.float64)
        self.last_info = {"value": float(ev.value)}
        if temperature <= 0:
            return ev.moves[int(scores.argmax())]
        p = scores ** (1.0 / temperature)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            return ev.moves[int(scores.argmax())]
        return ev.moves[self.rng.choices(range(len(ev.moves)), weights=(p / total))[0]]

    # ---------- 树复用 ----------

    def _take_root(self, board: chess.Board):
        """把上一步的搜索树对齐到当前局面；对不上返回 None。

        真的把局面退回去比对 EPD：PUCT.search 在 root.expanded 时不核对局面，
        接错树不会报错，而是静默按另一个局面下棋。
        """
        root, ply, epd = self._root, self._root_ply, self._root_epd
        self._root, self._root_ply, self._root_epd = None, -1, None
        if root is None or ply < 0 or epd is None:
            return None
        back = len(board.move_stack) - ply
        if back < 0 or back > _ROOT_MAX_SKIP:
            return None
        probe = board.copy(stack=True)
        for _ in range(back):
            probe.pop()
        if probe.epd() != epd:
            return None
        for mv in board.move_stack[ply:]:
            root = self.search.advance_root(root, mv)
            if root is None:
                return None
        return root

    def _search_info(self, root) -> dict:
        m = self.search.last_metrics
        info = {"sims": int(m.get("simulations", 0)), "nodes": int(m.get("network_positions", 0)),
                "depth": int(m.get("max_depth", 0)), "reused": bool(m.get("reused_root", False)),
                "stopped_early": bool(m.get("stopped_early", False))}
        if root is not None and root.moves and root.sum_N > 0:
            i = int(np.argmax(root.N))
            if root.N[i] > 0:
                info["q"] = float(root.W[i]) / float(root.N[i])
            info["pv"] = [mv.uci() for mv in self.search.principal_variation(root)]
            if self.record_visits:
                info["visits"] = [(mv.uci(), int(n)) for mv, n in zip(root.moves, root.N) if n > 0]
        return info
