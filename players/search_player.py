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
from ..api.types import GameStart, Leaf, MoveDecision, PlayerError, SearchBudget, Think
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
                 planes_evaluator=None, avoid_repetition: bool = True):
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
        # 重复规避：最优着法导致重复局面时改选访问数次优的非重复着法（默认开）
        self.avoid_repetition = bool(avoid_repetition)
        self._game_book: list = []                  # GameStart.book（开局库强制着法）
        self._book_len = 0
        self._ply = 0
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
        # 自对弈的多局多样性：把局序号混进随机数流，口径与 SsmSelfPlayer 相同
        # （``default_rng(SeedSequence(seed, spawn_key=(index,)))``）。index=0（批量对弈
        # 口径）时保持旧行为逐位不变——arena 那边每局的种子本来就是按局现算的。
        # 注意不能在管线里改成传 per-game 整数种子：S 的开局 π′ 缓存键是
        # (seed, book_id, ply)，故意不含局序号，换了种子缓存就全落空。
        seed = int(start.seed)
        if getattr(start, "index", 0):
            seed = int(np.random.SeedSequence(seed, spawn_key=(int(start.index),))
                       .generate_state(1, dtype=np.uint32)[0])
        self._reset(seed)
        self.record_visits = start.both_sides      # 自对弈：决策里带根访问分布（训练目标）
        # pipelines.selfplay 的 GameStart.book：前若干 ply 原样走出，不搜索。
        # 这些 ply 没有访问分布，sink 自动跳过，与 polyglot 书的行为一致。
        # 不认这个字段的话，开局注入会让整批自对弈以 PlayerError 中止。
        self._game_book = [chess.Move.from_uci(u) for u in start.book]
        self._book_len = len(self._game_book)
        self._ply = 0
        self.new_game_called = True
        return immediate(None)

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]:
        return immediate(None)       # 树的推进在 choose 里按 EPD 核对后进行（见 _take_root）

    def close(self) -> None:
        self._root = None

    # ---------- 出招 ----------

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]:
        temperature = self.temperature if budget.temperature is None else budget.temperature
        self._ply = len(board.move_stack)          # choose 每 ply 调一次，开局库计数以它为准
        mv = self._from_tablebase(board)
        if mv is not None:
            return MoveDecision(mv, "tablebase")
        if self._ply < self._book_len:               # 开局库着法：原样走，不搜索
            mv = self._game_book[self._ply]
            if mv not in board.legal_moves:
                raise PlayerError(f"{self.name}: 开局库着法 {mv.uci()} 在 "
                                  f"{board.fen()} 不合法")
            return MoveDecision(mv, "book")
        mv = self._from_book(board)
        if mv is not None:
            return MoveDecision(mv, "book")
        sims = self.simulations if budget.simulations is None else budget.simulations
        if sims > 0:
            mv, root = yield from self._from_search(board, sims, temperature, budget)
            if mv is not None and mv in board.legal_moves:
                mv = self._avoid_repetition(board, mv, root)
                return MoveDecision(mv, "search", dict(self.last_info))
        mv = yield from self._from_policy(board, temperature)
        return MoveDecision(mv, "policy", dict(self.last_info))

    def _avoid_repetition(self, board: chess.Board, mv: chess.Move, root):
        """最优着法导致重复局面时，改选访问数次优的非重复着法。

        为什么必须做：自对弈里同一个 Player 执双方，两边都走「搜索认为最好」的一手；
        而对重复局面的估值在 evals 蒸馏数据上从没训过（输入第 18 通道 rep 恒 0），
        搜索也把重复当成普通节点，于是两个副本互相镜像进三次重复循环——
        2026-09-26 实测 64/128/256 sims 全部 16/16 threefold，终局 z 全 0，
        价值头拿不到任何梯度，生成的数据是废的。

        规则：只在**确有非重复替代**时才改选（所有着法都重复的极端局面仍走最优），
        按根节点访问数降序取第一个非重复着法；确定性，不引入新的随机数。
        """
        if not self.avoid_repetition or root is None:
            return mv
        moves = list(getattr(root, "moves", None) or [])
        if not moves:
            return mv
        N, W = root.N, root.W
        probe = board.copy(stack=True)
        probe.push(mv)
        if not probe.is_repetition(2):
            return mv                             # 最优着法不重复，正常走
        for i in sorted(range(len(moves)), key=lambda j: -int(N[j])):
            if N[i] <= 0:
                break                              # 之后都是没访问过的着法，没有次优可言
            cand = moves[i]
            if cand == mv:
                continue
            probe = board.copy(stack=True)
            probe.push(cand)
            if not probe.is_repetition(2):
                self._repoint_info(moves, N, W, i)
                return cand
        return mv                                  # 全是重复着法：无处可躲

    def _repoint_info(self, moves, N, W, i: int) -> None:
        """改选了着法：把 info 里的 q / pv 从原最优着法改到实际走的着法。"""
        info = self.last_info
        if N[i] > 0:
            info["q"] = float(W[i]) / float(N[i])
        info["pv"] = [moves[i].uci()]

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
        return mv, root

    def _from_policy(self, board, temperature) -> Think:
        (ev,) = yield from self.expander.expand([Leaf(board=board)])
        if not ev.moves:
            raise ValueError(f"{self.name}: 局面没有合法着法 {board.fen()}")
        scores = np.asarray(ev.priors, dtype=np.float64)
        self.last_info = {"value": float(ev.value)}
        if self._ply < self._book_len:               # 开局库着法同样优先于网络直出
            mv = self._game_book[self._ply]
            if mv not in board.legal_moves:
                raise PlayerError(f"{self.name}: 开局库着法 {mv.uci()} 在 "
                                  f"{board.fen()} 不合法")
            return mv
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
