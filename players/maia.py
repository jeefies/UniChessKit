"""Maia Player：把 ``Kit.maia`` 的网络包成 Kit 的 Player 契约。

Maia 的用法与自研引擎不同：**它不用 MCTS**。Maia 论文的 9 个档位是在对应 Elo 段的
人类棋谱上分别训练的，1 个节点（纯策略网络）就是它的标称强度；加搜索会让它变强、
也不再像人类。所以这里默认 argmax 策略（确定性、可复现、可批处理），
``temperature > 0`` 时按 softmax 采样（复现论文口径）。

历史**增量维护**：``new_game`` 记下根局面（含开局库），之后每步由 ``observe``
追加（驱动器对双方 Player 都会在每步后调用）。只在历史与 ``choose`` 拿到的
``board`` 对不上时才回退到从 ``move_stack`` 重放——重放是 O(n²)，
长局会很疼，不能每步都做。repetition 计数同样增量维护。
"""
from __future__ import annotations

import math
import os
import random
from typing import Optional

import chess

from ..api import immediate
from ..api.types import GameStart, MoveDecision, SearchBudget, Think
from ..maia import encode_input, load_maia, move_to_index

__all__ = ['MaiaPlayer', 'make_maia_player_factory']


class MaiaPlayer:
    """纯策略（1 节点）Maia 引擎。"""

    def __init__(self, name: str, net, *, temperature: float = 0.0,
                 device: str = 'cpu', weights_path: str = ''):
        self.name = name
        self.net = net
        self.temperature = float(temperature)
        self.device = device
        self.weights_path = weights_path
        self.rng = random.Random(0)
        self._boards: list = []
        self._counts: dict = {}
        self._root0: Optional[chess.Board] = None

    # ---- Player 契约 ----
    def new_game(self, start: GameStart) -> Think[None]:
        self.rng = random.Random(start.seed)
        # 开局库之前的根局面：重放兜底时要用
        self._root0 = chess.Board(start.fen) if start.fen else chess.Board()
        root = self._root0.copy()
        for uci in (start.opening or ()):
            root.push_uci(uci)
        self._boards = [root.copy()]
        self._counts = {root._transposition_key(): 1}
        return immediate(None)

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]:
        history, counts = self._sync(board)
        import torch
        planes = encode_input(board, history, counts)
        with torch.no_grad():
            logits, _ = self.net.forward(planes[None])
        logits = logits[0]

        moves, scores = [], []
        for mv in board.legal_moves:
            idx = move_to_index(mv, board)
            if idx is None:          # 马升变不在 lc0 表内
                continue
            moves.append(mv)
            scores.append(float(logits[idx]))
        if not moves:
            raise ValueError(f'Maia 无法为 {board.fen()} 找到任何表内着法')

        if self.temperature > 0:
            t = self.temperature
            mx = max(s / t for s in scores)
            ws = [math.exp(s / t - mx) for s in scores]
            total = sum(ws)
            r = self.rng.random() * total
            acc, pick = 0.0, moves[-1]
            for mv, w in zip(moves, ws):
                acc += w
                if r <= acc:
                    pick = mv
                    break
            return immediate(MoveDecision(pick, 'policy',
                                          {'engine': self.name,
                                           'temperature': t}))

        best = max(range(len(moves)), key=lambda k: scores[k])
        return immediate(MoveDecision(moves[best], 'policy',
                                      {'engine': self.name,
                                       'policy_logit': scores[best]}))

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]:
        """着法生效后的局面；驱动器对双方 Player 都会调，所以两边历史都完整。"""
        last = self._boards[-1] if self._boards else None
        if (last is not None and last.move_stack
                and len(board.move_stack) == len(last.move_stack) + 1
                and board.move_stack[-1] == move):
            self._boards.append(board.copy())
            k = board._transposition_key()
            self._counts[k] = self._counts.get(k, 0) + 1
        return immediate(None)

    def close(self) -> None:
        self._boards = []
        self._counts = {}
        self._root0 = None

    # ---- 内部 ----
    def _sync(self, board: chess.Board):
        """返回 (history, repetition_counts)，与 board 对齐。"""
        if self._boards and self._boards[-1].fen() == board.fen():
            return self._boards, self._counts
        # 对不上（换局/续跑/外部摆局）：从 move_stack 重放兜底
        root = self._root0 if self._root0 is not None else chess.Board()
        hist = [root.copy()]
        b = root.copy()
        for mv in board.move_stack:
            b.push(mv)
            hist.append(b.copy())
        if hist[-1].fen() != board.fen():
            return [board.copy()], {board._transposition_key(): 1}
        counts: dict = {}
        for x in hist:
            k = x._transposition_key()
            counts[k] = counts.get(k, 0) + 1
        return hist, counts


def make_maia_player_factory(weights: str, *, name: Optional[str] = None,
                             device: str = 'cpu', temperature: float = 0.0):
    """EngineSpec 工厂：每局一个 MaiaPlayer，共享同一份网络权重。

    ``weights`` 是 ``maia-<elo>.pb.gz`` 的路径。档位名默认从文件名推
    （``maia-1500.pb.gz`` -> ``maia-1500``）。例::

        {"factory": "Kit.players.maia:make_maia_player_factory",
         "root": "/home/jeefy/UniChess",
         "kwargs": {"weights": "/home/jeefy/UniChess/Maia/maia-1500.pb.gz",
                    "device": "cpu"}}

    ⚠ 网络在工厂调用时加载一次、所有对局共享（只读推理），所以**权重路径要对**；
    换档位要新建工厂。跑 CPU 就够（435K 参数，单步约 1ms），不必跟 SSM 抢 GPU。
    """
    if not os.path.exists(weights):
        raise FileNotFoundError(f'Maia 权重不存在: {weights}（'
                                f'来源：CSSLab/maia-chess v1.0 release）')
    if name is None:
        base = os.path.basename(weights)
        name = base[:-6] if base.endswith('.pb.gz') else os.path.splitext(base)[0]

    net = load_maia(weights, device)
    net.eval()

    def factory():
        return MaiaPlayer(name, net, temperature=temperature, device=device,
                          weights_path=weights)

    factory.search_impl = 'policy'
    return factory
