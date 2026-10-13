"""不需要网络前向的 Player：随机、UCI 引擎（Stockfish 作锚点）。"""
from __future__ import annotations

import random
from typing import Optional

import chess

from ..api import immediate
from ..api.types import GameStart, MoveDecision, SearchBudget, Think


class RandomPlayer:
    """均匀随机着法；随机数由 GameStart.seed 派生，可复现。"""

    def __init__(self, name: str = "random"):
        self.name = name
        self.rng = random.Random(0)

    def new_game(self, start: GameStart) -> Think[None]:
        self.rng = random.Random(start.seed)
        return immediate(None)

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]:
        return immediate(MoveDecision(self.rng.choice(list(board.legal_moves)), "random"))

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]:
        return immediate(None)

    def close(self) -> None:
        pass


class UciPlayer:
    """外部 UCI 引擎。**同步阻塞**：它思考时驱动器里的其他对局也在等，只适合作锚点。

    limit 例：{"nodes": 100000} / {"time": 0.1} / {"depth": 12}。
    """

    def __init__(self, name: str, command, limit: dict, options: Optional[dict] = None):
        self.name = name
        self.command = command
        self.limit = limit
        self.options = options or {}
        self.engine = None

    def new_game(self, start: GameStart) -> Think[None]:
        import chess.engine
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.command)
            if self.options:
                self.engine.configure(self.options)
        return immediate(None)

    def choose(self, board: chess.Board, budget: SearchBudget) -> Think[MoveDecision]:
        import chess.engine
        result = self.engine.play(board, chess.engine.Limit(**self.limit),
                                  game=id(self))
        return immediate(MoveDecision(result.move, "uci"))

    def observe(self, board: chess.Board, move: chess.Move) -> Think[None]:
        return immediate(None)

    def close(self) -> None:
        engine, self.engine = self.engine, None
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def make_uci_player_factory(command, limit: dict, *, name: str = "uci",
                            options: Optional[dict] = None):
    """EngineSpec 工厂：无参 PlayerFactory（每局一个 UciPlayer，惰性起 UCI 子进程）。

    给 ``Kit match`` 当**外部锚点**用——自研引擎的 arena 全是内部相对 Elo（以自己冠军
    为零点），要报绝对 Elo 必须有一个外部锚，Stockfish 是最合适的那个。例::

        {"factory": "Kit.players.simple:make_uci_player_factory",
         "kwargs": {"command": "/path/to/stockfish", "limit": {"nodes": 100000},
                    "options": {"Threads": 1, "Hash": 16}}}

    ⚠ UciPlayer **同步阻塞**：它思考时驱动器里的其他对局都在等。锚点场景（对手弱、
    步数少、并发不高）可接受，别拿它当主力引擎跑长会战。
    """
    def factory():
        return UciPlayer(name, command, dict(limit), dict(options or {}))
    factory.search_impl = "uci"
    return factory
