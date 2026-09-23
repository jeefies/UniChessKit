"""不依赖 torch 的假引擎：确定性、便宜，用于 kit 自身的测试与引擎契约测试的对照。"""
from __future__ import annotations

import zlib

import chess
import numpy as np

from ..contrib.planes19 import Planes19Expander, POLICY_SIZE, PROMO_SIZE
from ..players import SearchPlayer
from ..search import PUCTConfig

_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class FakePlanesEvaluator:
    """(policy[4096], promo[4], wdl[3])：策略由局面 FEN 的哈希播种，价值由子力差决定。

    与 T/R 的 evaluate_batch 同形状、同视角约定（行棋方视角），可直接喂 Planes19Expander。
    calls 记录每次 evaluate 的批大小，便于断言攒批行为。
    """

    def __init__(self, model_key: str = "fake", salt: str = ""):
        self.model_key = model_key
        self.salt = salt
        self.calls: list = []

    def _one(self, board: chess.Board):
        seed = zlib.crc32((self.salt + board.fen()).encode())
        rng = np.random.default_rng(seed)
        policy = rng.random(POLICY_SIZE).astype(np.float32) ** 4
        policy /= policy.sum()
        promo = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)[:PROMO_SIZE]
        diff = 0
        for piece in board.piece_map().values():
            v = _VALUES.get(piece.piece_type, 0)
            diff += v if piece.color == board.turn else -v
        win = 1.0 / (1.0 + np.exp(-diff / 3.0))
        draw = 0.2
        wdl = np.array([win * (1 - draw), draw, (1 - win) * (1 - draw)], dtype=np.float32)
        return policy, promo, wdl

    def evaluate_batch(self, boards):
        """T/R 风格的接口：返回三个堆叠数组。"""
        outs = [self._one(b) for b in boards]
        return (np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs]),
                np.stack([o[2] for o in outs]))

    def evaluate(self, payloads):
        self.calls.append(len(payloads))
        return [self._one(b) for b in payloads]


class _FakeFactory:
    def __init__(self, name, evaluator, simulations, batch_size, temperature):
        self.name = name
        self.evaluator = evaluator
        self.expander = Planes19Expander(evaluator)
        self.simulations = simulations
        self.batch_size = batch_size
        self.temperature = temperature

    def __call__(self):
        return SearchPlayer(self.name, self.expander, simulations=self.simulations,
                            puct=PUCTConfig(batch_size=self.batch_size),
                            temperature=self.temperature)


def make_fake_player_factory(name: str = "fake", salt: str = "", simulations: int = 16,
                             batch_size: int = 8, temperature: float = 0.0):
    """registry 工厂约定的示例实现：加载一次「模型」，返回每局新建 Player 的工厂。"""
    return _FakeFactory(name, FakePlanesEvaluator(f"fake:{name}:{salt}", salt), simulations,
                        batch_size, temperature)


def make_failing_player_factory(fail_after: int = 3):
    """走 fail_after 步后抛异常的 Player 工厂（测试「出错整批停止」）。"""
    from ..players import RandomPlayer
    from ..api import immediate

    class _Failing(RandomPlayer):
        def __init__(self):
            super().__init__("failing")
            self.n = 0

        def choose(self, board, budget):
            self.n += 1
            if self.n > fail_after:
                raise RuntimeError("故意失败")
            return super().choose(board, budget)

    return _Failing


def make_random_player_factory(name: str = "random"):
    from ..players import RandomPlayer
    return lambda: RandomPlayer(name)


def crash_worker(wid, task, emit, stop_event):
    """模拟 worker 被 OOM / kill：不发 done/error 直接退出进程。"""
    import os
    emit({"wid": wid})
    os._exit(3)


def ok_worker(wid, task, emit, stop_event):
    for x in task:
        emit(x)


def error_worker(wid, task, emit, stop_event):
    emit({"wid": wid})
    raise ValueError(f"worker {wid} 故意失败")


def make_slow_player_factory(delay_s: float = 0.2, name: str = "slow"):
    """每步 sleep delay_s 的随机 Player（测试 job 的停止与实时快照）。"""
    import time as _time

    from ..players import RandomPlayer

    class _Slow(RandomPlayer):
        def choose(self, board, budget):
            _time.sleep(delay_s)
            return super().choose(board, budget)

    return lambda: _Slow(name)


class FakeGameEngine:
    """六方法 GameEngine 的最小实现（总走 UCI 字典序第一个合法着法），测试 serving 适配用。"""

    IMPLEMENTED = True
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.board = chess.Board()
        self.cleaned = 0
        self.calls: list = []
        FakeGameEngine.instances.append(self)

    def setup(self, fen=None):
        self.calls.append(("setup", fen))
        self.board = chess.Board(fen) if fen else chess.Board()
        return self.state()

    def human_move(self, uci):
        self.calls.append(("human_move", uci))
        self.board.push_uci(uci)
        return self.state()

    def engine_move(self):
        self.calls.append(("engine_move",))
        mv = sorted(self.board.legal_moves, key=lambda m: m.uci())[0]
        self.board.push(mv)
        return {"engine_move": mv.uci(), "fen": self.board.fen(), "done": False,
                "eval": np.float32(0.25), "nodes": np.int64(7)}

    def state(self):
        return {"fen": self.board.fen(), "done": self.board.is_game_over()}

    def undo(self):
        self.board.pop()
        return self.state()

    def cleanup(self):
        self.cleaned += 1
