"""serving：kit Player 与 Server 六方法 GameEngine 契约之间的双向适配。

- ``make_game_engine(factory)``：kit 原生引擎 → Server 的 ``GameEngine`` 类。
  ``models/<name>/engine.py`` 只需 ``GameEngine = make_game_engine(make_player_factory)``。
  同一组参数（预设）只加载一次模型，所有会话共享；每个会话持有自己的 Player（搜索树）。
- ``GameEnginePlayer``：已有的六方法 GameEngine → kit Player，让没有 kit 适配层的引擎
  （M6、或暂时保留 C++ 搜索的 T）也能进 kit 的对局管线（Server 的观战 / 批量对弈 job）。

六方法契约见 Server ``models/__init__.py``。终局判定统一用 ``rules.classify``（claim_draw 语义）。
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import random
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import chess

from .api.types import GameStart, MoveDecision, PlayerError, SearchBudget
from .jsonutil import json_safe
from .rules.referee import classify
from .runtime.batcher import run_sync

# ====================================================================== Player → GameEngine

class KitGameEngine:
    """Server 六方法 GameEngine 的 kit 实现基类；用 ``make_game_engine`` 生成具体类。"""

    IMPLEMENTED = True
    NOT_IMPLEMENTED_REASON = ""
    _factory: Callable = None                  # factory(**kwargs) -> PlayerFactory
    _shared: dict = None                       # kwargs 键 -> PlayerFactory（每个具体类一份）
    _shared_lock: threading.Lock = None

    def __init__(self, **kwargs):
        key = json.dumps(kwargs, sort_keys=True, default=str)
        cls = type(self)
        with cls._shared_lock:
            if key not in cls._shared:
                cls._shared[key] = cls._factory(**kwargs)
            self.player_factory = cls._shared[key]
        self.player = None
        self.start_fen: Optional[str] = None
        self.board = chess.Board()
        self.rng = random.Random()                 # 系统熵：不同会话 / 重启后走法可以不同
        self.setup()

    # ---------- 内部 ----------

    def _new_player(self) -> None:
        if self.player is not None:
            self.player.close()
        self.player = self.player_factory()
        start = GameStart(color=self.board.turn, seed=self.rng.randrange(1 << 31),
                          fen=self.start_fen)
        run_sync(self.player.new_game(start))

    def _replay(self) -> None:
        """按当前 board 重建 Player 状态（悔棋后用：搜索树不能跨越被撤销的着法）。"""
        moves = list(self.board.move_stack)
        self.board = chess.Board(self.start_fen) if self.start_fen else chess.Board()
        self._new_player()
        for mv in moves:
            self.board.push(mv)
            run_sync(self.player.observe(self.board.copy(), mv))

    def _done(self) -> bool:
        return classify(self.board) is not None

    # ---------- 六方法契约 ----------

    def setup(self, fen: Optional[str] = None) -> dict:
        board = chess.Board(fen) if fen else chess.Board()
        self.start_fen = None if board.fen() == chess.STARTING_FEN else board.fen()
        self.board = board
        self._new_player()
        return self.state()

    def human_move(self, uci: str) -> dict:
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"非法着法 {uci} @ {self.board.fen()}")
        self.board.push(move)
        run_sync(self.player.observe(self.board.copy(), move))
        return self.state()

    def engine_move(self) -> dict:
        if self._done():
            return {"engine_move": None, "fen": self.board.fen(), "done": True}
        decision = run_sync(self.player.choose(self.board.copy(), SearchBudget()))
        if not isinstance(decision, MoveDecision) or decision.move not in self.board.legal_moves:
            raise PlayerError(f"{getattr(self.player, 'name', '?')} 返回了非法着法 {decision!r}")
        mover = self.board.turn
        self.board.push(decision.move)
        run_sync(self.player.observe(self.board.copy(), decision.move))
        out = {"engine_move": decision.move.uci(), "fen": self.board.fen(), "done": self._done(),
               "source": decision.source}
        info = json_safe(decision.info) or {}
        if "q" in info:                      # 行棋方视角 → 白方视角，前端评估条用
            out["eval"] = info["q"] if mover == chess.WHITE else -info["q"]
        out["info"] = info
        return out

    def state(self) -> dict:
        return {"fen": self.board.fen(), "done": self._done()}

    def undo(self) -> dict:
        n = 2 if len(self.board.move_stack) >= 2 else len(self.board.move_stack)
        for _ in range(n):
            self.board.pop()
        self._replay()
        return self.state()

    def cleanup(self) -> None:
        if self.player is not None:
            self.player.close()
            self.player = None


def make_game_engine(factory: Callable, *, name: str = "GameEngine",
                     kit_factory: Optional[str] = None) -> type:
    """factory(**kwargs) -> PlayerFactory（与 EngineSpec 工厂同一约定）→ GameEngine 类。

    ``kit_factory``（"包.模块:函数"）写进类属性 ``KIT_FACTORY``，Server 据此让 job 进程
    直接用 kit 原生 Player（跨局攒批），而不是经 GameEnginePlayer 包一层。
    """
    attrs = {"_factory": staticmethod(factory), "_shared": {}, "_shared_lock": threading.Lock()}
    if kit_factory:
        attrs["KIT_FACTORY"] = kit_factory
    return type(name, (KitGameEngine,), attrs)


# ====================================================================== GameEngine → Player

class GameEnginePlayer:
    """把六方法 GameEngine 当 kit Player 用。**同步阻塞**（思考时不 yield），
    适合单局观战或每进程一局的多进程批量；跨局攒批请用 kit 原生 Player。

    引擎内部局面与对局局面的同步在 choose 时进行：对局局面是引擎已知局面的延续就补 human_move，
    否则（悔棋、换局面、契约测试直接给局面）setup 后整盘重放，不信任引擎自己的局面。
    """

    def __init__(self, name: str, make_engine: Callable[[], Any]):
        self.name = name
        self.make_engine = make_engine
        self.engine = None
        self.known: Optional[chess.Board] = None     # 引擎当前所处的局面（含历史）

    def new_game(self, start: GameStart):
        if self.engine is None:
            self.engine = self.make_engine()
        self._reset_to(start.board())
        return None
        yield  # noqa: 不可达；使本方法成为生成器

    def _reset_to(self, board: chess.Board) -> None:
        root = board.root()
        fen = root.fen()
        self.engine.setup(None if fen == chess.STARTING_FEN else fen)
        self.known = root
        for mv in board.move_stack:
            self.engine.human_move(mv.uci())
            self.known.push(mv)

    def _sync(self, board: chess.Board) -> None:
        known = self.known
        if known is None or known.root().fen() != board.root().fen():
            return self._reset_to(board)
        n = len(known.move_stack)
        if board.move_stack[:n] != known.move_stack:
            return self._reset_to(board)
        for mv in board.move_stack[n:]:
            self.engine.human_move(mv.uci())
            known.push(mv)

    def choose(self, board: chess.Board, budget: SearchBudget):
        if self.engine is None:
            self.engine = self.make_engine()
        self._sync(board)
        res = self.engine.engine_move()
        uci = res.get("engine_move") if isinstance(res, dict) else None
        if not uci:
            raise PlayerError(f"{self.name}.engine_move 未返回 engine_move：{res!r}")
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise PlayerError(f"{self.name} 走了非法着法 {uci} @ {board.fen()}")
        self.known.push(move)
        info = json_safe({k: v for k, v in res.items()
                          if k not in ("engine_move", "fen", "done", "source")}) or {}
        return MoveDecision(move, str(res.get("source") or "engine"), info)
        yield  # noqa: 不可达

    def observe(self, board: chess.Board, move: chess.Move):
        return None                        # 同步推迟到下一次 choose（见 _sync）
        yield  # noqa: 不可达

    def close(self) -> None:
        if self.engine is not None:
            engine, self.engine, self.known = self.engine, None, None
            engine.cleanup()


def load_engine_class(engine_path) -> type:
    """按路径加载 engine.py 的 GameEngine 类（与 Server models 加载器同样的方式）。"""
    path = Path(engine_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"engine.py 不存在：{path}")
    mod_name = "unichess_kit_engines.m" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    module = sys.modules.get(mod_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(mod_name, None)
            raise
    cls = getattr(module, "GameEngine", None)
    if cls is None:
        raise AttributeError(f"{path} 未定义 GameEngine")
    return cls


def game_engine_player_factory(engine_path, engine_kwargs: Optional[dict] = None,
                               name: str = "engine"):
    """EngineSpec 工厂：``{"factory": "unichess_kit.serving:game_engine_player_factory",
    "kwargs": {"engine_path": ".../engine.py", "engine_kwargs": {...}, "name": "T/max_mcts"}}``。"""
    cls = load_engine_class(engine_path)
    kwargs = dict(engine_kwargs or {})

    def factory():
        return GameEnginePlayer(name, lambda: cls(**kwargs))
    factory.name = name
    return factory
