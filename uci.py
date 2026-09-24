"""通用 UCI 前端：把任意 ``EngineSpec`` 包成 UCI 引擎（接 cutechess / 图形界面 / Stockfish 对测）。

    cd ~/UniChess && python -m Kit uci <spec.json>
    # spec.json 即一个 EngineSpec：{"factory": "ResNet.kit:make_player_factory", "kwargs": {...}}

支持：uci / isready / ucinewgame / position [startpos|fen ...] [moves ...] / go / stop / quit。

- ``go nodes N`` → ``SearchBudget(simulations=N)``；``go movetime T`` → 墙钟截止；
  ``go wtime/btime/winc/binc`` → 本方剩余时间 / 30 + 0.8 × 加秒（至少 50 ms）；``go`` / ``go infinite``
  → Player 自己的默认预算（同步搜索，``stop`` 无法打断）。
- 新 position 若是当前对局的延续，只把新增着法 observe 给 Player（保留搜索树）；否则新开一局。
- Player 给出 ``info["q"]``（行棋方视角 [-1, 1]）时换算成 ``info score cp`` 输出。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

import chess

from .api.types import GameStart, MoveDecision, SearchBudget
from .registry import EngineSpec, build_player_factory
from .runtime.batcher import run_sync


def q_to_cp(q: float) -> int:
    q = max(-0.999, min(0.999, float(q)))
    return int(round(111.714640912 * math.tan(1.5620688421 * q)))


def parse_position(tokens) -> chess.Board:
    """``position`` 之后的记号 → Board（带着法栈）。非法着法抛 ValueError。"""
    if not tokens:
        raise ValueError("position 缺参数")
    if tokens[0] == "startpos":
        board, rest = chess.Board(), tokens[1:]
    elif tokens[0] == "fen":
        if "moves" in tokens:
            k = tokens.index("moves")
            fen, rest = " ".join(tokens[1:k]), tokens[k:]
        else:
            fen, rest = " ".join(tokens[1:]), []
        board = chess.Board(fen)
    else:
        raise ValueError(f"position 不认识 {tokens[0]!r}")
    if rest:
        if rest[0] != "moves":
            raise ValueError(f"position 期望 moves，得到 {rest[0]!r}")
        for uci in rest[1:]:
            board.push(board.parse_uci(uci))
    return board


def budget_from_go(tokens, turn: bool, now: Optional[float] = None) -> SearchBudget:
    args = {}
    it = iter(tokens)
    for tok in it:
        if tok in ("nodes", "movetime", "wtime", "btime", "winc", "binc", "movestogo", "depth"):
            args[tok] = int(next(it))
    now = time.perf_counter() if now is None else now
    if "nodes" in args:
        return SearchBudget(simulations=max(1, args["nodes"]))
    if "movetime" in args:
        return SearchBudget(deadline=now + args["movetime"] / 1000.0)
    left = args.get("wtime" if turn == chess.WHITE else "btime")
    if left is not None:
        inc = args.get("winc" if turn == chess.WHITE else "binc", 0)
        togo = max(1, args.get("movestogo", 30))
        ms = max(50.0, min(left * 0.5, left / togo + 0.8 * inc))
        return SearchBudget(deadline=now + ms / 1000.0)
    return SearchBudget()


class UciSession:
    def __init__(self, spec: EngineSpec, out: TextIO = sys.stdout):
        self.spec = spec
        self.out = out
        self.player_factory = None
        self.player = None
        self.root_fen: Optional[str] = None      # 当前 Player 的起始局面
        self.moves: list = []                    # Player 已 observe 的着法
        self.board = chess.Board()
        self.game_seed = 0

    def send(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()

    # ---------------------------------------------------------------- Player 生命周期
    def _ensure_factory(self) -> None:
        if self.player_factory is None:
            self.player_factory = build_player_factory(self.spec)

    def _close_player(self) -> None:
        if self.player is not None:
            self.player.close()
            self.player = None

    def _sync_player(self) -> None:
        """让 Player 的局面与 self.board 一致：能续则 observe 新着法，否则新开一局。"""
        self._ensure_factory()
        stack = list(self.board.move_stack)
        root = self.board.root()
        root_fen = None if root.fen() == chess.STARTING_FEN else root.fen()
        extends = (self.player is not None and root_fen == self.root_fen
                   and stack[:len(self.moves)] == self.moves)
        if not extends:
            self._close_player()
            self.player = self.player_factory()
            self.game_seed += 1
            run_sync(self.player.new_game(GameStart(color=root.turn, seed=self.game_seed,
                                                    fen=root_fen, both_sides=True)))
            self.root_fen, self.moves = root_fen, []
        replay = root.copy()
        for mv in self.moves:
            replay.push(mv)
        for mv in stack[len(self.moves):]:
            replay.push(mv)
            run_sync(self.player.observe(replay.copy(), mv))
            self.moves.append(mv)

    # ---------------------------------------------------------------- 命令
    def handle(self, line: str) -> bool:
        """处理一行命令；返回 False 表示退出。"""
        tokens = line.strip().split()
        if not tokens:
            return True
        cmd, rest = tokens[0], tokens[1:]
        if cmd == "uci":
            self.send(f"id name {self.spec.name}")
            self.send("id author UniChess")
            self.send("uciok")
        elif cmd == "isready":
            self._ensure_factory()
            self.send("readyok")
        elif cmd == "ucinewgame":
            self._close_player()
            self.board = chess.Board()
        elif cmd == "position":
            try:
                self.board = parse_position(rest)
            except ValueError as e:
                self.send(f"info string 非法 position：{e}")
        elif cmd == "go":
            self.go(rest)
        elif cmd == "quit":
            self._close_player()
            return False
        elif cmd in ("stop", "ponderhit", "setoption", "debug", "register"):
            pass
        else:
            self.send(f"info string 未知命令 {cmd}")
        return True

    def go(self, tokens) -> None:
        if self.board.is_game_over(claim_draw=False):
            self.send("bestmove 0000")
            return
        self._sync_player()
        t0 = time.perf_counter()
        budget = budget_from_go(tokens, self.board.turn, now=t0)
        decision = run_sync(self.player.choose(self.board.copy(), budget))
        if not isinstance(decision, MoveDecision) or decision.move not in self.board.legal_moves:
            raise RuntimeError(f"{self.spec.name} 返回了非法着法 {decision!r}")
        info = decision.info or {}
        parts = [f"time {int((time.perf_counter() - t0) * 1000)}"]
        if "q" in info:
            parts.insert(0, f"score cp {q_to_cp(info['q'])}")
        if "simulations" in info:
            parts.append(f"nodes {int(info['simulations'])}")
        self.send("info " + " ".join(parts) + f" pv {decision.move.uci()}")
        self.send(f"bestmove {decision.move.uci()}")

    def loop(self, inp: TextIO = sys.stdin) -> None:
        try:
            for line in inp:
                if not self.handle(line):
                    break
        finally:
            self._close_player()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="UniChessKit 通用 UCI 前端")
    ap.add_argument("spec", help="EngineSpec JSON 文件")
    args = ap.parse_args(argv)
    spec = EngineSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    UciSession(spec).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
