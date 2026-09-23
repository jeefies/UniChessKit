"""自对弈管线：一个 Player 执双方下完整盘，逐局交给引擎的 RecordSink 落盘。

口径移植自 S 的 ``tools/ssm_gumbel_selfplay.py``（Stage B 生成器），逐条对应：

- **开局注入**：第 g 局用开局库第 ``g % 库大小`` 条，裁到 ``book_plies``；这些着法**经 choose 走出**
  （GameStart.book），Player 照常搜索给出训练目标，管线校验走的正是 book 着法。
  开局库**不去重**：裁切后重复的线各自保留序号（S 的 ``load_openings`` 如此；去重会改变
  开局分配，破坏与既有数据的可比性）。非法线丢弃并计数。
- **分片**：局序号是全局的（``first_game`` 起），开局、随机数流、局键都只取决于全局序号，
  因此把 N 局拆给多个进程（各取不相交区间、同一 seed）与单进程跑出的是同一批对局。
- **随机数**：每局的随机数流由 (seed, 局序号) 决定——Player 用
  ``np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))``，与 S 的
  ``SeedSequence(seed).spawn(num_games)[index]`` 逐位相同（见 ``game_seed_sequence``）。
- **裁决**：rules.StandardReferee（claim_draw 语义；max_plies 按整盘计，含 book）。
- 出错整批停止（CoroutinePool 语义），Sink 不会收到半局。

可复现性：并发 1 时与 S 原生成器逐字节一致；并发 > 1 时批组成随完成顺序变化，
``SeqModel.step`` 的输出随批大小有 ~1e-5 浮点差，两边都只在统计意义上一致。
"""
from __future__ import annotations

import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess
import numpy as np

from ..api.types import GameStart, MoveDecision, PlayerError, SearchBudget
from ..rules.openings import parse_line
from ..rules.referee import StandardReferee
from ..runtime.batcher import Batcher, CoroutinePool


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int
    seed: int = 0
    max_plies: int = 300
    concurrency: int = 128
    openings: Optional[str] = None      # 开局文件路径；None = 全部从初始局面开始
    book_plies: int = 6
    simulations: Optional[int] = None   # 覆盖 Player 默认模拟数
    first_game: int = 0                 # 全局局序号起点（多进程分片：各进程取不相交区间）

    def __post_init__(self):
        if (self.games < 0 or self.max_plies < 1 or self.concurrency < 1 or self.book_plies < 0
                or self.first_game < 0):
            raise ValueError("games / book_plies / first_game >= 0，max_plies / concurrency >= 1")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SelfPlayTask:
    game: int
    book: tuple = ()
    book_id: Optional[int] = None


def game_seed_sequence(seed: int, index: int) -> np.random.SeedSequence:
    """第 index 局的随机数种子序列（= ``SeedSequence(seed).spawn(n)[index]``）。"""
    return np.random.SeedSequence(int(seed), spawn_key=(int(index),))


def load_book_lines(path, book_plies: int) -> tuple:
    """开局文件 → (UCI 元组列表, 丢弃的非法行数)。裁到 book_plies，不去重（见模块说明）。"""
    lines, dropped = [], 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        toks = raw.split("#", 1)[0].split()
        if not toks:
            continue
        try:
            lines.append(parse_line(" ".join(toks[:book_plies])))
        except ValueError:
            dropped += 1
    return lines, dropped


def plan_selfplay(cfg: SelfPlayConfig, lines: Optional[list] = None) -> list:
    games = range(cfg.first_game, cfg.first_game + cfg.games)
    if not lines:
        return [SelfPlayTask(game=g) for g in games]
    return [SelfPlayTask(game=g, book=tuple(lines[g % len(lines)]), book_id=g % len(lines))
            for g in games]


def play_selfplay_game(task: SelfPlayTask, player, referee: StandardReferee,
                       budget: SearchBudget, seed: int):
    """一局自对弈协程 → (record, 终局 board, decisions)。结束时（含异常）关闭 Player。"""
    t0 = time.perf_counter()
    board = chess.Board()
    decisions: list = []
    moves: list = []
    book = [chess.Move.from_uci(u) for u in task.book]
    try:
        yield from player.new_game(GameStart(color=chess.WHITE, seed=seed, game_id=f"g{task.game}",
                                             both_sides=True, book=tuple(task.book),
                                             book_id=task.book_id, index=task.game))
        while True:
            verdict = referee.verdict(board)
            if verdict is not None:
                break
            ply = len(board.move_stack)
            decision = yield from player.choose(board.copy(), budget)
            if not isinstance(decision, MoveDecision):
                raise PlayerError(f"{player.name}.choose 应返回 MoveDecision，"
                                  f"收到 {type(decision).__name__}")
            if ply < len(book) and decision.move != book[ply]:
                raise PlayerError(f"{player.name} 在第 {ply} ply 应走 book 着法 {book[ply]}，"
                                  f"却走了 {decision.move}（第 {task.game} 局）")
            if decision.move not in board.legal_moves:
                raise PlayerError(f"{player.name} 走了非法着法 {decision.move} @ {board.fen()}"
                                  f"（第 {task.game} 局）")
            board.push(decision.move)
            moves.append(decision.move.uci())
            decisions.append(decision)
            yield from player.observe(board.copy(), decision.move)
    finally:
        try:
            player.close()
        except Exception as e:  # noqa: close 失败不应掩盖真正的异常
            print(f"[selfplay] {getattr(player, 'name', player)}.close() 失败：{e}", file=sys.stderr)
    record = {
        "type": "game", "game": task.game, "book_id": task.book_id,
        "book_plies": min(len(book), len(moves)), "moves": moves,
        "result": verdict.result, "termination": verdict.termination,
        "plies": len(board.move_stack), "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    return record, board, decisions


def run_selfplay(cfg: SelfPlayConfig, make_player: Callable, sink,
                 progress: Optional[Callable[[dict, int], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """跑 cfg.games 局自对弈，逐局调 ``sink.on_game_end``，返回汇总。

    make_player 为 PlayerFactory（每局一个 Player）；跨局共享的状态（如 book ply 的搜索缓存）
    由引擎放在工厂里。
    """
    lines, dropped = (load_book_lines(cfg.openings, cfg.book_plies) if cfg.openings
                      else ([], 0))
    if cfg.openings and not lines:
        raise ValueError(f"开局文件 {cfg.openings} 没有有效开局")
    tasks = plan_selfplay(cfg, lines)
    referee = StandardReferee(max_plies=cfg.max_plies)
    budget = SearchBudget(simulations=cfg.simulations, add_noise=True)
    batcher = Batcher()
    pool = CoroutinePool(cfg.concurrency, batcher)
    totals = {"games": 0, "plies": 0, "book_plies": 0, "termination": {}}
    t0 = time.perf_counter()

    def on_done(_gid, out):
        record, board, decisions = out
        sink.on_game_end(record, board, decisions)
        totals["games"] += 1
        totals["plies"] += record["plies"]
        totals["book_plies"] += record["book_plies"]
        term = totals["termination"]
        term[record["termination"]] = term.get(record["termination"], 0) + 1
        if progress is not None:
            progress(record, totals["games"])

    jobs = ((t.game, (lambda t=t: play_selfplay_game(t, make_player(), referee, budget, cfg.seed)))
            for t in tasks)
    pool.run(jobs, on_done, should_stop=should_stop or (lambda: False))
    elapsed = time.perf_counter() - t0
    return {**totals, "termination": dict(sorted(totals["termination"].items())),
            "games_planned": cfg.games, "opening_lines": len(lines),
            "opening_lines_dropped": dropped, "elapsed_s": round(elapsed, 1),
            "batch": batcher.stats.as_dict(), "config": cfg.to_dict()}
