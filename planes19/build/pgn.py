"""Lichess 月度 PGN → 96 字节分片 ``pgn_XXXX.bin``，**只带价值标签**（对局结果 WDL，真实的
五十步计数与重复次数）。策略概率全部留 0，训练时按「策略标签全零」屏蔽策略损失；
``with_policy`` 可写人类实走的 one-hot（仅供对照实验：会把策略头往人类风格拉）。

过滤：rapid / classical；双方 Elo ∈ [800, 2800] 且差 ≤ 300；Termination = Normal；
总步数 ≥ min_plies；跳过前 skip_plies 步。
"""
from __future__ import annotations

import io
import multiprocessing as mp
import re
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from ..encoding import move_to_index, move_to_promo_index, orient_move, repetitions_of
from ..records import RECORD_DTYPE, board_to_record
from .evals import ShardWriter

_TC = re.compile(r"^(\d+)\+(\d+)$")


def time_control_ok(tc: str) -> bool:
    """rapid / classical：初始 ≥ 600 秒，或 ≥ 180 秒且加秒 ≥ 2。"""
    m = _TC.match((tc or "").strip())
    if not m:
        return False
    base, inc = int(m.group(1)), int(m.group(2))
    return base >= 600 or (base >= 180 and inc >= 2)


def outcome_wdl(result: str, turn: bool):
    """对局结果 → 当前行棋方视角 (胜, 和, 负)；未完成的对局返回 None。"""
    if result == "1/2-1/2":
        return (0.0, 1.0, 0.0)
    if result not in ("1-0", "0-1"):
        return None
    win = (turn == chess.WHITE) == (result == "1-0")
    return (1.0, 0.0, 0.0) if win else (0.0, 0.0, 1.0)


def game_ok(headers) -> bool:
    if headers.get("Termination") != "Normal" or not time_control_ok(headers.get("TimeControl", "")):
        return False
    try:
        we, be = int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0))
    except ValueError:
        return False
    if not (800 <= we <= 2800 and 800 <= be <= 2800) or abs(we - be) > 300:
        return False
    return headers.get("Result", "*") in ("1-0", "0-1", "1/2-1/2")


def process_chunk(args) -> bytes:
    """一段 PGN 文本 → 记录字节串（worker 进程里跑）。"""
    text, skip_plies, min_plies, with_policy, cap = args
    out = np.zeros(cap, dtype=RECORD_DTYPE)
    n = 0
    stream = io.StringIO(text)
    while n < cap:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:
            break
        if game is None:
            break
        if not game_ok(game.headers):
            continue
        result = game.headers["Result"]
        moves = list(game.mainline_moves())
        if len(moves) < min_plies:
            continue
        board = game.board()                    # 重复次数要走子历史，不能丢栈
        for ply, mv in enumerate(moves):
            if ply >= skip_plies and n < cap:
                policy = promo = None
                if with_policy:
                    om = orient_move(mv, board.turn)
                    policy = [(move_to_index(om), 1.0)]
                    promo = move_to_promo_index(om)
                out[n] = board_to_record(board, policy=policy, wdl=outcome_wdl(result, board.turn),
                                         promo=promo, rep=repetitions_of(board))
                n += 1
            board.push(mv)
    return out[:n].tobytes()


def iter_chunks(path: Path, games_per_chunk: int):
    """流式读 pgn(.zst)，按若干局切块（以着法段后的空行为局界）。"""
    if str(path).endswith(".zst"):
        import zstandard as zstd
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(open(path, "rb")),
                                encoding="utf-8", errors="replace")
    else:
        text = open(path, encoding="utf-8", errors="replace")
    with text:
        buf, games, in_moves = [], 0, False
        for line in text:
            buf.append(line)
            if line.startswith("1."):
                in_moves = True
            elif in_moves and not line.strip():
                games += 1
                in_moves = False
                if games >= games_per_chunk:
                    yield "".join(buf)
                    buf, games = [], 0
        if buf:
            yield "".join(buf)


def build(pgn, out, *, workers: int = 4, games_per_chunk: int = 200,
          shard_records: int = 2_000_000, skip_plies: int = 8, min_plies: int = 20,
          with_policy: bool = False, max_records: int = 0, prefix: str = "pgn") -> dict:
    writer = ShardWriter(out, prefix, shard_records)
    t0 = time.time()
    tasks = ((c, skip_plies, min_plies, with_policy, 4096)
             for c in iter_chunks(Path(pgn), games_per_chunk))
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, blob in enumerate(pool.imap(process_chunk, tasks, chunksize=4), 1):
                writer.write(blob)
                if i % 500 == 0:
                    print(f"[{(time.time() - t0) / 60:6.1f} 分] 约 {i * games_per_chunk:,} 局 | "
                          f"保留 {writer.kept:,} | 分片 {writer.shard}", flush=True)
                if max_records and writer.kept >= max_records:
                    print("达到产出上限，停止")
                    break
    finally:
        writer.close()
    return {"kept": writer.kept, "sec": round(time.time() - t0, 1)}
