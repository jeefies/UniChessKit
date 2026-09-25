"""用本机 Stockfish 给 FEN 打 MultiPV 策略软标签 + WDL → ``anno_XXXXXX.bin``。

每个 worker 独占一个 Stockfish 进程；按分片写（先写 .tmp 再原子改名），重启时跳过已完成的分片。
Stockfish 是 GPL 第三方二进制，**不入库**，路径由调用方给出。
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from ..records import RECORD_DTYPE, board_to_record
from .labels import policy_from_multipv, score_to_wdl


def _worker(args):
    shard_id, fens, out_path, sf_path, nodes, multipv, threads, hash_mb = args
    if Path(out_path).exists():
        return shard_id, -1
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": threads, "Hash": hash_mb})
    limit = chess.engine.Limit(nodes=nodes)
    recs = np.zeros(len(fens), dtype=RECORD_DTYPE)
    n = 0
    try:
        for fen in fens:
            try:
                board = chess.Board(fen)
                if board.is_game_over(claim_draw=False) or not board.is_valid():
                    continue
                infos = engine.analyse(board, limit, multipv=multipv)
                policy, promo = policy_from_multipv(board, infos)
                if not policy:
                    continue
                score = infos[0]["score"].pov(board.turn)
                wdl = score_to_wdl(None if score.mate() is not None else float(score.score()),
                                   score.mate())
                recs[n] = board_to_record(board, policy=policy, wdl=wdl, promo=promo)
                n += 1
            except (ValueError, chess.engine.EngineError):
                continue
    finally:
        engine.quit()
    tmp = str(out_path) + ".tmp"
    recs[:n].tofile(tmp)
    os.replace(tmp, out_path)
    return shard_id, n


def annotate(fens, out_dir, sf_path: str, *, workers: int = 14, shard_size: int = 20000,
             nodes: int = 100_000, multipv: int = 5, threads: int = 1, hash_mb: int = 256) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks, buf, sid = [], [], 0
    for fen in fens:
        buf.append(fen)
        if len(buf) >= shard_size:
            tasks.append((sid, buf, out_dir / f"anno_{sid:06d}.bin", sf_path, nodes, multipv,
                          threads, hash_mb))
            buf, sid = [], sid + 1
    if buf:
        tasks.append((sid, buf, out_dir / f"anno_{sid:06d}.bin", sf_path, nodes, multipv,
                      threads, hash_mb))
    todo = [t for t in tasks if not t[2].exists()]
    print(f"共 {len(tasks)} 个分片，待处理 {len(todo)}，{workers} 进程，"
          f"每局面 {nodes:,} nodes，MultiPV={multipv}", flush=True)
    done, t0 = 0, time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (shard_id, n) in enumerate(pool.imap_unordered(_worker, todo), 1):
            done += max(n, 0)
            print(f"[{i}/{len(todo)}] shard {shard_id}: {n} 条 | 累计 {done:,} | "
                  f"{done / max(time.time() - t0, 1e-9):.1f} 局面/秒", flush=True)
    return {"shards": len(tasks), "done": len(todo), "records": done}
