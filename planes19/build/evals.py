"""Lichess 评估库（``lichess_db_eval.jsonl.zst``）→ 96 字节监督分片 ``evals_XXXX.bin``。

每条自带 Stockfish 多条 PV，同时给出价值标签与策略软标签。默认只保留 PV ≥ 2 的条目
（单条 PV 只能当 one-hot，策略信号太弱）。

格式要点（已实测）：
1. **cp 是白方视角**：黑方行棋时取负才是行棋方视角。符号弄反网络会学着走最差的一手。
2. FEN 只有 4 个字段，补 `` 0 1``（五十步计数、重复次数在这批数据里恒为 0）。
3. 评估库把易位写成「王吃己车」e1h1 / e1a1。**必须用 ``board.parse_uci``**：
   ``Move.from_uci`` 原样返回 Move(e1, h1) 且能通过 ``in board.legal_moves``（python-chess 内部
   先归一化再判定），索引会落到推理端永远读不到的 4*64+7。旧 ``data/shards_evals`` 就是这样建坏的
   （训练端 ``repair_castling`` 补救），``tests/test_planes19_build.py`` 回归。
"""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import chess
import numpy as np

from ..encoding import move_to_index, move_to_promo_index, orient_move
from ..records import RECORD_DTYPE, board_to_record
from .labels import MATE_CP, score_to_wdl, softmax_policy


def _pv_score_cp(pv: dict, sign: int):
    """一条 PV 的分值 → 行棋方视角厘兵（sign：白方行棋 +1，黑方 -1）。"""
    if pv.get("mate") is not None:
        return MATE_CP if int(pv["mate"]) * sign > 0 else -MATE_CP
    if pv.get("cp") is not None:
        return float(pv["cp"]) * sign
    return None


def eval_line_to_record(line: str, *, temperature: float = 90.0, min_pv: int = 2,
                        min_depth: int = 12):
    """一行 jsonl → 记录（np.void）或 None（被过滤）。"""
    try:
        d = json.loads(line)
    except Exception:
        return None
    evals = d.get("evals") or []
    if not evals:
        return None
    best = max(evals, key=lambda e: e.get("depth", 0))
    if best.get("depth", 0) < min_depth:
        return None
    pvs = best.get("pvs") or []
    if len(pvs) < min_pv:
        return None
    fen = d.get("fen")
    if not fen:
        return None
    if len(fen.split()) == 4:
        fen = fen + " 0 1"
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    if not board.is_valid() or board.is_game_over(claim_draw=False):
        return None
    sign = 1 if board.turn == chess.WHITE else -1
    cands, seen = [], set()
    for pv in pvs:
        uci = (pv.get("line") or "").split(" ", 1)[0]
        if not uci:
            continue
        score = _pv_score_cp(pv, sign)
        if score is None:
            continue
        try:
            mv = board.parse_uci(uci)          # 见模块说明第 3 条
        except Exception:
            continue
        om = orient_move(mv, board.turn)
        idx = move_to_index(om)
        if idx in seen:
            continue
        seen.add(idx)
        cands.append((idx, move_to_promo_index(om), score))
    if len(cands) < min_pv:
        return None
    policy = softmax_policy(cands, temperature)
    top = max(cands, key=lambda c: c[2])
    wdl = score_to_wdl(None, 1 if top[2] > 0 else -1) if abs(top[2]) >= MATE_CP \
        else score_to_wdl(top[2], None)
    return board_to_record(board, policy=policy, wdl=wdl, promo=top[1], rep=0)


def process_lines(args) -> bytes:
    lines, temperature, min_pv, min_depth = args
    out = np.zeros(len(lines), dtype=RECORD_DTYPE)
    n = 0
    for line in lines:
        rec = eval_line_to_record(line, temperature=temperature, min_pv=min_pv,
                                  min_depth=min_depth)
        if rec is not None:
            out[n] = rec
            n += 1
    return out[:n].tobytes()


def iter_lines(path: Path, batch: int, limit: int = 0):
    """流式解压 jsonl.zst（或读纯文本 .jsonl），按批产出文本行。"""
    if str(path).endswith(".zst"):
        import zstandard as zstd
        raw = open(path, "rb")
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
    else:
        text = open(path, encoding="utf-8")
    with text:
        buf, total = [], 0
        for line in text:
            buf.append(line)
            total += 1
            if len(buf) >= batch:
                yield buf
                buf = []
            if limit and total >= limit:
                break
        if buf:
            yield buf


class ShardWriter:
    """按记录数切分片写出：``<prefix>_0000.bin`` …；空的最后一片删除。"""

    def __init__(self, out_dir, prefix: str, shard_records: int, dtype=RECORD_DTYPE):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix, self.limit, self.itemsize = prefix, shard_records * dtype.itemsize, dtype.itemsize
        self.shard, self.bytes, self.kept = 0, 0, 0
        self.fh = open(self._path(), "wb")

    def _path(self) -> Path:
        return self.out_dir / f"{self.prefix}_{self.shard:04d}.bin"

    def write(self, blob: bytes) -> None:
        if blob:
            self.fh.write(blob)
            self.bytes += len(blob)
            self.kept += len(blob) // self.itemsize
        if self.bytes >= self.limit:
            self.fh.close()
            self.shard += 1
            self.bytes = 0
            self.fh = open(self._path(), "wb")

    def close(self) -> None:
        self.fh.close()
        last = self._path()
        if last.exists() and last.stat().st_size == 0:
            last.unlink()


def build(evals, out, *, workers: int = 4, batch: int = 4000, shard_records: int = 2_000_000,
          temperature: float = 90.0, min_pv: int = 2, min_depth: int = 12, limit: int = 0,
          prefix: str = "evals") -> dict:
    writer = ShardWriter(out, prefix, shard_records)
    t0 = time.time()
    lines_in = 0
    tasks = ((b, temperature, min_pv, min_depth) for b in iter_lines(Path(evals), batch, limit))
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, blob in enumerate(pool.imap(process_lines, tasks, chunksize=2), 1):
                lines_in += batch
                writer.write(blob)
                if i % 250 == 0:
                    el = time.time() - t0
                    print(f"[{el / 60:6.1f} 分] 读入约 {lines_in:,} 行 | 保留 {writer.kept:,} | "
                          f"分片 {writer.shard}", flush=True)
    finally:
        writer.close()
    return {"kept": writer.kept, "shards": writer.shard + (1 if writer.bytes else 0),
            "sec": round(time.time() - t0, 1)}
