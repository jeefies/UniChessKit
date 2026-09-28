"""自对弈 sink：把 ``pipelines.selfplay`` 的对局写成 160 字节自对弈记录分片（``*.sp.bin``）。

每个带根访问分布的决策（``MoveDecision.info["visits"]``，SearchPlayer 在 ``both_sides`` 对局里给出）
落一条记录；开局书 / 残局表 / 网络直出的决策没有访问分布，跳过。

- 着法索引按行棋方视角（与监督分片、``planes19.encoding`` 相同）；``promo`` 取访问最多的着法。
- ``z`` 由终局结果换到行棋方视角；截断局（``truncated``）记 0 并置 ``FLAG_TRUNCATED``。
- 一局的记录一次性追加写入并 flush：进程被杀最多丢掉正在写的那一局，
  已写的分片大小始终是记录长度的整数倍（``open_shard`` 会校验）。
- 旁边的 ``<name>.games.jsonl`` 记每局元数据（局号、结果、写入条数），续跑时据此跳过已写的局。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import chess
import numpy as np

from .encoding import move_to_index, move_to_promo_index, orient_move, repetitions_of
from .records import SELFPLAY_DTYPE, selfplay_record

_Z = {"1-0": (1, -1), "0-1": (-1, 1), "1/2-1/2": (0, 0)}


def game_records(record: dict, decisions) -> np.ndarray:
    """一局（``run_selfplay`` 的 record + decisions）→ 自对弈记录数组。"""
    z_white, z_black = _Z[record["result"]]
    truncated = record["termination"] == "truncated"
    board = chess.Board()
    rows = []
    for ply, (uci, dec) in enumerate(zip(record["moves"], decisions)):
        visits = (dec.info or {}).get("visits")
        if visits:
            turn = board.turn
            pairs = [(move_to_index(orient_move(chess.Move.from_uci(u), turn)), n)
                     for u, n in visits]
            best = chess.Move.from_uci(max(visits, key=lambda t: t[1])[0])
            q = dec.info.get("q", 0.0)
            rows.append(selfplay_record(
                board, visits=pairs, z=z_white if turn == chess.WHITE else z_black, q=q,
                game=record["game"], ply=ply, promo=move_to_promo_index(best),
                rep=repetitions_of(board), truncated=truncated))
        board.push(chess.Move.from_uci(uci))
    return np.array(rows, dtype=SELFPLAY_DTYPE) if rows else np.zeros(0, dtype=SELFPLAY_DTYPE)


class SelfPlayShardSink:
    """``sink.on_game_end(record, board, decisions)``：追加到 ``path``（须以 ``.sp.bin`` 结尾）。"""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.name.endswith(".sp.bin"):
            raise ValueError(f"自对弈分片须以 .sp.bin 结尾：{self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = self.path.with_name(self.path.name[:-len(".sp.bin")] + ".games.jsonl")
        self._repair()
        self.games = self.done_games()
        self.records = self.path.stat().st_size // SELFPLAY_DTYPE.itemsize \
            if self.path.exists() else 0

    def _repair(self) -> None:
        """截掉写了一半的尾巴：分片按元数据记录的总条数截断，元数据去掉残行。

        只允许**截小**：分片比元数据还短时（元数据留着、分片被外部截断 / 删掉 / 只拷了
        一半）说明两边已经对不上，必须报错——``truncate`` 往大改会把分片用零字节撑到
        元数据说的长度，多出来的"记录"棋盘字段全零、z/q 全零，``open_shard`` 还校验得过，
        训练侧默默读进一批空局面噪声（2026-09-27 补：宁可拒绝续跑，不留垃圾数据）。
        """
        n = 0
        lines = []
        if self.meta.exists():
            for line in self.meta.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break
                lines.append(line)
                n += int(rec["records"])
            self.meta.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        size = n * SELFPLAY_DTYPE.itemsize
        actual = self.path.stat().st_size if self.path.exists() else 0
        if actual > size:
            # 写了一半的尾巴（或元数据整体丢失）：截回整数条，宁少勿假
            with open(self.path, "r+b") as f:
                f.truncate(size)
        elif n and actual < size:
            raise ValueError(f"{self.path} 只有 {actual} 字节，元数据记录了 {n} 条共 {size} "
                             f"字节：分片比元数据短，无法安全续跑（截大只会造出全零记录），"
                             f"请核对这两个文件")

    def done_games(self) -> set:
        if not self.meta.exists():
            return set()
        return {json.loads(l)["game"] for l in self.meta.read_text(encoding="utf-8").splitlines()}

    def on_game_end(self, record: dict, board, decisions) -> None:
        recs = game_records(record, decisions)
        with open(self.path, "ab") as f:
            f.write(recs.tobytes())
            f.flush()
            os.fsync(f.fileno())
        meta = {"game": record["game"], "result": record["result"],
                "termination": record["termination"], "plies": record["plies"],
                "records": int(len(recs))}
        with open(self.meta, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.games.add(record["game"])
        self.records += len(recs)
