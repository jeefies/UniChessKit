"""开局库与开局分配。

移植自 S 的 ``tools/ssm_gumbel_arena.py``（load_openings_file / opening_plan）。
要点（都是踩过的坑）：

- **确定性对局下，开局多样性是对局多样性的唯一来源。** 同一开局对（A/B 各执白一次）
  之外的任何重复都会产生逐字节相同的棋谱，白白虚增样本量。所以 n_pairs <= 库规模时
  用带种子的排列保证每对开局互不相同；库不够时循环复用，由结果里的 duplicate_rate 报警。
- 按固定 ply 数裁切会把不同开局折叠成相同开局（S 实测 200 → 111），所以默认不裁切，
  去重在最终形态上做。
- 每行逐着校验合法性；UCI 与 SAN 都接受（Server/R 的开局文件是 UCI，S 的是 SAN）；
  ``#`` 之后为注释。非法行默认报错而不是静默跳过——静默跳过会让库悄悄变小。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import chess
import numpy as np

BUNDLED_OPENINGS = Path(__file__).resolve().parent.parent / "data" / "openings.txt"


def parse_line(text: str) -> tuple:
    """一行开局（UCI 或 SAN，可混用）→ UCI 元组。非法着法抛 ValueError。"""
    board = chess.Board()
    out = []
    for tok in text.split():
        move = None
        try:
            cand = chess.Move.from_uci(tok)
            if cand in board.legal_moves:
                move = cand
        except ValueError:
            pass
        if move is None:
            try:
                move = board.parse_san(tok)
            except ValueError as e:
                raise ValueError(f"开局 {text!r} 第 {len(out) + 1} 着 {tok!r} 非法：{e}") from None
        out.append(move.uci())
        board.push(move)
    if board.is_game_over(claim_draw=True):
        raise ValueError(f"开局 {text!r} 走完后对局已结束")
    return tuple(out)


class OpeningBook:
    """一组去重后的开局（UCI 元组，保持文件顺序）。"""

    def __init__(self, lines: Iterable[Sequence[str]]):
        seen: set = set()
        self.lines: list = []
        for line in lines:
            t = tuple(line)
            if t in seen:
                continue
            seen.add(t)
            self.lines.append(t)

    def __len__(self) -> int:
        return len(self.lines)

    @classmethod
    def from_file(cls, path, *, max_plies: Optional[int] = None,
                  strict: bool = True) -> "OpeningBook":
        """读开局文件。max_plies 只在调用方明确需要时裁切（见模块说明）。

        strict=True 时任何非法行都抛错；False 时跳过（并在 skipped 里记下行号）。
        """
        lines, skipped = [], []
        for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            try:
                ucis = parse_line(text)
            except ValueError as e:
                if strict:
                    raise ValueError(f"{path}:{lineno}: {e}") from None
                skipped.append(lineno)
                continue
            if max_plies is not None:
                ucis = ucis[:max_plies]
            lines.append(ucis)
        book = cls(lines)
        book.skipped = skipped
        if not book.lines:
            raise ValueError(f"开局文件 {path} 没有有效开局")
        return book

    @classmethod
    def bundled(cls) -> "OpeningBook":
        return cls.from_file(BUNDLED_OPENINGS)

    def plan(self, n_pairs: int, seed: int) -> list:
        """为 n_pairs 个开局对分配开局，返回 [(库下标, UCI 元组)]，长度恒为 n_pairs。

        与 S 的 opening_plan 同算法（np.random.default_rng(seed).permutation），同 seed 可复现。
        """
        n = len(self.lines)
        if n == 0:
            raise ValueError("开局库为空，无法分配开局")
        idx = np.random.default_rng(seed).permutation(n)
        return [(int(idx[i % n]), self.lines[int(idx[i % n])]) for i in range(n_pairs)]
