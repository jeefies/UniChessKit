"""Syzygy 残局表：精确价值与最佳着法。

以 R 的实现为准（R ``engine/engine.py:_from_tablebase`` 与 ``search/mcts.py:_exact_value``），
T 的同名逻辑缺了其中的 50 步保护，属于漂移 bug，不采用。

两条必须保留的规则：
1. **50 步规则先于残局表结论生效。** DTZ 不计半步钟；``halfmove_clock + dtz >= 100`` 时
   「必胜 / 必负」实战兑现不了，按和棋处理（排序时降级为 0；搜索里则放弃精确值、交还给搜索）。
2. **结果相同时优先清零着法（推兵 / 吃子）。** 兵残局里 DTZ 恒在 2 附近，不提供梯度，
   不优先推兵的话王会原地打转直到 50 步和棋。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import chess


class TablebaseOracle:
    """包装一个具有 probe_wdl / probe_dtz 的对象（chess.syzygy.Tablebase 或测试替身）。"""

    def __init__(self, tablebase, max_pieces: int = 5):
        self.tablebase = tablebase
        self.max_pieces = max_pieces
        self._reported: set = set()

    @classmethod
    def open(cls, path, max_pieces: int = 5) -> Optional["TablebaseOracle"]:
        """路径不存在或打开失败返回 None（调用方据此退回搜索）。"""
        if not path or not Path(path).is_dir():
            return None
        try:
            import chess.syzygy
            return cls(chess.syzygy.open_tablebase(str(path)), max_pieces)
        except Exception as e:  # noqa: 表文件损坏等，只报一次
            print(f"info string 残局表加载失败: {e}", file=sys.stderr)
            return None

    def applicable(self, board: chess.Board) -> bool:
        # 非法局面（例如未行棋方被将军）会让 python-chess 生成吃王着法，进而探到不存在的表
        return chess.popcount(board.occupied) <= self.max_pieces and board.is_valid()

    def exact_value(self, board: chess.Board) -> Optional[float]:
        """行棋方视角的精确价值 {-1, 0, 1}；不能确定（含 50 步临界）返回 None。

        对应 R search/mcts.py 的 _exact_value 残局表分支（终局判断由调用方先做）。
        """
        if not self.applicable(board):
            return None
        try:
            wdl = self.tablebase.probe_wdl(board)
        except Exception:
            return None
        # WDL: 2 必胜 / 1 困难胜（50 步内兑现不了）/ 0 和 / -1 / -2
        if abs(wdl) == 2 and board.halfmove_clock:
            try:
                dtz = abs(self.tablebase.probe_dtz(board))
            except Exception:
                return None
            if board.halfmove_clock + dtz >= 100:
                return None      # 临近 50 步申和，需要按规则搜索，不给精确值
        return float((wdl == 2) - (wdl == -2))

    def dtz(self, board: chess.Board) -> int:
        return abs(self.tablebase.probe_dtz(board))

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """残局表最佳着法；不适用或任何一步探测失败时返回 None（整步交还给搜索）。

        排序键 (对手 WDL, 是否非清零着法, DTZ) 取最小。
        """
        if not self.applicable(board):
            return None
        best, best_key = None, None
        for mv in list(board.legal_moves):
            board.push(mv)
            try:
                if board.is_checkmate():
                    wdl, dtz = -2, 0             # 对手被将死 = 我方必胜且最快
                elif board.is_stalemate() or board.is_insufficient_material():
                    wdl, dtz = 0, 0
                else:
                    wdl = self.tablebase.probe_wdl(board)        # 对手视角
                    dtz = abs(self.tablebase.probe_dtz(board))
                    if abs(wdl) == 2 and board.halfmove_clock + dtz >= 100:
                        wdl = 0                  # 50 步内兑现不了，实际就是和棋
            except Exception as e:
                key = f"{type(e).__name__}:{e}"
                if key not in self._reported:
                    self._reported.add(key)
                    print(f"info string 残局表探测失败，本局面回退到搜索: {e}", file=sys.stderr)
                return None
            finally:
                board.pop()
            zeroing = 0 if board.is_zeroing(mv) else 1
            key = (wdl, zeroing, dtz)
            if best_key is None or key < best_key:
                best, best_key = mv, key
        return best
