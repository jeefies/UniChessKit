"""JSON 工具：把引擎给出的附加信息（可能含 numpy 标量、chess.Move）转成可序列化的形式。"""
from __future__ import annotations

from typing import Any

import chess

_JSON_SCALARS = (str, int, float, bool, type(None))


def json_safe(info: Any, depth: int = 0) -> Any:
    """只保留可 JSON 序列化的部分（numpy 标量转成 Python 数，其余丢弃）。"""
    if isinstance(info, _JSON_SCALARS):
        return info
    if hasattr(info, "item") and callable(info.item) and getattr(info, "ndim", 1) == 0:
        return info.item()
    if depth > 4:
        return None
    if isinstance(info, dict):
        out = {}
        for k, v in info.items():
            v = json_safe(v, depth + 1)
            if v is not None or info[k] is None:
                out[str(k)] = v
        return out
    if isinstance(info, (list, tuple)):
        return [json_safe(v, depth + 1) for v in info]
    if isinstance(info, chess.Move):
        return info.uci()
    return None
