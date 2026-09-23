"""搜索算法（策略模式）。搜索只通过 api.Expander 接触模型。"""
from .puct import PUCT, Node, PUCTConfig

__all__ = ["PUCT", "Node", "PUCTConfig"]
