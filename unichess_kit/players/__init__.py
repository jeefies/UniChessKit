"""Player 实现。它们可以互换（里氏替换），都要通过 testing.contracts 的契约测试。"""
from .search_player import SearchPlayer, open_polyglot
from .simple import RandomPlayer, UciPlayer

__all__ = ["SearchPlayer", "open_polyglot", "RandomPlayer", "UciPlayer"]
