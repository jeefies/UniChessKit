"""T/R 共用的 PlayerFactory 组装：引擎只需提供一个 BatchEvaluator。

显式给出的残局表 / 开局书路径打不开时直接报错：批量对弈里静默降级等于换了一个对手，
比报错更难发现。
"""
from __future__ import annotations

from typing import Optional

from ...players.search_player import SearchPlayer, open_polyglot
from ...rules.tablebase import TablebaseOracle
from ...search.puct import PUCTConfig
from .expander import Planes19Expander


def make_search_player_factory(name: str, evaluator, *, simulations: int = 800,
                               batch_size: int = 64, syzygy_path: Optional[str] = None,
                               book_path: Optional[str] = None, book_plies: int = 10,
                               temperature: float = 0.0, reuse_tree: bool = True,
                               **puct_kwargs):
    """返回无参 PlayerFactory；评估器、残局表、开局书在所有对局间共享，每局新建 Player。"""
    oracle = None
    if syzygy_path:
        oracle = TablebaseOracle.open(syzygy_path)
        if oracle is None:
            raise FileNotFoundError(f"{name}: 残局表目录打不开：{syzygy_path}")
    book = None
    if book_path:
        book = open_polyglot(book_path)
        if book is None:
            raise FileNotFoundError(f"{name}: 开局书不存在：{book_path}")
    cfg = PUCTConfig(simulations=simulations, batch_size=batch_size, **puct_kwargs)
    expander = Planes19Expander(evaluator)

    def factory():
        return SearchPlayer(name, expander, simulations=simulations, puct=cfg, oracle=oracle,
                            book=book, book_plies=book_plies, temperature=temperature,
                            reuse_tree=reuse_tree)
    factory.evaluator = evaluator
    return factory
