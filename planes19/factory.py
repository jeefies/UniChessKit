"""T/R 共用的 PlayerFactory 组装：引擎只需提供一个 BatchEvaluator。

显式给出的残局表 / 开局书路径打不开时直接报错：批量对弈里静默降级等于换了一个对手，
比报错更难发现。

搜索实现（``search_impl``，与 Python 版逐位一致，不影响结果，应放在 EngineSpec.runtime 里）：
- ``"auto"``（默认）：给了 ``planes_evaluator`` 就用 C++ PUCT，否则 Python PUCT；
- ``"cpp"``：必须给 ``planes_evaluator``，编译失败直接报错（不静默回退）；
- ``"python"``：强制 Python PUCT（对照 / 排障用）。
"""
from __future__ import annotations

from typing import Optional

from ..players.search_player import SearchPlayer, open_polyglot
from ..rules.tablebase import TablebaseOracle
from ..search.puct import PUCTConfig
from .expander import Planes19Expander

SEARCH_IMPLS = ("auto", "cpp", "python")


def make_search_player_factory(name: str, evaluator, *, simulations: int = 800,
                               batch_size: int = 64, syzygy_path: Optional[str] = None,
                               book_path: Optional[str] = None, book_plies: int = 10,
                               temperature: float = 0.0, reuse_tree: bool = True,
                               avoid_repetition: bool = True,
                               planes_evaluator=None, search_impl: str = "auto",
                               **puct_kwargs):
    """返回无参 PlayerFactory；评估器、残局表、开局书在所有对局间共享，每局新建 Player。

    evaluator 的负载是 chess.Board（网络直出兜底用，Python PUCT 也用它）；
    planes_evaluator 的负载是 (19,8,8) float32 编码（C++ PUCT 用）。
    ``avoid_repetition``：最优着法导致重复局面时改选次优非重复着法（自对弈必须开，
    否则同一 Player 执双方会镜像进三次重复循环，生成的数据全是和棋）。
    """
    if search_impl not in SEARCH_IMPLS:
        raise ValueError(f"{name}: search_impl 只能是 {SEARCH_IMPLS}，收到 {search_impl!r}")
    if search_impl == "cpp" and planes_evaluator is None:
        raise ValueError(f"{name}: search_impl='cpp' 需要 planes_evaluator")
    use_cpp = planes_evaluator is not None and search_impl != "python"
    if use_cpp:
        from ..search import native
        native.lib()                      # 立即编译 / 加载：出错在建工厂时就暴露
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
                            reuse_tree=reuse_tree, avoid_repetition=avoid_repetition,
                            planes_evaluator=planes_evaluator if use_cpp else None)
    factory.evaluator = evaluator
    factory.planes_evaluator = planes_evaluator if use_cpp else None
    factory.search_impl = "cpp" if use_cpp else "python"
    return factory
