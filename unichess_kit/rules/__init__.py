"""规则层：裁判、开局库、残局表。只依赖 chess / numpy。"""
from .openings import BUNDLED_OPENINGS, OpeningBook, parse_line
from .referee import TERMINATION_REASON, TRUNCATED, StandardReferee, classify
from .tablebase import TablebaseOracle

__all__ = ["BUNDLED_OPENINGS", "OpeningBook", "parse_line", "TERMINATION_REASON",
           "TRUNCATED", "StandardReferee", "classify", "TablebaseOracle"]
