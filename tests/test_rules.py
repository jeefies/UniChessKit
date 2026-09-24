import tempfile
import unittest
from pathlib import Path

import chess

from Kit.rules import (BUNDLED_OPENINGS, OpeningBook, StandardReferee, TablebaseOracle,
                                classify, parse_line)


def play(*sans, fen=chess.STARTING_FEN):
    b = chess.Board(fen)
    for s in sans:
        b.push_san(s)
    return b


class TestReferee(unittest.TestCase):
    def test_ongoing(self):
        self.assertIsNone(classify(chess.Board()))

    def test_checkmate(self):
        v = classify(play("f3", "e5", "g4", "Qh4#"))
        self.assertEqual((v.result, v.termination, v.winner), ("0-1", "checkmate", chess.BLACK))
        self.assertEqual(v.white_score(), 0.0)

    def test_stalemate(self):
        v = classify(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
        self.assertEqual((v.result, v.termination), ("1/2-1/2", "stalemate"))

    def test_insufficient(self):
        self.assertEqual(classify(chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")).termination,
                         "insufficient_material")

    def test_fifty_move_claim(self):
        v = classify(chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 100 80"))
        self.assertEqual(v.termination, "fifty_move")

    def test_threefold_claimed_one_ply_early(self):
        """claim_draw 语义：下一步即可形成三次重复时就判和（S 的口径，比严格判定早一 ply）。"""
        b = play("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1")
        self.assertFalse(b.is_repetition(3))
        self.assertEqual(classify(b).termination, "threefold")

    def test_truncation(self):
        ref = StandardReferee(max_plies=4)
        b = play("e4", "e5", "Nf3")
        self.assertIsNone(ref.verdict(b))
        b.push_san("Nc6")
        v = ref.verdict(b)
        self.assertTrue(v.truncated)
        self.assertEqual(v.white_score(), 0.5)

    def test_rules_beat_truncation(self):
        v = StandardReferee(max_plies=4).verdict(play("f3", "e5", "g4", "Qh4#"))
        self.assertEqual(v.termination, "checkmate")


class TestOpenings(unittest.TestCase):
    def write(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "o.txt"
        p.write_text(text, encoding="utf-8")
        return p

    def test_uci_and_san_mixed(self):
        self.assertEqual(parse_line("e4 e7e5 Nf3"), ("e2e4", "e7e5", "g1f3"))

    def test_illegal_line(self):
        with self.assertRaises(ValueError):
            parse_line("e2e4 e2e4")
        with self.assertRaises(ValueError):
            parse_line("f3 e5 g4 Qh4#")          # 开局走完对局已结束

    def test_file_comments_dedupe_strict(self):
        p = self.write("# 注释\ne2e4 e7e5  # 开放\n\ne4 e5\nd4 d5\r\n")
        book = OpeningBook.from_file(p)
        self.assertEqual(book.lines, [("e2e4", "e7e5"), ("d2d4", "d7d5")])
        bad = self.write("e4 e5\ne4 e4\n")
        with self.assertRaises(ValueError):
            OpeningBook.from_file(bad)
        lenient = OpeningBook.from_file(bad, strict=False)
        self.assertEqual((len(lenient), lenient.skipped), (1, [2]))

    def test_dedupe_after_trim(self):
        p = self.write("e4 e5 Nf3\ne4 e5 Bc4\n")
        self.assertEqual(len(OpeningBook.from_file(p)), 2)
        self.assertEqual(len(OpeningBook.from_file(p, max_plies=2)), 1)

    def test_bundled_openings_legal_and_distinct(self):
        book = OpeningBook.bundled()
        self.assertTrue(BUNDLED_OPENINGS.exists())
        self.assertGreaterEqual(len(book), 32)
        self.assertEqual(book.skipped, [])

    def test_plan_is_seeded_permutation(self):
        book = OpeningBook([("e2e4",), ("d2d4",), ("c2c4",), ("g1f3",)])
        plan = book.plan(4, seed=3)
        self.assertEqual(plan, book.plan(4, seed=3))
        self.assertEqual(sorted(i for i, _ in plan), [0, 1, 2, 3])     # 不超过库规模时互不重复
        self.assertNotEqual([i for i, _ in plan], [i for i, _ in book.plan(4, seed=4)])
        long = book.plan(10, seed=3)
        self.assertEqual(len(long), 10)
        self.assertEqual(long[4:8], plan)                               # 超过库规模时循环复用

    def test_plan_matches_s_algorithm(self):
        """与 S tools/ssm_gumbel_arena.py:opening_plan 同算法。"""
        import numpy as np
        lines = [(m,) for m in ("e2e4", "d2d4", "c2c4", "g1f3", "b1c3")]
        idx = np.random.default_rng(11).permutation(5)
        expect = [(int(idx[i % 5]), lines[int(idx[i % 5])]) for i in range(7)]
        self.assertEqual(OpeningBook(lines).plan(7, 11), expect)


class FakeTablebase:
    """按「刚走的那步」查表：table[uci] = (对手视角 wdl, dtz)，其余走法用 default。"""

    def __init__(self, table, default=(0, 0), root=None):
        self.table, self.default, self.root = table, default, root

    def probe_wdl(self, board):
        if not board.move_stack:
            return self.root[0]
        return self.table.get(board.peek().uci(), self.default)[0]

    def probe_dtz(self, board):
        if not board.move_stack:
            return self.root[1]
        return self.table.get(board.peek().uci(), self.default)[1]


class TestTablebase(unittest.TestCase):
    def test_prefers_opponent_loss(self):
        b = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        tb = TablebaseOracle(FakeTablebase({"a1a2": (-2, 7)}))
        self.assertEqual(tb.best_move(b).uci(), "a1a2")

    def test_prefers_zeroing_among_wins(self):
        """结果相同优先推兵：兵残局 DTZ 不提供梯度，否则王原地打转到 50 步。"""
        b = chess.Board("8/8/8/4k3/8/8/P7/4K3 w - - 10 50")
        tb = TablebaseOracle(FakeTablebase({"a2a3": (-2, 20), "e1d1": (-2, 3)}))
        self.assertEqual(tb.best_move(b).uci(), "a2a3")

    def test_fifty_move_downgrade(self):
        """半步钟 + DTZ >= 100 的「必胜」兑现不了，按和棋排序（R 修过、T 缺失的漂移 bug）。"""
        b = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 99 80")
        tb = TablebaseOracle(FakeTablebase({"a1a5": (-2, 5), "a1a2": (0, 0)}, default=(0, 10)))
        self.assertEqual(tb.best_move(b).uci(), "a1a2")
        b2 = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 10 80")
        self.assertEqual(tb.best_move(b2).uci(), "a1a5")

    def test_exact_value_fifty_guard(self):
        b = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 90 80")
        self.assertIsNone(TablebaseOracle(FakeTablebase({}, root=(2, 15))).exact_value(b))
        self.assertEqual(TablebaseOracle(FakeTablebase({}, root=(2, 5))).exact_value(b), 1.0)
        self.assertEqual(TablebaseOracle(FakeTablebase({}, root=(-2, 5))).exact_value(b), -1.0)
        self.assertEqual(TablebaseOracle(FakeTablebase({}, root=(1, 5))).exact_value(b), 0.0)

    def test_not_applicable(self):
        tb = TablebaseOracle(FakeTablebase({}, root=(2, 1)))
        self.assertIsNone(tb.best_move(chess.Board()))           # 子力太多
        self.assertIsNone(tb.exact_value(chess.Board()))

    def test_probe_failure_returns_none(self):
        class Broken:
            def probe_wdl(self, board):
                raise KeyError("missing table")
            probe_dtz = probe_wdl
        tb = TablebaseOracle(Broken())
        b = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        self.assertIsNone(tb.best_move(b))
        self.assertIsNone(tb.exact_value(b))

    def test_open_missing_path(self):
        self.assertIsNone(TablebaseOracle.open(None))
        self.assertIsNone(TablebaseOracle.open("/no/such/dir"))


if __name__ == "__main__":
    unittest.main()
