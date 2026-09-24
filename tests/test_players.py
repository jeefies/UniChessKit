import unittest
from collections import namedtuple

import chess

from Kit.api import GameStart, SearchBudget
from Kit.planes19 import Planes19Expander
from Kit.players import RandomPlayer, SearchPlayer
from Kit.rules import TablebaseOracle
from Kit.runtime import run_sync
from Kit.search import PUCTConfig
from Kit.testing import FakePlanesEvaluator, PlayerContract
from Kit.tests.test_rules import FakeTablebase


def search_player(sims=24, **kw):
    return SearchPlayer("s", Planes19Expander(FakePlanesEvaluator()), simulations=sims,
                        puct=PUCTConfig(batch_size=8), **kw)


class TestRandomContract(PlayerContract, unittest.TestCase):
    def make_player(self):
        return RandomPlayer()


class TestSearchContract(PlayerContract, unittest.TestCase):
    def make_player(self):
        return search_player()


class TestPolicyContract(PlayerContract, unittest.TestCase):
    def make_player(self):
        return search_player(sims=0)


class TestSampledContract(PlayerContract, unittest.TestCase):
    def make_player(self):
        return search_player(sims=16, temperature=1.0)


Entry = namedtuple("Entry", "move weight")


class FakeBook:
    def __init__(self, moves):
        self.moves = moves

    def find_all(self, board):
        return [Entry(chess.Move.from_uci(m), w) for m, w in self.moves.get(board.epd(), [])]


def choose(player, board, seed=0, budget=SearchBudget()):
    run_sync(player.new_game(GameStart(color=board.turn, seed=seed)))
    return run_sync(player.choose(board.copy(), budget))


class TestSearchPlayer(unittest.TestCase):
    def test_chain_tablebase_first(self):
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        p = search_player(oracle=TablebaseOracle(FakeTablebase({"a1a2": (-2, 7)})))
        d = choose(p, board)
        self.assertEqual((d.move.uci(), d.source), ("a1a2", "tablebase"))

    def test_book_within_plies(self):
        start = chess.Board()
        book = FakeBook({start.epd(): [("d2d4", 1)]})
        p = search_player(book=book, book_plies=2)
        d = choose(p, start)
        self.assertEqual((d.move.uci(), d.source), ("d2d4", "book"))
        late = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 5")
        self.assertEqual(choose(p, late).source, "search")          # 超过 book_plies 不查书

    def test_book_weighted_sampling_seeded(self):
        start = chess.Board()
        book = FakeBook({start.epd(): [("d2d4", 5), ("e2e4", 5)]})
        picks = {choose(search_player(book=book), start, seed=s).move.uci() for s in range(12)}
        self.assertEqual(picks, {"d2d4", "e2e4"})
        self.assertEqual(choose(search_player(book=book), start, seed=4).move,
                         choose(search_player(book=book), start, seed=4).move)

    def test_budget_overrides(self):
        p = search_player(sims=0)
        self.assertEqual(choose(p, chess.Board()).source, "policy")
        d = choose(p, chess.Board(), budget=SearchBudget(simulations=20))
        self.assertEqual(d.source, "search")
        self.assertEqual(d.info["sims"], 20)

    def test_tree_reuse_across_moves(self):
        p = search_player(sims=64)
        board = chess.Board()
        run_sync(p.new_game(GameStart(color=chess.WHITE)))
        d1 = run_sync(p.choose(board.copy(), SearchBudget()))
        self.assertFalse(d1.info["reused"])
        board.push(d1.move)
        board.push(next(iter(board.legal_moves)))
        d2 = run_sync(p.choose(board.copy(), SearchBudget()))
        self.assertTrue(d2.info["reused"])

    def test_tree_not_reused_for_other_game(self):
        """树只能接到同一条棋路上（EPD 核对）；换一盘棋必须从零建树。"""
        p = search_player(sims=64)
        a = chess.Board()
        run_sync(p.new_game(GameStart(color=chess.WHITE)))
        d = run_sync(p.choose(a.copy(), SearchBudget()))
        other = chess.Board()
        other.push_san("d4")
        other.push_san("d5")                     # 同样是 2 ply 后轮白走，但不是同一棋路
        d2 = run_sync(p.choose(other.copy(), SearchBudget()))
        self.assertFalse(d2.info["reused"])
        self.assertIn(d2.move, other.legal_moves)

    def test_new_game_resets_tree_and_rng(self):
        p = search_player(sims=32, temperature=1.0)
        first = [choose(p, chess.Board(), seed=5).move for _ in range(3)]
        self.assertEqual(len(set(first)), 1)       # 每次 new_game 都按同一 seed 重置


if __name__ == "__main__":
    unittest.main()
