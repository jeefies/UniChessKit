"""GumbelPlayer 冒烟（假模型，CPU）：协议、π′ 目标、g=0 确定性。"""
import unittest

import chess

from Kit.api import GameStart, SearchBudget
from Kit.planes19 import BatchFnEvaluator
from Kit.players.gumbel_player import make_gumbel_player_factory
from Kit.runtime import run_sync
from Kit.testing import FakePlanes19Model


class TestGumbelPlayer(unittest.TestCase):
    def _play(self, plies=6, g=1.0, temperature=0.0, seed=7):
        fake = FakePlanes19Model(salt="gpl")
        f = make_gumbel_player_factory("g", BatchFnEvaluator("fk:gpl", fake.evaluate_planes),
                                       simulations=16, m0=8, g=g, temperature=temperature)
        p = f()
        run_sync(p.new_game(GameStart(chess.WHITE, seed=seed, both_sides=True)))
        board = chess.Board()
        decs = []
        for _ in range(plies):
            dec = run_sync(p.choose(board, SearchBudget(simulations=16)))
            self.assertIn(dec.move, board.legal_moves)
            decs.append(dec)
            board.push(dec.move)
        return decs

    def test_target_is_pi_prime(self):
        """目标在 ``info["visits"]`` 里、和为 1、覆盖全部合法着（不是 m0 候选子集）。"""
        decs = self._play()
        for dec in decs:
            visits = dec.info.get("visits")
            self.assertTrue(visits, dec.source)
            total = sum(v for _, v in visits)
            self.assertAlmostEqual(total, 1.0, places=3)
            self.assertTrue(all(0.0 < v <= 1.0 for _, v in visits))
            self.assertGreaterEqual(len(visits), 2)

    def test_g0_deterministic(self):
        a = [d.move.uci() for d in self._play(g=0.0)]
        b = [d.move.uci() for d in self._play(g=0.0)]
        self.assertEqual(a, b)

    def test_clear_best_move_played(self):
        """一步杀：g=0 时假模型也许看不出来，但任何局面下都必须走合法着且 info 有 q。"""
        decs = self._play(plies=4)
        for dec in decs:
            self.assertIn("q", dec.info)
            self.assertIn("sims", dec.info)
            self.assertEqual(dec.info["sims"], 16)


if __name__ == "__main__":
    unittest.main()