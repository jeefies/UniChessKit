"""契约测试（里氏替换的保证）：每个 Player / Expander 实现都要通过。

用法（引擎仓库里）::

    class TestMyPlayer(PlayerContract, unittest.TestCase):
        def make_player(self):
            return my_factory()

驱动用 runtime.run_sync：协程若 yield 了 EvalRequest 以外的东西会直接报错。
"""
from __future__ import annotations

import chess
import numpy as np

from ..api.types import GameStart, Leaf, MoveDecision, SearchBudget
from ..runtime import run_sync

# 覆盖易错局面：初始、升变、被将军、唯一合法着法、吃过路兵、易位、残局
CONTRACT_FENS = (
    chess.STARTING_FEN,
    "8/P7/8/8/8/8/8/k6K w - - 0 1",                                  # 升变
    "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # 被将死（无合法着法，不参测）
    "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",                               # 被将军，唯一着法 Kxe2
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", # 吃过路兵
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",                          # 黑方易位
    "8/8/8/4k3/8/8/2P5/4K3 b - - 0 1",                               # 残局，黑方走
)


def contract_boards():
    boards = []
    for fen in CONTRACT_FENS:
        b = chess.Board(fen)
        if any(b.legal_moves):
            boards.append(b)
    return boards


class PlayerContract:
    deterministic = True     # 同 seed 是否必须走出同样的棋

    def make_player(self):
        raise NotImplementedError

    def _choose(self, player, board, seed=0, color=None):
        color = board.turn if color is None else color
        run_sync(player.new_game(GameStart(color=color, seed=seed)))
        return run_sync(player.choose(board.copy(), SearchBudget()))

    def test_contract_legal_moves(self):
        for board in contract_boards():
            player = self.make_player()
            try:
                d = self._choose(player, board)
                self.assertIsInstance(d, MoveDecision)
                self.assertIn(d.move, board.legal_moves, board.fen())
            finally:
                player.close()

    def test_contract_self_play_with_observe(self):
        white, black = self.make_player(), self.make_player()
        try:
            run_sync(white.new_game(GameStart(color=chess.WHITE, seed=1)))
            run_sync(black.new_game(GameStart(color=chess.BLACK, seed=2)))
            board = chess.Board()
            for _ in range(16):
                if board.is_game_over(claim_draw=True):
                    break
                mover = white if board.turn == chess.WHITE else black
                d = run_sync(mover.choose(board.copy(), SearchBudget()))
                self.assertIn(d.move, board.legal_moves)
                board.push(d.move)
                for p in (white, black):
                    run_sync(p.observe(board.copy(), d.move))
        finally:
            white.close()
            black.close()

    def test_contract_close_idempotent(self):
        player = self.make_player()
        self._choose(player, chess.Board())
        player.close()
        player.close()

    def test_contract_deterministic(self):
        if not self.deterministic:
            self.skipTest("该实现声明为非确定性")
        for board in contract_boards()[:3]:
            p1, p2 = self.make_player(), self.make_player()
            try:
                self.assertEqual(self._choose(p1, board, seed=7).move,
                                 self._choose(p2, board, seed=7).move, board.fen())
            finally:
                p1.close()
                p2.close()

    def test_contract_has_name(self):
        player = self.make_player()
        self.assertTrue(isinstance(player.name, str) and player.name)
        player.close()


class ExpanderContract:
    def make_expander(self):
        raise NotImplementedError

    def test_contract_expand(self):
        exp = self.make_expander()
        boards = contract_boards()
        evals = run_sync(exp.expand([Leaf(board=b) for b in boards]))
        self.assertEqual(len(evals), len(boards))
        for b, ev in zip(boards, evals):
            self.assertEqual(set(ev.moves), set(b.legal_moves), b.fen())
            self.assertEqual(len(ev.moves), len(ev.priors))
            pri = np.asarray(ev.priors, dtype=np.float64)
            self.assertTrue((pri >= 0).all())
            self.assertAlmostEqual(float(pri.sum()), 1.0, places=4)
            self.assertTrue(-1.0 - 1e-6 <= float(ev.value) <= 1.0 + 1e-6)

    def test_contract_batch_equals_single(self):
        exp = self.make_expander()
        boards = contract_boards()
        batch = run_sync(exp.expand([Leaf(board=b) for b in boards]))
        for b, ev in zip(boards, batch):
            (single,) = run_sync(exp.expand([Leaf(board=b)]))
            self.assertEqual(list(single.moves), list(ev.moves))
            np.testing.assert_allclose(np.asarray(single.priors), np.asarray(ev.priors),
                                       rtol=1e-3, atol=1e-5)
            self.assertAlmostEqual(float(single.value), float(ev.value), places=3)

    def test_contract_no_legal_moves(self):
        exp = self.make_expander()
        mated = chess.Board(CONTRACT_FENS[2])
        self.assertFalse(any(mated.legal_moves))
        (ev,) = run_sync(exp.expand([Leaf(board=mated)]))
        self.assertEqual(len(ev.moves), 0)
