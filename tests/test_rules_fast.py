"""P4：rules.fast 与 python-chess 原函数逐局面对照。"""
from __future__ import annotations

import random
import unittest

import chess

from Kit.rules import fast


def _positions(seed: int, n_games: int = 40, max_plies: int = 160):
    """随机对局 + 偏向来回走子（制造大量重复局面），逐局面产出。"""
    rng = random.Random(seed)
    for g in range(n_games):
        board = chess.Board()
        shuffle = g % 2 == 1
        for _ in range(max_plies):
            yield board
            if board.is_game_over(claim_draw=False):
                break
            moves = list(board.legal_moves)
            if shuffle and len(board.move_stack) >= 2 and rng.random() < 0.8:
                # 撤回本方上一步（制造来回走子的重复局面）
                last = board.move_stack[-2]
                back = chess.Move(last.to_square, last.from_square)
                if back in moves and not board.is_zeroing(back):
                    board.push(back)
                    continue
            board.push(rng.choice(moves))


class FastOutcomeTest(unittest.TestCase):
    def _check(self, board):
        for claim in (False, True):
            self.assertEqual(fast.outcome(board, claim), board.outcome(claim_draw=claim),
                             (board.fen(), claim))
        if not fast.may_claim_threefold(board):
            self.assertFalse(board.can_claim_threefold_repetition(), board.fen())

    def test_random_and_shuffling_games(self):
        n = threefold = 0
        for board in _positions(1):
            self._check(board)
            n += 1
            if board.can_claim_threefold_repetition():
                threefold += 1
        self.assertGreater(n, 3000)
        self.assertGreater(threefold, 20)       # 确实覆盖了申和分支

    def test_fen_start_and_fifty(self):
        # FEN 起步（halfmove_clock 大于着法栈）、五十步边界
        board = chess.Board("8/8/4k3/8/8/3NK3/8/7R w - - 98 80")
        for mv in ["h1h2", "e6f7", "h2h1", "f7e6", "h1h2", "e6f7", "h2h1", "f7e6"]:
            self._check(board)
            board.push_uci(mv)
        self._check(board)
        self.assertEqual(fast.outcome(board, True), board.outcome(claim_draw=True))

    def test_copy_board_equivalent(self):
        for i, board in enumerate(_positions(2, n_games=4)):
            if i % 7:
                continue
            a, b = board.copy(stack=True), fast.copy_board(board)
            self.assertEqual(a.fen(), b.fen())
            self.assertEqual(a.move_stack, b.move_stack)
            for mv in list(board.legal_moves)[:3]:
                a.push(mv)
                b.push(mv)
                self.assertEqual(a.fen(), b.fen())
                self.assertEqual(a.outcome(claim_draw=True), b.outcome(claim_draw=True))
                a.pop()
                b.pop()
            self.assertEqual(board.fen(), b.fen())   # 改副本不影响原局面


if __name__ == "__main__":
    unittest.main()
