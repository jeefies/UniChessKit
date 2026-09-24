import unittest

import chess

from Kit.api import GameStart, SearchBudget
from Kit.runtime import run_sync
from Kit.serving import GameEnginePlayer, make_game_engine
from Kit.testing import PlayerContract, make_fake_player_factory
from Kit.testing.fakes import FakeGameEngine

LOADS = []


def fake_factory(**kwargs):
    LOADS.append(kwargs)
    return make_fake_player_factory("fake", simulations=kwargs.get("simulations", 8))


class TestKitGameEngine(unittest.TestCase):
    def setUp(self):
        LOADS.clear()
        self.cls = make_game_engine(fake_factory, kit_factory="x.y:z")

    def test_six_methods_and_shared_model(self):
        self.assertEqual(self.cls.KIT_FACTORY, "x.y:z")
        e1, e2 = self.cls(simulations=8), self.cls(simulations=8)
        self.cls(simulations=4)
        self.assertEqual(len(LOADS), 2)                  # 同一组参数只加载一次
        self.assertIs(e1.player_factory, e2.player_factory)
        self.assertIsNot(e1.player, e2.player)           # 每个会话自己的 Player
        self.assertEqual(e1.state()["fen"], chess.STARTING_FEN)
        e1.human_move("e2e4")
        out = e1.engine_move()
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertIn(chess.Move.from_uci(out["engine_move"]), board.legal_moves)
        self.assertEqual(out["source"], "search")
        self.assertIsInstance(out["eval"], float)
        self.assertEqual(len(e1.board.move_stack), 2)
        e1.undo()
        self.assertEqual(len(e1.board.move_stack), 0)
        self.assertEqual(e1.engine_move()["fen"].split()[1], "b")   # 悔棋后照常出招
        e1.cleanup()
        e1.cleanup()

    def test_custom_fen(self):
        e = self.cls()
        fen = "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"
        self.assertEqual(e.setup(fen)["fen"], fen)
        self.assertIn(chess.Move.from_uci(e.engine_move()["engine_move"]),
                      chess.Board(fen).legal_moves)
        e.undo()
        self.assertEqual(e.state()["fen"], fen)

    def test_claim_draw_counts_as_done(self):
        e = self.cls()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            e.human_move(uci)
        self.assertFalse(e.state()["done"])
        # 下一步 f6g8 可使初始局面第三次出现：可申请和棋即视为终局（claim_draw 语义）
        e.human_move("f3g1")
        self.assertTrue(e.state()["done"])
        self.assertIsNone(e.engine_move()["engine_move"])

    def test_illegal_human_move_rejected(self):
        e = self.cls()
        with self.assertRaises(ValueError):
            e.human_move("e2e5")


class TestGameEnginePlayer(PlayerContract, unittest.TestCase):
    def make_player(self):
        return GameEnginePlayer("wrapped", FakeGameEngine)

    def test_sync_extends_or_resets(self):
        FakeGameEngine.instances.clear()
        p = self.make_player()
        run_sync(p.new_game(GameStart(color=chess.BLACK, opening=("e2e4",))))
        eng = FakeGameEngine.instances[-1]
        board = chess.Board()
        board.push_uci("e2e4")
        d = run_sync(p.choose(board.copy(), SearchBudget()))
        self.assertEqual(d.source, "engine")
        self.assertEqual(d.info, {"eval": 0.25, "nodes": 7})    # numpy 标量转成 Python 数
        board.push(d.move)
        board.push_uci("d2d4")
        run_sync(p.choose(board.copy(), SearchBudget()))
        self.assertIn(("human_move", "d2d4"), eng.calls)
        n_setup = sum(1 for c in eng.calls if c[0] == "setup")
        other = chess.Board()                    # 换成另一盘棋：setup 后整盘重放
        other.push_uci("d2d4")
        run_sync(p.choose(other.copy(), SearchBudget()))
        self.assertEqual(sum(1 for c in eng.calls if c[0] == "setup"), n_setup + 1)
        self.assertEqual(eng.board.move_stack[0].uci(), "d2d4")
        p.close()
        self.assertEqual(eng.cleaned, 1)


if __name__ == "__main__":
    unittest.main()
