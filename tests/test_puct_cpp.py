"""C++ PUCT（search/puct_cpp.py）与 Python PUCT 的逐位对照。

三层：
1. 棋规：C++ 棋盘与 python-chess 在随机对局的每一步上比对合法着法**顺序**、重复判定、
   终局精确值、19 平面编码、FEN；
2. 搜索：同一假模型下整棵树（moves / P / N / W / VL / sum_N / 终局值）与 last_metrics 逐位相同，
   覆盖批大小、碰撞、root_min_visits、claim_draw、噪声、树复用、残局表暂停；
3. 对局：SearchPlayer 用两种实现走完整盘棋，每步着法与 info 相同。
"""
import ctypes
import os
import random
import shutil
import unittest
import zlib

import chess
import numpy as np

from Kit.api import GameStart, SearchBudget
from Kit.planes19 import Planes19Expander, encode, make_search_player_factory
from Kit.players import SearchPlayer
from Kit.rules import TablebaseOracle
from Kit.runtime import run_sync
from Kit.search import PUCT, PUCTConfig
from Kit.search import native
from Kit.search.puct_cpp import PUCTCpp, move_code
from Kit.testing import FakePlanes19Model

if shutil.which(os.environ.get("CXX", "g++")) is None:
    # 只有没有编译器时才跳过；有编译器而编译失败必须报错（不静默跳过）
    raise unittest.SkipTest("没有 C++ 编译器，跳过 C++ PUCT 测试")

START_FENS = (
    chess.STARTING_FEN,
    "r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 3 8",   # 双方可易位
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            # 吃过路兵牵制
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",         # 可吃过路兵
    "8/P7/8/8/8/8/6kp/4K3 w - - 0 1",                                        # 双方升变
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 140 120",                                  # 临近 75 步
    "4k3/8/8/8/8/8/2n5/4K1N1 w - - 0 1",                                     # 子力不足边缘
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",   # 一步杀
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Qk - 0 1",                # 残缺易位权
    "rnbqk2r/pppp1ppp/5n2/4p3/1b2P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 1",
)


def _ctx_for(board):
    p = PUCTCpp(None, PUCTConfig())
    p._load_root(board)
    return p


def _codes(moves):
    return [move_code(m) for m in moves]


class TestBoardParity(unittest.TestCase):
    """C++ 棋规与 python-chess 1.11 逐步一致。"""

    def _check(self, board, tag):
        p = _ctx_for(board)
        L, ctx = native.lib(), p._ctx
        buf = np.zeros(512, dtype=np.uint16)
        n = L.kp_root_legal(ctx, buf.ctypes.data_as(native._u16p), 512)
        self.assertEqual([int(c) for c in buf[:n]], _codes(board.legal_moves), tag)
        for claim in (False, True):
            v = ctypes.c_double()
            reps = (ctypes.c_int * 4)()
            info = (ctypes.c_int * 2)()
            has = L.kp_root_rules(ctx, int(claim), ctypes.byref(v), reps, info)
            ref = PUCT(None, PUCTConfig(claim_draw=claim)).exact_value(board)
            self.assertEqual(None if not has else v.value, ref, f"{tag} claim={claim}")
            self.assertEqual(list(reps), [int(board.is_repetition(k)) for k in (2, 3, 4, 5)], tag)
            self.assertEqual(info[0], int(board.is_check()), tag)
            self.assertEqual(info[1], int(board.has_legal_en_passant()), tag)
        planes = np.zeros((19, 8, 8), dtype=np.float32)
        L.kp_encode_root(ctx, planes.ctypes.data_as(native._f32p))
        np.testing.assert_array_equal(planes, encode(board), err_msg=tag)

    def _random_game(self, fen, seed, plies, shuffle):
        rng = random.Random(seed)
        board = chess.Board(fen)
        for ply in range(plies):
            self._check(board, f"{fen} seed={seed} ply={ply} {board.fen()}")
            legal = list(board.legal_moves)
            if not legal:
                break
            if shuffle and rng.random() < 0.8:
                quiet = [m for m in legal if board.piece_type_at(m.from_square) in
                         (chess.KNIGHT, chess.KING, chess.ROOK) and not board.is_capture(m)]
                legal = quiet or legal
            board.push(rng.choice(legal))

    def test_random_games(self):
        for i, fen in enumerate(START_FENS):
            for seed in range(4):
                self._random_game(fen, seed * 100 + i, 160, shuffle=seed % 2 == 1)

    def test_repetitions_and_irreversible(self):
        b = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 3 + ["e2e4", "g8f6", "g1f3", "f6g8",
                                                          "f3g1", "g8f6"]:
            self._check(b, b.fen())
            b.push_uci(uci)
        self._check(b, b.fen())
        # 易位权丢失是不可逆着法
        b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        for uci in ["e1f1", "e8f8", "f1e1", "f8e8"] * 3:
            self._check(b, b.fen())
            b.push_uci(uci)
        self._check(b, b.fen())

    def test_rejects_illegal_history(self):
        b = chess.Board()
        b.push(chess.Move.from_uci("e2e5"))       # 不检查合法性的 push
        with self.assertRaises(RuntimeError):
            _ctx_for(b)

    def test_pairwise_sum_matches_numpy(self):
        L = native.lib()
        rng = np.random.default_rng(0)
        for _ in range(500):
            n = int(rng.integers(1, 400))
            a = (rng.standard_normal(n) * rng.choice([1e-3, 1, 1e3], n)).astype(np.float32)
            self.assertEqual(L.kp_pairwise_f32(a.ctypes.data_as(native._f32p), n), a.sum())
            d = a.astype(np.float64) * 1.1
            self.assertEqual(L.kp_pairwise_f64(d.ctypes.data_as(native._f64p), n), d.sum())


class HashTablebase:
    """确定性的假 Syzygy：WDL / DTZ 由 FEN（含回合数）哈希决定。"""

    def __init__(self, salt=""):
        self.salt = salt

    def _h(self, board):
        return zlib.crc32((self.salt + board.fen()).encode())

    def probe_wdl(self, board):
        h = self._h(board)
        if h % 7 == 0:
            raise KeyError("缺表")
        return h % 5 - 2

    def probe_dtz(self, board):
        return self._h(board) % 60


def _trees_equal(tc, a, b, path="root", budget=None):
    budget = budget if budget is not None else [20000]
    budget[0] -= 1
    tc.assertGreater(budget[0], 0)
    tc.assertEqual(a.expanded, b.expanded, path)
    tc.assertEqual(a.terminal_value, b.terminal_value, path)
    tc.assertEqual(int(a.sum_N), int(b.sum_N), path)
    tc.assertEqual(list(a.moves), list(b.moves), path)
    for name in ("P", "N", "W", "VL"):
        x, y = getattr(a, name), getattr(b, name)
        tc.assertEqual(x.dtype, y.dtype, f"{path}.{name}")
        tc.assertEqual(x.tobytes(), y.tobytes(), f"{path}.{name}")
    ca, cb = a.children, b.children
    tc.assertEqual(len(ca), len(cb), path)
    for mv, x, y in zip(a.moves, ca, cb):
        tc.assertEqual(x is None, y is None, f"{path}/{mv}")
        if x is not None:
            _trees_equal(tc, x, y, f"{path}/{mv}", budget)


def _pair(model, cfg, oracle=None, seed=0):
    ev_board, ev_planes = model.evaluators()
    py = PUCT(Planes19Expander(ev_board), cfg, oracle=oracle, rng=np.random.default_rng(seed))
    cc = PUCTCpp(ev_planes, cfg, oracle=oracle, rng=np.random.default_rng(seed),
                 expander=Planes19Expander(ev_board))
    return py, cc


def _board_with_history(fen, ucis):
    b = chess.Board(fen)
    for u in ucis:
        b.push_uci(u)
    return b


SEARCH_BOARDS = (
    chess.Board(),
    chess.Board(START_FENS[1]),
    chess.Board(START_FENS[2]),
    chess.Board(START_FENS[3]),
    chess.Board(START_FENS[5]),
    chess.Board(START_FENS[8]),
    chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
    chess.Board("7k/8/8/8/8/8/8/K6Q w - - 90 100"),          # 很快碰到 50 / 75 步
    _board_with_history(chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6",
                                             "f3g1"]),        # 重复局面
)


class TestSearchParity(unittest.TestCase):
    CONFIGS = (
        dict(simulations=200, batch_size=16),
        dict(simulations=160, batch_size=1),
        dict(simulations=300, batch_size=64),
        dict(simulations=200, batch_size=32, max_collision=1),
        dict(simulations=200, batch_size=16, root_min_visits=0, fpu_reduction=0.0),
        dict(simulations=200, batch_size=16, root_min_visits=3, virtual_loss=3.0),
        dict(simulations=200, batch_size=16, claim_draw=True),
        dict(simulations=120, batch_size=8, c_puct_init=0.5, c_puct_base=50.0),
    )

    def _compare(self, board, cfg, model, noise=False, oracle=None, seed=0):
        py, cc = _pair(model, cfg, oracle, seed)
        r1 = run_sync(py.search(board.copy(), add_noise=noise))
        r2 = run_sync(cc.search(board.copy(), add_noise=noise))
        _trees_equal(self, r1, r2)
        self.assertEqual(py.last_metrics, cc.last_metrics)
        return py, cc, r1, r2

    def test_trees_identical(self):
        for quantize in (0, 3):
            model = FakePlanes19Model("s", quantize=quantize)
            for board in SEARCH_BOARDS:
                for kw in self.CONFIGS:
                    with self.subTest(fen=board.fen(), q=quantize, **kw):
                        self._compare(board, PUCTConfig(**kw), model)

    def test_repetition_draw_trees_identical(self):
        """``repetition_draw=True``：走到已出现过的局面按和棋结算，Python 与 C++ 逐位一致。

        三个局面都带重复历史，最后一步把局面走回之前出现过的一次 ⇒ ``is_repetition(2)``。
        两个实现都必须把那条边建成终局（值 0）且不再下探送网络。
        """
        model = FakePlanes19Model("rep")
        cases = (
            (["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"], "f3g1"),
            (["e2e4", "e7e5", "g1f3", "b8c6", "f3g1", "c6b8"], "g1f3"),
            (["d2d4", "d7d5", "c1f4", "c8f5", "f4c1", "f5c8"], "c1f4"),
        )
        for ucis, repeat in cases:
            board = _board_with_history(chess.STARTING_FEN, ucis)
            cand = board.copy(stack=True)
            cand.push_uci(repeat)
            self.assertTrue(cand.is_repetition(2), cand.fen())
            for kw in (dict(simulations=200, batch_size=16),
                       dict(simulations=200, batch_size=1),
                       dict(simulations=200, batch_size=16, claim_draw=True, root_min_visits=3)):
                with self.subTest(moves=tuple(ucis), repeat=repeat, **kw):
                    py, cc, r1, r2 = self._compare(board, PUCTConfig(repetition_draw=True, **kw),
                                                   model)
                    i = [m.uci() for m in r1.moves].index(repeat)
                    self.assertEqual(float(r1.children[i].terminal_value), 0.0)
                    self.assertEqual(float(r2.children[i].terminal_value), 0.0)
                    # 终局子节点没有着法 ⇒ 不会为该局面送网络
                    self.assertEqual(list(r1.children[i].moves), [])
                    self.assertEqual(list(r2.children[i].moves), [])

    def test_noise(self):
        model = FakePlanes19Model("n")
        for seed in range(3):
            for board in SEARCH_BOARDS[:3]:
                self._compare(board, PUCTConfig(simulations=150, batch_size=16), model,
                              noise=True, seed=seed)

    def test_tablebase_pause(self):
        """叶子落入残局表范围时 C++ 暂停、Python 在带走子栈的棋盘上探测、回填。"""
        model = FakePlanes19Model("tb")
        boards = [chess.Board("4k3/8/3n4/3p4/4P3/8/8/R3K3 w Q - 0 1"),
                  chess.Board("4k3/3q4/8/3p4/4P3/2N5/8/4K3 w - - 30 60"),
                  _board_with_history("4k3/3r4/8/3p4/4P3/8/8/R3K3 w - - 0 1",
                                      ["a1a2", "d7d6", "a2a1", "d6d7"])]
        for salt in ("a", "b", "c"):
            for board in boards:
                for kw in (dict(simulations=200, batch_size=16),
                           dict(simulations=100, batch_size=1, claim_draw=True)):
                    oracle = TablebaseOracle(HashTablebase(salt), max_pieces=5)
                    with self.subTest(salt=salt, fen=board.fen(), **kw):
                        self._compare(board, PUCTConfig(**kw), model, oracle=oracle)

    def test_reuse_over_game(self):
        """best_move → advance_root → 复用子树再搜，逐步整树一致。"""
        model = FakePlanes19Model("r")
        cfg = PUCTConfig(simulations=120, batch_size=16, temperature=1.0)
        py, cc = _pair(model, cfg, seed=7)
        board = chess.Board()
        r1 = r2 = None
        for ply in range(24):
            mv1, t1 = run_sync(py.best_move(board.copy(), root=r1, add_noise=True))
            mv2, t2 = run_sync(cc.best_move(board.copy(), root=r2, add_noise=True))
            self.assertEqual(mv1, mv2, ply)
            _trees_equal(self, t1, t2, f"ply{ply}")
            self.assertEqual(py.last_metrics, cc.last_metrics, ply)
            self.assertEqual(PUCT.principal_variation(t1), cc.principal_variation(t2))
            board.push(mv1)
            r1, r2 = PUCT.advance_root(t1, mv1), cc.advance_root(t2, mv2)
            self.assertEqual(r1 is None, r2 is None)
            if board.is_game_over():
                break

    def test_terminal_root_and_no_moves(self):
        model = FakePlanes19Model()
        for fen in ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1"):
            py, cc = _pair(model, PUCTConfig(simulations=50))
            r1 = run_sync(py.search(chess.Board(fen)))
            r2 = run_sync(cc.search(chess.Board(fen)))
            self.assertEqual(r1.terminal_value, r2.terminal_value)
            self.assertTrue(r2.expanded)
            self.assertEqual(r2.moves, [])

    def test_node_handles_keep_subtree_alive(self):
        model = FakePlanes19Model()
        _, cc = _pair(model, PUCTConfig(simulations=100, batch_size=8))
        root = run_sync(cc.search(chess.Board()))
        mv = root.moves[int(np.argmax(root.N))]
        child = cc.advance_root(root, mv)
        n_before = int(child.N.sum())
        del root                         # 父节点释放后子树仍在
        self.assertEqual(int(child.N.sum()), n_before)
        self.assertEqual(float(child.VL.sum()), 0.0)


class TestPlayerParity(unittest.TestCase):
    def _play(self, impl, temperature, noise, syzygy=False):
        model = FakePlanes19Model("g")
        ev_board, ev_planes = model.evaluators()
        cfg = PUCTConfig(simulations=64, batch_size=16)
        oracle = TablebaseOracle(HashTablebase("g"), max_pieces=5) if syzygy else None
        player = SearchPlayer("p", Planes19Expander(ev_board), simulations=64, puct=cfg,
                              oracle=oracle, temperature=temperature,
                              planes_evaluator=ev_planes if impl == "cpp" else None)
        run_sync(player.new_game(GameStart(color=chess.WHITE, seed=11)))
        board = chess.Board("r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 3 8")
        record = []
        for _ in range(60):
            if board.is_game_over(claim_draw=True):
                break
            d = run_sync(player.choose(board, SearchBudget(add_noise=noise)))
            record.append((d.move.uci(), d.source, d.info))
            board.push(d.move)
        return record

    def test_same_game(self):
        for temperature, noise in ((0.0, False), (1.0, True)):
            self.assertEqual(self._play("python", temperature, noise),
                             self._play("cpp", temperature, noise))


class TestFactory(unittest.TestCase):
    def test_search_impl_selection(self):
        model = FakePlanes19Model()
        ev_board, ev_planes = model.evaluators()
        f = make_search_player_factory("x", ev_board, simulations=8, planes_evaluator=ev_planes)
        self.assertEqual(f.search_impl, "cpp")
        self.assertIsInstance(f().search, PUCTCpp)
        f = make_search_player_factory("x", ev_board, simulations=8, planes_evaluator=ev_planes,
                                       search_impl="python")
        self.assertEqual(f.search_impl, "python")
        self.assertNotIsInstance(f().search, PUCTCpp)
        self.assertEqual(make_search_player_factory("x", ev_board).search_impl, "python")
        with self.assertRaises(ValueError):
            make_search_player_factory("x", ev_board, search_impl="cpp")
        with self.assertRaises(ValueError):
            make_search_player_factory("x", ev_board, search_impl="gpu")

    def test_build_failure_is_loud(self):
        """编译器不可用 / 编译失败要报错，不能静默回退 Python（T 5f81219 的教训）。"""
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"CXX": "/nonexistent/g++",
                                              "UNICHESS_KIT_NATIVE_CACHE": tmp}):
                with self.assertRaises(native.NativeBuildError):
                    native.build(force=True)


if __name__ == "__main__":
    unittest.main()
