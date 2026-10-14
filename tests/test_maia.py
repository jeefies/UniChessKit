"""Maia（lc0 net.proto 权重）移植的单测：不依赖权重文件与 torch 的部分。

覆盖三条最容易错又最关键的路径：
1. 1858 policy 索引表与 lc0 ``kIdxToMove`` 一致，且查表往返（含易位/升变）正确
2. 73 个卷积 policy 平面由着法表算法化推导，与 lc0 ``kConvPolicyMap`` 语义一致
3. 112 输入平面的行棋方定向：每个历史局面都按"该局面自己的行棋方在底部"编码
   （lc0 encoder.cc 的 flip 每步交替，等价于按 b.turn 定向）

权重要下载（``Server/tools/fetch_maia.py``），所以加载/前向的测试放在需要权重的
集成脚本里，这里只测纯数据与编码逻辑。
"""
import unittest

import chess

from Kit.maia import (AUX_PLANE_BASE, MOVES, MOVE_TO_IDX, NUM_POLICY,
                      encode_input, index_to_move, move_to_index,
                      _CONV_MAP, _PIECE_ORDER)


def _replay(sans):
    b = chess.Board()
    h = [b.copy()]
    for s in sans:
        b.push_san(s)
        h.append(b.copy())
    return b, h


class TestMoveTable(unittest.TestCase):
    def test_size_and_uniqueness(self):
        self.assertEqual(len(MOVES), NUM_POLICY)
        self.assertEqual(len(MOVE_TO_IDX), NUM_POLICY)

    def test_promotions_present(self):
        promo = [m for m in MOVES if len(m) == 5]
        # lc0 表只有 q/r/b 升变（马升变是已知缺口）
        self.assertEqual(len(promo), 66)
        self.assertEqual(sorted({m[4] for m in promo}), ['b', 'q', 'r'])

    def test_round_trip_normal(self):
        b, _ = _replay(['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6'])
        n = 0
        for mv in b.legal_moves:
            idx = move_to_index(mv, b)
            self.assertIsNotNone(idx, f'{mv.uci()} 不在表内')
            self.assertEqual(index_to_move(idx, b), mv)
            n += 1
        self.assertGreater(n, 20)

    def test_round_trip_castling_both_sides(self):
        for fen, uci in (('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
                         ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1c1'),
                         ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8g8'),
                         ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8c8')):
            b = chess.Board(fen)
            mv = chess.Move.from_uci(uci)
            idx = move_to_index(mv, b)
            self.assertIn(idx, (97, 103), f'{uci} 的索引应为王车易位那两个')
            back = index_to_move(idx, b)
            self.assertEqual(back.uci(), uci)

    def test_round_trip_promotion(self):
        b = chess.Board('8/P7/8/8/8/8/8/k6K w - - 0 1')
        for uci in ('a7a8q', 'a7a8r', 'a7a8b'):
            mv = chess.Move.from_uci(uci)
            idx = move_to_index(mv, b)
            self.assertIsNotNone(idx)
            self.assertEqual(index_to_move(idx, b).uci(), uci)

    def test_knight_promotion_absent(self):
        # lc0 的 kIdxToMove 没有马升变，必须返回 None 而不是猜一个
        b = chess.Board('8/P7/8/8/8/8/8/k6K w - - 0 1')
        mv = chess.Move.from_uci('a7a8n')
        self.assertIsNone(move_to_index(mv, b))

    def test_castling_encoding_is_rook_capture(self):
        # lc0 用 e1h1/e1a1（王吃自家车）表示易位；e1g1 也在表里但那是普通王着法
        self.assertIn('e1h1', MOVE_TO_IDX)
        self.assertIn('e1a1', MOVE_TO_IDX)
        # 王一步走不到 h1/a1，所以这两个串只可能是易位
        self.assertEqual(MOVE_TO_IDX['e1h1'], 103)
        self.assertEqual(MOVE_TO_IDX['e1a1'], 97)


class TestConvPolicyMap(unittest.TestCase):
    def test_coverage(self):
        valid = [i for i, v in enumerate(_CONV_MAP) if v >= 0]
        self.assertEqual(len(valid), NUM_POLICY)
        # 每个有效位置恰好一个 policy 索引，且不重复
        idxs = [_CONV_MAP[i] for i in valid]
        self.assertEqual(len(set(idxs)), NUM_POLICY)

    def test_plane_layout(self):
        # 56 个滑动平面 = 8 方向 × 7 距离；56..63 马步；64..72 升变
        for p in range(56):
            n = sum(1 for s in range(64) if _CONV_MAP[p * 64 + s] >= 0)
            self.assertGreater(n, 0)
        # 距离 1 的方向平面覆盖最多格，距离 7 最少
        self.assertGreater(
            sum(1 for s in range(64) if _CONV_MAP[0 * 64 + s] >= 0),
            sum(1 for s in range(64) if _CONV_MAP[6 * 64 + s] >= 0))

    def test_matches_lc0_semantics(self):
        # 平面 0 = 方向 (0,+1) 距离 1：起始格不能在最上横线
        for s in range(64):
            if s // 8 == 7:
                self.assertLess(_CONV_MAP[0 * 64 + s], 0,
                                '最上横线的格不能向下走一格')
            else:
                self.assertGreaterEqual(_CONV_MAP[0 * 64 + s], 0)

    def test_promotion_planes_only_from_rank7(self):
        # 升变平面只从第 7 横线（索引 6，格 48..55）出发
        for p in range(64, 73):
            for s in range(64):
                v = _CONV_MAP[p * 64 + s]
                if v >= 0:
                    self.assertEqual(s // 8, 6, f'平面 {p} 的起始格应在第 7 横线')


class TestInputEncoding(unittest.TestCase):
    def test_shape_and_dtype(self):
        b = chess.Board()
        p = encode_input(b, [b.copy()])
        self.assertEqual(p.shape, (112, 8, 8))
        self.assertEqual(p.dtype.name, 'float32')

    def test_start_position_planes(self):
        b = chess.Board()
        p = encode_input(b, [b.copy()])
        # 平面 0..5 = 白方 P/N/B/R/Q/K，平面 6..11 = 黑方
        self.assertEqual(int(p[0].sum()), 8)      # 白兵
        self.assertEqual(int(p[1].sum()), 2)      # 白马
        self.assertEqual(int(p[5].sum()), 1)      # 白王
        self.assertEqual(int(p[6].sum()), 8)      # 黑兵
        # 辅助平面
        self.assertEqual(int(p[AUX_PLANE_BASE + 0].sum()), 64)   # 白可长易位
        self.assertEqual(int(p[AUX_PLANE_BASE + 1].sum()), 64)   # 白可短易位
        self.assertEqual(int(p[AUX_PLANE_BASE + 2].sum()), 64)
        self.assertEqual(int(p[AUX_PLANE_BASE + 3].sum()), 64)
        self.assertEqual(int(p[AUX_PLANE_BASE + 4].sum()), 0)    # 非黑方行棋
        self.assertEqual(float(p[AUX_PLANE_BASE + 5].max()), 0.0)  # 50 步计数
        self.assertEqual(int(p[AUX_PLANE_BASE + 6].sum()), 0)    # 恒 0
        self.assertEqual(int(p[AUX_PLANE_BASE + 7].sum()), 64)   # 恒 1

    def test_history_planes_zero_at_start(self):
        b = chess.Board()
        p = encode_input(b, [b.copy()])
        # 只有当前局面的 13 个平面有内容，历史平面（13..103）应全零
        self.assertEqual(float(p[13:104].sum()), 0.0)
        self.assertGreater(float(p[:13].sum()), 0.0)

    def test_we_are_black_plane(self):
        b, _ = _replay(['e4'])
        p = encode_input(b, [chess.Board(), b.copy()])
        self.assertEqual(int(p[AUX_PLANE_BASE + 4].sum()), 64)

    def test_black_to_move_is_rank_flipped(self):
        # 黑方行棋时"我方"（黑）必须在底部。平面行号就是 rank 索引（row 0 = 第 1
        # 横线），黑方行棋要 rank 镜像：rank r -> row 7-r。
        b, _ = _replay(['e4'])
        p = encode_input(b, [chess.Board(), b.copy()])
        black_pawns = p[0]          # 平面 0 = 我方兵 = 黑兵
        self.assertEqual(int(black_pawns.sum()), 8)
        # 1.e4 后黑方 8 只兵都还在第 7 横线（rank 索引 6）-> row 1
        self.assertEqual(int(black_pawns[1].sum()), 8)
        self.assertEqual(int(black_pawns[6].sum()), 0)

    def test_history_uses_each_boards_own_mover(self):
        # 关键不变量：每个历史局面按"该局面自己的行棋方在底部"定向，而不是按当前行棋方。
        # 1.e4 之后：当前局面黑方行棋（要翻转），上一步（开局）白方行棋（不翻转）。
        # 两边平面 0 都是"我方兵"= 黑兵，但落在不同行——这正是定向差异的直接证据。
        b, h = _replay(['e4'])
        p = encode_input(b, h)
        cur_black_pawns = p[0]                 # 当前局面：黑方行棋，翻转
        hist_black_pawns = p[13]               # 上一步（开局）：白方行棋，不翻转
        self.assertEqual(int(cur_black_pawns.sum()), 8)
        self.assertEqual(int(hist_black_pawns.sum()), 8)
        # 黑兵天然在第 7 横线（rank 索引 6）
        self.assertEqual(int(hist_black_pawns[6].sum()), 8)   # 不翻转 -> 留在 row 6
        self.assertEqual(int(cur_black_pawns[1].sum()), 8)    # 翻转 -> 落到 row 1
        self.assertEqual(int(cur_black_pawns[6].sum()), 0)
        self.assertEqual(int(hist_black_pawns[1].sum()), 0)

    def test_repetition_plane(self):
        b, h = _replay(['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3'])
        p = encode_input(b, h)
        # 当前局面（Nf3 后）与两步前同形，重复平面应置 1
        self.assertEqual(float(p[12].max()), 1.0)

    def test_halfmove_clock_plane(self):
        b = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 3')
        p = encode_input(b, [b.copy()])
        self.assertEqual(float(p[AUX_PLANE_BASE + 5].max()), 2.0)

    def test_no_castling_rights(self):
        b = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1')
        p = encode_input(b, [b.copy()])
        for k in range(4):
            self.assertEqual(int(p[AUX_PLANE_BASE + k].sum()), 0)

    def test_history_length_capped(self):
        b, h = _replay(['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'Ng5', 'd5',
                        'exd5', 'Nxd5', 'Nxf7'])
        p = encode_input(b, h)
        # 只有前 8 步历史有平面，更早的没有对应平面可写
        self.assertEqual(p.shape, (112, 8, 8))
        self.assertGreater(float(p[:104].sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
