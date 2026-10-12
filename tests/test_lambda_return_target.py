"""λ-return 价值目标（``value_from="wdl_g"``）的回归测试。

钉住三条口径：
1. **默认路径逐位不变**：``value_from="z"``（默认）与改动前完全一致——老分片、
   老配置的训练目标不许变；
2. **λ=1 预算的 g 与硬标签 MC 逐位一致**：g=±1 时最大熵投影精确 one-hot；
3. **g=0 的保留位 = "没预算过"**：``value_from="wdl_g"`` 读到 0 必须响亮报错，
   不静默回退 z（否则目标口径会在两种分片之间漂移）。
"""
from __future__ import annotations

import unittest

import numpy as np

from Kit.planes19 import records as R
from Kit.planes19.records import (SELFPLAY_DTYPE, decode_targets, max_entropy_wdl,
                                  wdl_g_from_records)


def _sp_record(*, z: int, q: float, prob: float = 0.5, mv: int = 8) -> np.ndarray:
    """造一条最小自对弈记录（visit_move/prob 各一个非零项）。"""
    rec = np.zeros(1, dtype=SELFPLAY_DTYPE)[0]
    rec["visit_move"][0] = mv
    rec["visit_prob"][0] = int(round(prob * R.SCALE))
    rec["z"] = z
    rec["q"] = q
    return rec


def _pack(g: float) -> np.ndarray:
    u = int(round((g + 1.0) / 2.0 * 65534.0)) + 1
    return np.array([[[u, 0, 0]]], dtype=np.uint16)


class TestMaxEntropyWdl(unittest.TestCase):
    def test_projection_is_normalized_and_one_hot_at_extremes(self):
        for g in (1.0, 0.0, -1.0, 0.37, -0.37, 0.999):
            pW, pD, pL = max_entropy_wdl(np.array([g]))[0]
            self.assertAlmostEqual(float(pW + pD + pL), 1.0, places=6)
            self.assertGreaterEqual(float(pD), -1e-7)
        self.assertEqual(tuple(max_entropy_wdl(np.array([1.0]))[0]), (1.0, 0.0, 0.0))
        self.assertEqual(tuple(max_entropy_wdl(np.array([0.0]))[0]), (0.0, 1.0, 0.0))
        self.assertEqual(tuple(max_entropy_wdl(np.array([-1.0]))[0]), (0.0, 0.0, 1.0))

    def test_projection_matches_z_one_hot_at_extremes(self):
        """|g|=1 时投影与 onehot(z) 逐位一致（λ=1 与硬标签 MC 对拍的桥梁）。"""
        for z in (1, 0, -1):
            g = float(z)
            pW, pD, pL = max_entropy_wdl(np.array([g]))[0]
            self.assertEqual((pW, pD, pL), (float(z == 1), float(z == 0), float(z == -1)))


class TestValueFromWdlG(unittest.TestCase):
    def test_default_path_unchanged(self):
        """value_from 缺省 = 'z'：与历史行为逐位一致。"""
        recs = np.array([_sp_record(z=1, q=0.3), _sp_record(z=0, q=-0.2),
                         _sp_record(z=-1, q=0.9)])
        p, pr, w = decode_targets(recs)
        self.assertEqual(w.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(w.dtype, np.float32)

    def test_q_ratio_path_unchanged(self):
        """q_ratio 混合路径与改动前一致：0.5·onehot(z) + 0.5·(q 折成的 W/L 二元)。"""
        recs = np.array([_sp_record(z=1, q=0.6)])
        _, _, w = decode_targets(recs, q_ratio=0.5)
        np.testing.assert_allclose(w, [[0.9, 0.0, 0.1]], atol=1e-6)

    def test_lambda1_g_matches_mc_one_hot(self):
        """λ=1 预算的 g ⇒ 与 z 硬标签同一分布（SPRT 前必须成立的不变式）。

        |g|=1 时逐位一致；g=0（和棋）受 u16 量化限制有 3.05e-5 的偏差——
        对训练目标无实质影响，在对拍笔记里记明。
        """
        for z in (1, 0, -1):
            recs = np.array([_sp_record(z=z, q=0.42)])
            recs["wdl"] = _pack(float(z))
            _, _, w = decode_targets(recs, value_from="wdl_g")
            if z == 0:
                np.testing.assert_allclose(w, decode_targets(recs)[2], atol=1e-4)
            else:
                np.testing.assert_array_equal(w, decode_targets(recs)[2])

    def test_intermediate_g_uses_max_entropy(self):
        recs = np.array([_sp_record(z=1, q=0.42)])
        recs["wdl"] = _pack(0.5)
        _, _, w = decode_targets(recs, value_from="wdl_g")
        np.testing.assert_allclose(w, [[0.5, 0.5, 0.0]], atol=1e-4)

    def test_zero_means_not_precomputed_and_raises(self):
        """保留位 0 = 没预算过：必须报错，不静默回退 z。"""
        recs = np.array([_sp_record(z=1, q=0.42)])       # wdl 恒 0
        with self.assertRaises(ValueError) as cm:
            decode_targets(recs, value_from="wdl_g")
        self.assertIn("没预算", str(cm.exception))

    def test_wdl_g_roundtrip(self):
        for g in (1.0, -1.0, 0.0, 0.123456, -0.987654):
            recs = np.zeros(1, dtype=SELFPLAY_DTYPE)
            recs["wdl"] = _pack(g)
            back = float(wdl_g_from_records(recs)[0])
            self.assertAlmostEqual(back, g, places=4,
                                   msg=f"g={g} 往返误差 {abs(back - g)}")

    def test_supervised_records_reject_wdl_g(self):
        rec = np.zeros(1, dtype=R.RECORD_DTYPE)[0]
        with self.assertRaises(ValueError):
            decode_targets(np.array([rec]), value_from="wdl_g")

    def test_unknown_value_from_rejected(self):
        recs = np.array([_sp_record(z=1, q=0.1)])
        with self.assertRaises(ValueError):
            decode_targets(recs, value_from="bogus")


class TestDecoderPlumbing(unittest.TestCase):
    def test_decoder_passes_value_from(self):
        from Kit.planes19.dataset import Decoder

        d = Decoder(value_from="wdl_g")
        self.assertEqual(d.value_from, "wdl_g")
        self.assertEqual(Decoder().value_from, "z")


if __name__ == "__main__":
    unittest.main()
