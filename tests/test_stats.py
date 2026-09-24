import math
import unittest

from Kit import stats
from Kit.tests import golden


class TestElo(unittest.TestCase):
    def test_even(self):
        elo, lo, hi = stats.elo_with_error(10, 10, 10)
        self.assertAlmostEqual(elo, 0.0)
        self.assertAlmostEqual(lo, -hi, places=6)

    def test_all_win_is_inf(self):
        elo, lo, hi = stats.elo_with_error(10, 0, 0)
        self.assertEqual((elo, hi), (math.inf, math.inf))
        self.assertEqual(stats.elo_with_error(0, 0, 5)[0], -math.inf)
        self.assertEqual(stats.elo_with_error(0, 0, 0), (0.0, -math.inf, math.inf))

    def test_roundtrip(self):
        for e in (-300, -50, 0, 35, 400):
            self.assertAlmostEqual(stats.score_to_elo(stats.elo_to_score(e)), e, places=6)

    def test_los(self):
        self.assertEqual(stats.los(0, 5, 0), 0.5)
        self.assertGreater(stats.los(12, 0, 2), 0.99)
        self.assertAlmostEqual(stats.los(3, 0, 7) + stats.los(7, 0, 3), 1.0)

    def test_confidence_table(self):
        with self.assertRaises(ValueError):
            stats.elo_with_error(1, 1, 1, confidence=0.5)


class TestSprt(unittest.TestCase):
    def test_bounds_and_verdict(self):
        lo, hi = stats.sprt_bounds(0.05, 0.05)
        self.assertAlmostEqual(hi, math.log(19))
        self.assertAlmostEqual(lo, -math.log(19))
        self.assertEqual(stats.sprt_verdict(3.0), "H1")
        self.assertEqual(stats.sprt_verdict(-3.0), "H0")
        self.assertIsNone(stats.sprt_verdict(0.1))

    def test_direction(self):
        self.assertGreater(stats.sprt_llr(60, 20, 20, 0, 10), 0)
        self.assertLess(stats.sprt_llr(20, 20, 60, 0, 10), 0)

    def test_zero_variance_regularized(self):
        """全胜时方差为 0；正则化后 LLR 为有限正数（否则永远不会停）。"""
        llr = stats.sprt_llr(20, 0, 0, 0, 10)
        self.assertTrue(0 < llr < math.inf)
        self.assertEqual(stats.sprt_verdict(llr), "H1")
        self.assertEqual(stats.sprt_verdict(stats.sprt_llr_pentanomial([0, 0, 0, 0, 20], 0, 10)),
                         "H1")

    def test_matches_r_when_no_empty_bin(self):
        """与旧 R ``eval/arena.py`` 的冻结结果（tests/fixtures/r_stats）一致。"""
        for row in golden.load("r_stats"):
            w, d, l = row["wdl"]
            self.assertAlmostEqual(stats.sprt_llr(w, d, l, 0, 10), float.fromhex(row["sprt_llr"]),
                                   places=9)
            for a, b in zip(stats.elo_with_error(w, d, l), row["elo_with_error"]):
                self.assertAlmostEqual(a, float.fromhex(b), places=9)
            self.assertAlmostEqual(stats.los(w, d, l), float.fromhex(row["los"]))


class TestPentanomial(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(stats.pentanomial([0, 0.5, 1, 1, 1.5, 2, 2, 2]), [1, 1, 2, 1, 3])
        with self.assertRaises(ValueError):
            stats.pentanomial([2.5])
        with self.assertRaises(ValueError):
            stats.pentanomial([0.3])

    def test_pairing_removes_colour_noise(self):
        """每对都是「执白胜、执黑负」（1 分）：逐局看方差最大，逐对看方差为 0 → 严格均势。"""
        elo, lo, hi = stats.elo_with_error_pentanomial([0, 0, 50, 0, 0])
        self.assertAlmostEqual(elo, 0.0)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0)
        tri = stats.elo_with_error(50, 0, 50)
        self.assertGreater(tri[2] - tri[1], 50)

    def test_score_consistent_with_trinomial(self):
        counts = [3, 7, 20, 9, 5]
        pairs = [0] * 3 + [0.5] * 7 + [1] * 20 + [1.5] * 9 + [2] * 5
        mean = sum(pairs) / (2 * len(pairs))
        self.assertAlmostEqual(stats.elo_with_error_pentanomial(counts)[0],
                               stats.score_to_elo(mean))


if __name__ == "__main__":
    unittest.main()
