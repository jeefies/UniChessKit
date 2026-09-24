"""数据层：分片集合、解码、可续跑采样器、专长采样池、课程流。

覆盖的坑：
- 分片样本常驻页缓存时，逐条取样会把 ``decode_batch`` 的向量化收益全吃掉（R 实测 16k → 96k 样本/秒），
  所以必须整批取样（``BatchShardDataset`` + ``batch_size=None``）；
- ``SpecialtyPool`` 的索引与旧 R ``_index_endgame_promotion`` 口径一致（每个分片上限、``rng(seed)`` 选择）；
- 课程过滤方向不能反（opening ≥ 24 / middlegame 12<c<24 / endgame ≤ 12，黑白同一套）；
- ``num_buckets > 1`` 时除 ``num_buckets`` 外不许动其余配置（R AGENTS 记过：replace 会退回默认）；
- 采样器第 0 轮与 torch ``BatchSampler(RandomSampler(ds), batch, drop_last=True)`` 逐位相同
  （R stage1 / T t20m 的 loss 轨迹对拍靠它）。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from Kit.planes19 import dataset as D
from Kit.planes19 import records as R
from Kit.train.data import ResumableBatchSampler


def make_shards(dirpath, n_records=97, prefix="evals", n_shards=1):
    """写 n_shards 个小分片，专长池两端都覆盖。第 i 条按 i%3 取：

    - i%3==0：满开盘（32 子，兵在初始格）；
    - i%3==1：残局（只留 4 个子力，残局池要）；
    - i%3==2：白兵已推进到第 6 横线（升变候选池要）。
    """
    paths = []
    for s in range(n_shards):
        p = Path(dirpath) / f"{prefix}_{s:04d}.bin"
        recs = np.zeros(n_records, dtype=R.RECORD_DTYPE)
        for i in range(n_records):
            kind = i % 3
            if kind == 0:
                b = chess.Board()
            elif kind == 1:
                b = chess.Board("4k3/8/4K3/8/8/8/8/8 w - - 0 1")
            else:
                b = chess.Board("k7/P7/8/1K6/8/8/8/8 w - - 0 1")
            recs[i] = R.board_to_record(b)
        recs.tofile(p)
        paths.append(p)
    return paths


def tmp_shards(n_records=97):
    d = tempfile.mkdtemp(prefix="p19ds_")
    self = None
    return d, D.ShardSet(make_shards(d, n_records))


class TestShardSet(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="p19ss_")
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)
        self.paths = make_shards(self.d, 97)
        self.ss = D.ShardSet(self.paths)

    def test_len_and_gather(self):
        self.assertEqual(len(self.ss), 97)
        direct = np.memmap(self.paths[0], dtype=R.RECORD_DTYPE, mode="r")
        for i in (0, 1, 63, 64, 95, 96):
            np.testing.assert_array_equal(self.ss.gather([i])[0], direct[i])
        # 顺序保持
        recs = self.ss.gather([96, 3, 3, 0])
        self.assertEqual(len(recs), 4)
        np.testing.assert_array_equal(recs[2], self.ss.gather([3])[0])

    def test_list_shards_excludes_selfplay(self):
        sp = Path(self.d) / "sp_0000.sp.bin"
        sp.write_bytes(np.zeros(4, dtype=R.SELFPLAY_DTYPE).tobytes())
        names = [p.name for p in D.list_shards(self.d)]
        self.assertEqual(names, ["evals_0000.bin"])
        names = [p.name for p in D.list_shards(self.d, exclude_selfplay=False)]
        self.assertIn("sp_0000.sp.bin", names)

    def test_decoder_options(self):
        recs = self.ss.gather([0])
        plain = D.Decoder()(recs)
        self.assertEqual(len(plain), 4)
        bucketed = D.Decoder(num_buckets=4)(recs)
        self.assertEqual(len(bucketed), 5)


class TestResumableSamplerParity(unittest.TestCase):
    def setUp(self):
        class DS(torch.utils.data.Dataset):
            def __len__(self):
                return 1000

            def __getitem__(self, i):
                return i

        self.ds = DS()

    def test_first_epoch_matches_torch(self):
        """第 0 轮与 torch 的 BatchSampler(RandomSampler(ds), batch, drop_last=True) 逐位相同。

        两边使用同一个种子（都从 ``manual_seed(0)`` 的全局 CPU RNG 里抽 long），
        各自建生成器做 ``randperm``。只取第 0 轮的 15 个 batch（drop_last 也止步于此），
        避免跨到第 1 轮（跨轮的 randperm 调用次数两边不同，见模块 docstring）。
        """
        torch.manual_seed(0)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        gen = torch.Generator()
        gen.manual_seed(seed)
        ref = list(torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self.ds, generator=gen), batch_size=64, drop_last=True))
        sampler = ResumableBatchSampler(1000, 64, seed0=seed)
        it = iter(sampler)
        got = [next(it) for _ in range(len(ref))]
        self.assertEqual(got, ref)

    def test_seed_drawn_from_global_rng_like_torch(self):
        """种子抽取与 torch RandomSampler 同源：迭代开始时从全局 CPU RNG 取 long，之后不再抽取。"""
        torch.manual_seed(11)
        want = int(torch.empty((), dtype=torch.int64).random_().item())
        torch.manual_seed(11)
        s = ResumableBatchSampler(1000, 64)
        self.assertIsNone(s.seed0)                # 构造时不抽
        next(iter(s))
        self.assertEqual(s.seed0, want)

    def test_seed0_roundtrip(self):
        s = ResumableBatchSampler(1000, 64, seed0=1234, start=0)
        it = iter(s)
        first = [next(it) for _ in range(5)]
        self.assertEqual(s.position(5), (0, 5))
        self.assertEqual(s.seed0, 1234)
        s2 = ResumableBatchSampler(1000, 64, seed0=1234, start=5)
        nxt = [next(iter(s2)) for _ in range(5)]
        self.assertNotEqual(first[0], nxt[0])       # 续的是第 6 批，不是从第 1 批重放
        s3 = ResumableBatchSampler(1000, 64, seed0=1234, start=64)
        # 每轮 1000 // 64 = 15 批：64 = 4 轮整 + 4 批
        self.assertEqual(s3.position(64), (4, 4))
        b = next(iter(s3))
        self.assertEqual(len(b), 64)
        # 同一个 start 重复续跑必须得到同一批（续跑的确定性）
        again = ResumableBatchSampler(1000, 64, seed0=1234, start=64)
        self.assertEqual(next(iter(again)), b)


class TestSpecialtyPool(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="p19sp_")
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)
        self.paths = make_shards(self.d, 40)
        self.ss = D.ShardSet(self.paths)

    def test_counts_and_cache(self):
        cache = Path(self.d) / "pools.npz"
        p1 = D.SpecialtyPool(self.ss, cache, index_seed=7, cap=5)
        self.assertGreater(len(p1.endgame), 0)
        self.assertGreater(len(p1.promotion), 0)
        self.assertLessEqual(len(p1.endgame), 5)
        self.assertLessEqual(len(p1.promotion), 5)
        np.testing.assert_array_equal(
            p1.indices(np.random.default_rng(0), 6, "mixed"),
            p1.indices(np.random.default_rng(0), 6, "mixed"))       # 同 rng 同结果
        total = sum(len(p1.indices(np.random.default_rng(1), 10, m))
                    for m in ("general", "endgame", "promotion"))
        self.assertEqual(total, 30)
        p2 = D.SpecialtyPool(self.ss, cache, index_seed=7, cap=5)   # 命中缓存
        np.testing.assert_array_equal(p1.endgame, p2.endgame)
        self.assertTrue(cache.with_suffix(".json").exists())

    def test_unknown_mode(self):
        p = D.SpecialtyPool(self.ss, None)
        with self.assertRaises(ValueError):
            p.indices(np.random.default_rng(0), 4, "weird")


class TestCurriculumStream(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="p19cs_")
        self.addCleanup(__import__("shutil").rmtree, self.d, ignore_errors=True)
        self.ss = D.ShardSet(make_shards(self.d, 97, n_shards=3))   # 3 个分片，够课程流复用多轮

    def test_filter_direction(self):
        self.assertEqual(list(D.PIECE_FILTERS["opening"](np.array([24, 23, 32]))),
                         [True, False, True])
        self.assertEqual(list(D.PIECE_FILTERS["middlegame"](np.array([12, 13, 23, 24]))),
                         [False, True, True, False])
        self.assertEqual(list(D.PIECE_FILTERS["endgame"](np.array([12, 13, 1]))),
                         [True, False, True])

    def _run(self, state, n_batches):
        st = D.CurriculumStream(self.ss, piece_filter="all", chunk_size=64, batch_size=8,
                                state=state)

        def score(recs):
            return -np.asarray(recs["halfmove"], dtype=np.float32)

        return [recs for _, recs in zip(range(n_batches), st.batches(score))]

    def test_full_batches_and_position(self):
        state = {}
        batches = self._run(state, 6)
        self.assertTrue(all(len(b) == 8 for b in batches))
        self.assertEqual(state["chunks"], 1)
        self.assertEqual(state["pos"], 6)          # 64 条 = 8 个整批

    def _same_batches(self, a, b) -> bool:
        self.assertEqual(len(a), len(b))
        return all(x.tobytes() == y.tobytes() for x, y in zip(a, b))

    def test_position_resume_is_exact(self):
        """中断在 chunk 中间：按 state 续跑，批内容与不中断的完全一致。"""
        full = self._run({}, 8)
        state = {}
        self._run(state, 3)                        # 产出 3 个 batch 后「中断」
        self.assertEqual(state["pos"], 3)
        rest = self._run(state, 5)
        self.assertTrue(self._same_batches(full[3:8], rest))

    def test_chunk_pending_carries_over(self):
        """chunk 0 走完仍有 pending：它们成为下一个 chunk 的前缀（与旧脚本的 buffer 行为一致）。"""
        full = self._run({}, 8)                       # 8 batch = chunk 0 全部
        state = {}
        self._run(state, 4)
        self.assertIsNotNone(state["chunk"])
        # 攒够一个 chunk 就停：第 0 片 97 条里的 64 条成块，余下 33 条留作下一个 chunk 的前缀
        pending = int(np.asarray(state["pending"]).shape[0]) if len(state["pending"]) else 0
        self.assertEqual(pending, 97 - 64)
        rest = self._run(state, 4)
        self.assertTrue(self._same_batches(full[4:8], rest))
        self.assertEqual(state["chunks"], 1)


if __name__ == "__main__":
    unittest.main()
