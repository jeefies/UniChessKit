"""多进程自对弈（``selfplay.workers > 1``）：分片、聚合、续跑。

用假模型 + 真 GumbelCpp（CPU-only），起真子进程验证工头语义。
"""
import tempfile
import unittest
from pathlib import Path

from Kit.pipelines.selfplay import run_selfplay_config
from Kit.planes19.sink import SelfPlayShardSink

#: import 根 = 本文件的上两级（Kit/tests/x.py -> Kit -> ~/UniChess）
IMPORT_ROOT = str(Path(__file__).resolve().parents[2])


def make_fake_gumbel_engine(**_kw):
    """EngineSpec 工厂：假模型 + GumbelCpp 的无参 PlayerFactory（子进程里也能加载）。"""
    from Kit.planes19 import BatchFnEvaluator
    from Kit.players.gumbel_player import make_gumbel_player_factory
    from Kit.testing import FakePlanes19Model

    fake = FakePlanes19Model(salt="workers")
    ev = BatchFnEvaluator("fk:workers", fake.evaluate_planes)
    return make_gumbel_player_factory("F", ev, simulations=8, m0=4, g=0.0)


def _conf(td: Path, games: int = 4, workers: int = 2) -> dict:
    return {
        "engine": {"root": IMPORT_ROOT,
                   "factory": "Kit.tests.test_selfplay_workers:make_fake_gumbel_engine",
                   "kwargs": {}},
        "selfplay": {"games": games, "seed": 11, "max_plies": 40, "concurrency": 2,
                     "workers": workers, "openings": None, "book_plies": 6,
                     "first_game": 0},
        "sink": {"factory": "Kit.planes19.sink:SelfPlayShardSink",
                 "kwargs": {"path": str(td / "selfplay.sp.bin")}},
    }


class TestSelfplayWorkers(unittest.TestCase):
    def test_split_aggregate_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="kit_spw_") as tmp:
            td = Path(tmp)
            conf = _conf(td, games=4, workers=2)
            s = run_selfplay_config(conf)
            self.assertEqual(s["games"], 4, s)
            self.assertEqual(s["workers"], 2)
            self.assertGreater(s["plies"], 0)

            shards = sorted(td.glob("*.sp.bin"))
            self.assertEqual([p.name for p in shards],
                             ["selfplay.w0.sp.bin", "selfplay.w1.sp.bin"])
            games = set()
            for sh in shards:
                games |= SelfPlayShardSink(str(sh)).done_games()
            self.assertEqual(games, {0, 1, 2, 3})

            # 续跑：两边都跳过，聚合 games=0、skipped_done=4
            s2 = run_selfplay_config(conf)
            self.assertEqual(s2["games"], 0, s2)
            self.assertEqual(s2["skipped_done"], 4)


if __name__ == "__main__":
    unittest.main()