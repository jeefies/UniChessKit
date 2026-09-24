"""旧实现冻结的黄金数据（``tests/fixtures/*.json.gz``）。

由 tag ``pre-rebuild-20260924`` 的旧代码生成：R ``search/mcts.py``（Python PUCT）、
R/T ``core.encoding`` / ``core.moves``、R ``eval/arena.py`` 的统计、S ``stateseq/gumbel.py``。
旧代码已删除，kit 的实现只和这些数据比；浮点一律按 ``float.hex`` 存（逐位）。

生成脚本（留档）：远端 ``/tmp/rebuild_baseline/gen_golden.py``，依赖 kit 的 ``FakePlanesEvaluator``——
它的输出一旦变化，这些数据就作废（r_priors 的逐位先验会最先报错）。
"""
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@lru_cache(maxsize=None)
def load(name: str):
    with gzip.open(FIXTURES / f"{name}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def floats(hexes) -> np.ndarray:
    return np.array([float.fromhex(h) for h in hexes], dtype=np.float64)


def assert_tree(tc, node, snap, depth=None, path="root"):
    """kit PUCT 节点与冻结快照逐位相等（N / W / P / 终局值 / 子树）。"""
    tc.assertEqual([m.uci() for m in node.moves], snap["moves"], path)
    tc.assertEqual(np.asarray(node.N).tolist(), snap["N"], path)
    np.testing.assert_array_equal(np.asarray(node.W, np.float64), floats(snap["W"]), err_msg=path)
    np.testing.assert_array_equal(np.asarray(node.P, np.float64), floats(snap["P"]), err_msg=path)
    tv = snap["terminal_value"]
    tc.assertEqual(node.terminal_value, None if tv is None else float.fromhex(tv), path)
    if "children" in snap and (depth is None or depth > 0):
        tc.assertEqual(len(node.children), len(snap["children"]), path)
        for i, (c, s) in enumerate(zip(node.children, snap["children"])):
            tc.assertEqual(c is None, s is None, f"{path}/{i}")
            if c is not None:
                assert_tree(tc, c, s, None if depth is None else depth - 1, f"{path}/{snap['moves'][i]}")
