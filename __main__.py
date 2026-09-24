"""``python -m Kit <命令>``：kit 的统一命令行入口。

    cd ~/UniChess
    python -m Kit match    <config.json> --out <results.jsonl>      # 成对对局评测（Elo / SPRT）
    python -m Kit selfplay <config.json>                            # 自对弈 → 训练分片
    python -m Kit train    <config.json>                            # 训练（可续跑）
    python -m Kit loop     <loop.json>                              # 自对弈→训练→换代 循环
    python -m Kit uci      <spec.json>                              # 通用 UCI 前端
    python -m Kit data     {evals|pgn|annotate|check} ...            # 训练数据构建与验收
    python -m Kit native-build                                       # 手动编译 C++ 搜索内核

命令的配置格式见各自模块（match / selfplay / train / loop）。所有命令都可以直接在
import 根下的任意目录运行；多进程子进程经 PYTHONPATH 继承 import 根。
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        from .cli import _USAGE
        print(_USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "match":
        from .pipelines.match import main as run
        return run(rest)
    if cmd == "selfplay":
        from .pipelines.selfplay import main as run
        return run(rest)
    if cmd == "loop":
        from .pipelines.loop import main as run
        return run(rest)
    if cmd == "train":
        from .train import run_train
        import json as _json
        if len(rest) != 1:
            raise SystemExit("用法：python -m Kit train <config.json>")
        summary = run_train(rest[0])
        print(_json.dumps(summary, ensure_ascii=False))
        return 0
    if cmd == "uci":
        from .uci import main as run
        return run(rest)
    if cmd == "data":
        from .cli import data_main
        return data_main(rest)
    if cmd == "native-build":
        from .search import native
        print(native.build(force=True))
        return 0
    if cmd == "jobs":
        from .jobs import main as run
        return run(rest)
    raise SystemExit(f"未知命令 {cmd!r}；可用：match selfplay train loop uci data native-build jobs")


if __name__ == "__main__":
    sys.exit(main())
