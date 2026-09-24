"""CLI 里与主子命令无关的部分：``python -m Kit`` 帮助文本、``data`` 子命令分发。"""
from __future__ import annotations

import json
from pathlib import Path

_USAGE = """Kit —— UniChess 对局 / 搜索 / 评测 / 训练框架

    python -m Kit match    <config.json> --out <results.jsonl>   成对对局评测（Elo / LOS / SPRT）
    python -m Kit selfplay <config.json>                         自对弈 → 163 字节训练分片
    python -m Kit train    <config.json>                         训练（中断后原目录续跑）
    python -m Kit loop     <loop.json>                           自对弈 → 训练 → 换代门禁
    python -m Kit uci      <spec.json>                           通用 UCI 前端
    python -m Kit data     {evals|pgn|annotate|check} ...         训练数据构建与验收
    python -m Kit native-build                                   手动编译 C++ 搜索内核
    python -m Kit jobs     <job.json>                             对局 job 进程（Server 内部用）

各命令的配置字段见对应模块（Kit/pipelines/match.py、selfplay.py、loop.py、train/config.py）。
服务器侧的对局由 Server 的 jobs.py 通过 ``python -m Kit.jobs`` 启动，一般不手动跑。
"""

_DATA_CMDS = {
    "evals": ("从 Lichess 评估库 jsonl(.zst) 生成 96 字节监督分片", [
        ("--src", {"required": True, "help": "lichess_db_eval.jsonl.zst"}),
        ("--out", {"required": True, "help": "分片输出目录"}),
        ("--workers", {"type": int, "default": 4}),
        ("--batch", {"type": int, "default": 4000, "help": "每个任务的行数"}),
        ("--shard-records", {"type": int, "default": 2000000}),
        ("--temperature", {"type": float, "default": 90.0, "help": "策略 softmax 温度（厘兵）"}),
        ("--min-pv", {"type": int, "default": 2}),
        ("--min-depth", {"type": int, "default": 12}),
        ("--limit", {"type": int, "default": 0, "help": "只读前 N 行，0 = 全部"}),
        ("--prefix", {"default": "evals"}),
    ]),
    "pgn": ("从 Lichess 月度 PGN(.zst) 生成 96 字节价值分片", [
        ("--src", {"required": True, "help": "lichess_db_standard_rated_YYYY-MM.pgn.zst"}),
        ("--out", {"required": True, "help": "分片输出目录"}),
        ("--workers", {"type": int, "default": 4}),
        ("--games-per-chunk", {"type": int, "default": 200}),
        ("--shard-records", {"type": int, "default": 2000000}),
        ("--skip-plies", {"type": int, "default": 8}),
        ("--min-plies", {"type": int, "default": 20}),
        ("--with-policy", {"action": "store_true", "help": "附带人类实走 one-hot（默认不带）"}),
        ("--max-records", {"type": int, "default": 0}),
        ("--prefix", {"default": "pgn"}),
    ]),
    "annotate": ("本机 Stockfish MultiPV 标注（每局面 10 万节点 × MultiPV 5）", [
        ("--fens", {"required": True, "help": "每行一个 FEN 的文本文件"}),
        ("--out", {"required": True, "help": "分片输出目录"}),
        ("--stockfish", {"required": True, "help": "stockfish 可执行文件路径（GPL，不入库）"}),
        ("--workers", {"type": int, "default": 14}),
        ("--shard-size", {"type": int, "default": 20000, "help": "每个分片的局面数"}),
        ("--nodes", {"type": int, "default": 100000}),
        ("--multipv", {"type": int, "default": 5}),
        ("--threads", {"type": int, "default": 1, "help": "每个 SF 进程的线程数"}),
        ("--hash-mb", {"type": int, "default": 256}),
    ]),
    "check": ("验收分片：着法合法 / 策略归一 / WDL 归一 / 易位写法", [
        ("shards", {"nargs": "+", "help": "分片文件路径"}),
        ("--sample", {"type": int, "default": 50000, "help": "每个分片抽查多少条"}),
    ]),
}


def _add(ap, spec):
    """spec = [(flag, kwargs)]；kwargs 里缺省 kind='flag' 的位置参数用 'shards' 形式传 None。"""
    for name, kw in spec:
        if name.startswith("-"):
            ap.add_argument(name, **kw)
        else:                                   # 位置参数
            ap.add_argument(name, **kw)


def data_main(argv=None) -> int:
    import argparse

    if not argv:
        print(_USAGE)
        print("\ndata 子命令：" + "、".join(f"{k}（{v[0]}）" for k, v in _DATA_CMDS.items()))
        return 0
    cmd = argv[0]
    if cmd not in _DATA_CMDS:
        raise SystemExit(f"未知 data 子命令 {cmd!r}；可用：{sorted(_DATA_CMDS)}")
    desc, spec = _DATA_CMDS[cmd]
    ap = argparse.ArgumentParser(prog=f"python -m Kit data {cmd}", description=desc)
    _add(ap, spec)
    a = ap.parse_args(argv[1:])
    args = dict(vars(a))
    if cmd == "evals":
        from .planes19.build.evals import build
        stats = build(args.pop("src"), args.pop("out"), **{k: v for k, v in args.items()
                                                           if k != "prefix" or v != "evals"})
    elif cmd == "pgn":
        from .planes19.build.pgn import build
        stats = build(args.pop("src"), args.pop("out"), **args)
    elif cmd == "annotate":
        from .planes19.build.annotate import annotate

        def fen_iter(path, limit=0):
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if limit and i >= limit:
                        break
                    line = line.strip()
                    if line:
                        yield line

        stats = annotate(fen_iter(a.fens), a.out, a.stockfish, workers=a.workers,
                         shard_size=a.shard_size, nodes=a.nodes, multipv=a.multipv,
                         threads=a.threads, hash_mb=a.hash_mb)
    else:
        from .planes19.build.check import check_shard
        results = [check_shard(p, a.sample) for p in a.shards]
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
        if not all(r["ok"] for r in results):
            raise SystemExit(1)
        stats = {"shards": len(results), "ok": True}
    print(json.dumps(stats, ensure_ascii=False))
    return 0
