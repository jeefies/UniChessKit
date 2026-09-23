"""Batch Arena：A 对 B 的成对对局评测。

蓝本是 S 的 ``tools/ssm_gumbel_arena.py``（BatchedArenaDriver），并吸收 Server P0 热修的教训：

- **成对开局**：第 p 对的两局用同一开局，A 先执白（第 2p 局）、B 再执白（第 2p+1 局）；
  开局按带种子的排列分配（rules.OpeningBook.plan），各对开局互不相同。
- **计分按模型，不按颜色**：每局记录 A 的得分；执白方只是记录字段。
- **跨局攒批**：所有在跑的对局是协程，同一模型的叶子每拍拼成一次前向（runtime.CoroutinePool）。
- **裁决统一**：rules.StandardReferee（claim_draw 语义，走满 max_plies 记截断 = 和）。
- **出错整批停止**：任何 Player 抛异常或走非法着法 → 整批中止并原样报错，不产出半截统计。
- **SPRT 早停**：以五项式（逐对）GSPRT 判定；只在完整的对上计算；达到判定后不再开新局，
  已开局的照常下完并入库。
- **断点续跑**：结果逐局追加写入 JSONL（每行 flush + fsync），首行是含配置哈希的表头；
  重跑同一命令时核对哈希、跳过已完成的局。文件末尾被截断的半行会被丢弃。
- **有效样本量**：distinct_games / duplicate_rate（S 教训：确定性对局下重复开局 = 逐字节重复棋谱）。

可复现性：单进程（workers=1）、不用 deadline 时，同一配置两次运行的着法与结果逐局一致。
多进程时各 worker 内部仍确定，但 SPRT 早停的时点取决于进程间的完成顺序。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import chess

from .. import stats
from ..api.types import GameStart, MoveDecision, PlayerError, SearchBudget
from ..jsonutil import json_safe
from ..provenance import collect as collect_provenance
from ..registry import EngineSpec, build_player_factory
from ..rules.openings import BUNDLED_OPENINGS, OpeningBook
from ..rules.referee import StandardReferee
from ..runtime.batcher import Batcher, CoroutinePool
from ..runtime.workers import WorkerPool

RESULT_SCHEMA = 1


@dataclass(frozen=True)
class SprtConfig:
    elo0: float = 0.0
    elo1: float = 10.0
    alpha: float = 0.05
    beta: float = 0.05
    min_pairs: int = 8          # 少于这么多完整对不做判定（正态近似在小样本下不可靠）


@dataclass(frozen=True)
class MatchConfig:
    pairs: int
    seed: int = 0
    max_plies: int = 400
    concurrency: int = 8
    openings: Optional[str] = "bundled"     # "bundled" / 文件路径 / None（全部从初始局面开始）
    simulations: Optional[int] = None       # 覆盖双方 Player 的默认模拟数；None = 各用各的
    sprt: Optional[SprtConfig] = None
    workers: int = 1

    def __post_init__(self):
        if self.pairs < 1:
            raise ValueError("pairs 必须 >= 1")
        if self.max_plies < 1 or self.concurrency < 1 or self.workers < 1:
            raise ValueError("max_plies / concurrency / workers 必须 >= 1")

    @classmethod
    def from_dict(cls, d: dict) -> "MatchConfig":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"MatchConfig 有未知字段 {sorted(unknown)}")
        if d.get("sprt") is not None:
            d["sprt"] = SprtConfig(**d["sprt"])
        return cls(**d)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------ 对局计划

@dataclass(frozen=True)
class GameTask:
    game: int
    pair: int
    a_is_white: bool
    opening: tuple
    seed_a: int
    seed_b: int
    fen: Optional[str] = None           # 起始局面（None = 标准初始局面；批量对弈不用）


def _seed(base: int, game: int, side: str) -> int:
    return zlib.crc32(f"{base}:{game}:{side}".encode()) & 0x7FFFFFFF


def load_book(openings: Optional[str]) -> Optional[OpeningBook]:
    if openings is None:
        return None
    path = BUNDLED_OPENINGS if openings == "bundled" else Path(openings)
    return OpeningBook.from_file(path)


def plan_games(cfg: MatchConfig, book: Optional[OpeningBook]) -> list:
    if book is None:
        openings = [()] * cfg.pairs
    else:
        openings = [line for _, line in book.plan(cfg.pairs, cfg.seed)]
    tasks = []
    for p, opening in enumerate(openings):
        for k in range(2):
            g = 2 * p + k
            tasks.append(GameTask(game=g, pair=p, a_is_white=(k == 0), opening=tuple(opening),
                                  seed_a=_seed(cfg.seed, g, "A"), seed_b=_seed(cfg.seed, g, "B")))
    return tasks


# ------------------------------------------------------------------ 单局协程

def play_game(task: GameTask, player_a, player_b, referee: StandardReferee,
              budget: SearchBudget, observer: Optional[Callable[[dict], None]] = None):
    """一局棋的协程，返回结果记录（dict）。结束时（含异常）关闭双方 Player。

    observer（可选）逐步收到 ``{"type": "game_start"}`` / ``{"type": "move"}`` 事件，
    用于观战与实时进度；它抛出的异常会中止对局（与 Player 出错同样处理）。
    """
    t0 = time.perf_counter()
    white, black = (player_a, player_b) if task.a_is_white else (player_b, player_a)
    side_of = {id(player_a): "A", id(player_b): "B"}
    sources = {"A": Counter(), "B": Counter()}
    board = chess.Board(task.fen) if task.fen else chess.Board()
    try:
        for uci in task.opening:
            board.push_uci(uci)
        yield from player_a.new_game(GameStart(color=task.a_is_white, seed=task.seed_a,
                                               opening=task.opening, game_id=f"g{task.game}",
                                               fen=task.fen))
        yield from player_b.new_game(GameStart(color=not task.a_is_white, seed=task.seed_b,
                                               opening=task.opening, game_id=f"g{task.game}",
                                               fen=task.fen))
        moves = []
        if observer is not None:
            observer({"type": "game_start", "game": task.game, "pair": task.pair,
                      "white": "A" if task.a_is_white else "B",
                      "opening": list(task.opening), "fen": task.fen})
        while True:
            verdict = referee.verdict(board)
            if verdict is not None:
                break
            mover = white if board.turn == chess.WHITE else black
            t_move = time.perf_counter()
            decision = yield from mover.choose(board.copy(), budget)
            if not isinstance(decision, MoveDecision):
                raise PlayerError(f"{mover.name}.choose 应返回 MoveDecision，"
                                  f"收到 {type(decision).__name__}")
            if decision.move not in board.legal_moves:
                raise PlayerError(f"{mover.name} 走了非法着法 {decision.move} @ {board.fen()}"
                                  f"（第 {task.game} 局）")
            sources[side_of[id(mover)]][decision.source] += 1
            board.push(decision.move)
            moves.append(decision.move.uci())
            if observer is not None:
                observer({"type": "move", "game": task.game, "ply": len(board.move_stack),
                          "uci": decision.move.uci(), "side": side_of[id(mover)],
                          "source": decision.source,
                          "ms": int((time.perf_counter() - t_move) * 1000),
                          "info": json_safe(decision.info) or {}})
            for p in (white, black):
                yield from p.observe(board.copy(), decision.move)
    finally:
        for p in (player_a, player_b):
            try:
                p.close()
            except Exception as e:  # noqa: close 失败不应掩盖真正的异常
                print(f"[match] {getattr(p, 'name', p)}.close() 失败：{e}", file=sys.stderr)
    white_score = verdict.white_score()
    return {
        "type": "game", "game": task.game, "pair": task.pair,
        "white": "A" if task.a_is_white else "B",
        "opening": list(task.opening), "moves": moves,
        "result": verdict.result, "termination": verdict.termination,
        "plies": len(board.move_stack),
        "a_score": white_score if task.a_is_white else 1.0 - white_score,
        "sources": {k: dict(sorted(v.items())) for k, v in sources.items()},
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


# ------------------------------------------------------------------ 统计

def fingerprint(record: dict) -> str:
    return " ".join(record["opening"] + record["moves"])


def summarize(records: list, cfg: MatchConfig, names: dict) -> dict:
    records = sorted(records, key=lambda r: r["game"])
    n = len(records)
    w = sum(1 for r in records if r["a_score"] == 1.0)
    l = sum(1 for r in records if r["a_score"] == 0.0)
    d = n - w - l
    by_pair: dict = {}
    for r in records:
        by_pair.setdefault(r["pair"], []).append(r["a_score"])
    pair_scores = [sum(v) for v in by_pair.values() if len(v) == 2]
    penta = stats.pentanomial(pair_scores)
    elo, lo, hi = stats.elo_with_error(w, d, l)
    p_elo, p_lo, p_hi = stats.elo_with_error_pentanomial(penta)
    terms = Counter(r["termination"] for r in records)
    distinct = len({fingerprint(r) for r in records})
    color = {}
    for side in ("white", "black"):
        rs = [r for r in records if (r["white"] == "A") == (side == "white")]
        color[f"a_as_{side}"] = {"games": len(rs), "score": sum(r["a_score"] for r in rs)}
    sources = {"A": Counter(), "B": Counter()}
    for r in records:
        for side in ("A", "B"):
            sources[side].update(r["sources"][side])
    out = {
        "a": names.get("A"), "b": names.get("B"),
        "games": n, "pairs_complete": len(pair_scores),
        "a_wins": w, "b_wins": l, "draws": d,
        "score_a": (w + 0.5 * d) / n if n else None,
        "elo": elo, "elo_ci95": [lo, hi], "los": stats.los(w, d, l),
        "pentanomial": penta, "elo_pentanomial": p_elo, "elo_pentanomial_ci95": [p_lo, p_hi],
        "termination": dict(sorted(terms.items())),
        "truncated_rate": terms.get("truncated", 0) / n if n else 0.0,
        "distinct_games": distinct,
        "duplicate_rate": 1.0 - distinct / n if n else 0.0,
        "by_color": color,
        "sources": {k: dict(sorted(v.items())) for k, v in sources.items()},
        "mean_plies": sum(r["plies"] for r in records) / n if n else 0.0,
    }
    if cfg.sprt is not None:
        s = cfg.sprt
        llr = stats.sprt_llr_pentanomial(penta, s.elo0, s.elo1)
        verdict = (stats.sprt_verdict(llr, s.alpha, s.beta)
                   if len(pair_scores) >= s.min_pairs else None)
        out["sprt"] = {**dataclasses.asdict(s), "llr": llr,
                       "llr_trinomial": stats.sprt_llr(w, d, l, s.elo0, s.elo1),
                       "bounds": list(stats.sprt_bounds(s.alpha, s.beta)),
                       "verdict": verdict}
    return out


def _sprt_decided(records: list, cfg: MatchConfig) -> bool:
    if cfg.sprt is None:
        return False
    return summarize(records, cfg, {}).get("sprt", {}).get("verdict") is not None


# ------------------------------------------------------------------ 结果文件

def config_hash(cfg: MatchConfig, players: dict) -> str:
    blob = json.dumps({"schema": RESULT_SCHEMA, "match": cfg.to_dict(), "players": players},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ResultLog:
    """JSONL 结果文件：表头 + 每局一行。支持断点续跑。"""

    def __init__(self, path, header: dict):
        self.path = Path(path)
        self.header = header
        self.records: list = []
        self._fh = None

    def open(self) -> list:
        """打开（或续接）文件，返回已完成的记录。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            self.records = self._load_existing()
            self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        else:
            self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
            self._write(self.header)
        return list(self.records)

    def _load_existing(self) -> list:
        raw = self.path.read_bytes()
        # 每条记录以换行结尾；没有换行结尾的最后一段是写入中途被杀留下的半行
        cut = raw.rfind(b"\n") + 1
        body, tail = raw[:cut], raw[cut:]
        parsed = []
        for i, line in enumerate(body.split(b"\n")[:-1], 1):
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"{self.path} 第 {i} 行损坏，无法续跑") from None
        if tail.strip():
            try:
                parsed.append(json.loads(tail))
                body += tail + b"\n"            # 完整但缺换行：补上，后续追加不粘连
            except json.JSONDecodeError:
                print(f"[match] 丢弃 {self.path} 末尾不完整的一行", file=sys.stderr)
        if body != raw:
            self.path.write_bytes(body)
        if not parsed or parsed[0].get("type") != "header":
            raise ValueError(f"{self.path} 缺少表头，不是本工具写的结果文件")
        old = parsed[0]
        if old.get("config_hash") != self.header["config_hash"]:
            raise ValueError(f"{self.path} 的配置哈希 {old.get('config_hash')} 与本次 "
                             f"{self.header['config_hash']} 不一致；换一个输出路径或删除旧文件")
        games = [r for r in parsed[1:] if r.get("type") == "game"]
        seen = set()
        for r in games:
            if r["game"] in seen:
                raise ValueError(f"{self.path} 中第 {r['game']} 局重复出现")
            seen.add(r["game"])
        return games

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def append(self, record: dict) -> None:
        self.records.append(record)
        self._write(record)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ------------------------------------------------------------------ 编排

class _Run:
    """一次评测的共享状态：已完成记录、结果文件、SPRT 停止判定。"""

    def __init__(self, cfg: MatchConfig, names: dict, log: Optional[ResultLog],
                 progress: Optional[Callable], observer: Optional[Callable] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.cfg, self.names, self.log, self.progress = cfg, names, log, progress
        self.observer = observer
        self.external_stop = should_stop
        self.records: list = []
        self.stopped = False

    def add(self, record: dict) -> None:
        if self.log is not None:
            self.log.append(record)
        self.records.append(record)
        if not self.stopped and _sprt_decided(self.records, self.cfg):
            self.stopped = True
        if self.progress is not None:
            self.progress(record, self.records)

    def should_stop(self) -> bool:
        return self.stopped or (self.external_stop is not None and self.external_stop())


def _run_local(cfg, tasks, make_a, make_b, run: _Run) -> dict:
    referee = StandardReferee(max_plies=cfg.max_plies)
    budget = SearchBudget(simulations=cfg.simulations)
    batcher = Batcher()
    pool = CoroutinePool(cfg.concurrency, batcher)
    jobs = ((t.game, (lambda t=t: play_game(t, make_a(), make_b(), referee, budget,
                                            run.observer)))
            for t in tasks)
    pool.run(jobs, lambda _gid, rec: run.add(rec), should_stop=run.should_stop)
    return batcher.stats.as_dict()


def _match_worker(wid, task, emit, stop_event):
    """WorkerPool 的 worker：在子进程里加载双方引擎，跑分到的局。"""
    cfg_d, spec_a, spec_b, tasks, observe = task
    cfg = MatchConfig.from_dict(cfg_d)
    make_a = build_player_factory(EngineSpec.from_dict(spec_a))
    make_b = build_player_factory(EngineSpec.from_dict(spec_b))
    referee = StandardReferee(max_plies=cfg.max_plies)
    budget = SearchBudget(simulations=cfg.simulations)
    batcher = Batcher()
    pool = CoroutinePool(cfg.concurrency, batcher)
    observer = (lambda ev: emit({"type": "event", "event": ev})) if observe else None
    jobs = ((t.game, (lambda t=t: play_game(t, make_a(), make_b(), referee, budget, observer)))
            for t in tasks)
    pool.run(jobs, lambda _gid, rec: emit(rec), should_stop=stop_event.is_set)
    emit({"type": "batch_stats", **batcher.stats.as_dict()})


def run_match(cfg: MatchConfig, *, spec_a: Optional[EngineSpec] = None,
              spec_b: Optional[EngineSpec] = None, make_a: Optional[Callable] = None,
              make_b: Optional[Callable] = None, out_path=None,
              names: Optional[dict] = None, progress: Optional[Callable] = None,
              observer: Optional[Callable[[dict], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """跑一次评测，返回汇总（同时写入 out_path 与 <out_path>.summary.json）。

    两种用法：给 EngineSpec（可多进程），或直接给 PlayerFactory（仅单进程，测试 / 嵌入用）。
    observer 收到逐步事件（见 play_game；多进程时经 worker 转发）；should_stop 返回真后
    不再开新局，已开局的下完入库。
    """
    if (spec_a is None) != (spec_b is None) or (make_a is None) != (make_b is None) \
            or (spec_a is None) == (make_a is None):
        raise ValueError("请二选一：同时给 spec_a/spec_b，或同时给 make_a/make_b")
    if cfg.workers > 1 and spec_a is None:
        raise ValueError("多进程需要 EngineSpec（PlayerFactory 无法跨进程传递）")
    if names is None:
        names = ({"A": spec_a.name, "B": spec_b.name} if spec_a is not None
                 else {"A": getattr(make_a, "name", "A"), "B": getattr(make_b, "name", "B")})
    players_id = ({"A": spec_a.to_dict(), "B": spec_b.to_dict()} if spec_a is not None
                  else dict(names))

    book = load_book(cfg.openings)
    tasks = plan_games(cfg, book)
    header = {"type": "header", "schema": RESULT_SCHEMA, "config_hash": config_hash(cfg, players_id),
              "match": cfg.to_dict(), "players": players_id, "names": names,
              "opening_library": len(book) if book is not None else 0,
              "provenance": collect_provenance(
                  {"A": spec_a.root, "B": spec_b.root} if spec_a is not None else None)}
    log = ResultLog(out_path, header) if out_path is not None else None
    run = _Run(cfg, names, log, progress, observer, should_stop)
    t0 = time.perf_counter()
    batch_stats: dict = {}
    try:
        if log is not None:
            for r in log.open():
                run.records.append(r)
            done = {r["game"] for r in run.records}
            if done:
                print(f"[match] 续跑：已完成 {len(done)} 局", file=sys.stderr)
            tasks = [t for t in tasks if t.game not in done]
            run.stopped = _sprt_decided(run.records, cfg)
        if tasks and not run.should_stop():
            if cfg.workers == 1:
                if make_a is None:
                    make_a, make_b = build_player_factory(spec_a), build_player_factory(spec_b)
                batch_stats = _run_local(cfg, tasks, make_a, make_b, run)
            else:
                batch_stats = _run_workers(cfg, tasks, spec_a, spec_b, run)
    finally:
        if log is not None:
            log.close()
    summary = summarize(run.records, cfg, names)
    summary.update(config_hash=header["config_hash"], games_planned=2 * cfg.pairs,
                   stopped_by_sprt=run.stopped and len(run.records) < 2 * cfg.pairs,
                   batch=batch_stats, elapsed_s=round(time.perf_counter() - t0, 1),
                   provenance=header["provenance"])
    if out_path is not None:
        Path(str(out_path) + ".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_workers(cfg, tasks, spec_a, spec_b, run: _Run) -> dict:
    shards = [[t for i, t in enumerate(tasks) if i % cfg.workers == w] for w in range(cfg.workers)]
    shards = [s for s in shards if s]
    totals: Counter = Counter()

    def on_result(_wid, obj):
        if obj.get("type") == "batch_stats":
            totals.update({k: v for k, v in obj.items() if k != "type"})
        elif obj.get("type") == "event":
            if run.observer is not None:
                run.observer(obj["event"])
        else:
            run.add(obj)

    observe = run.observer is not None
    WorkerPool().run(_match_worker,
                     [(cfg.to_dict(), spec_a.to_dict(), spec_b.to_dict(), s, observe)
                      for s in shards],
                     on_result, should_stop=run.should_stop)
    return dict(totals)


# ------------------------------------------------------------------ 命令行

def _print_progress(record: dict, records: list) -> None:
    n = len(records)
    w = sum(1 for r in records if r["a_score"] == 1.0)
    l = sum(1 for r in records if r["a_score"] == 0.0)
    print(f"[match] 第 {record['game']} 局 {record['result']} ({record['termination']}, "
          f"{record['plies']} ply, A 执{'白' if record['white'] == 'A' else '黑'})  "
          f"累计 {n} 局 A {w} 胜 {l} 负 {n - w - l} 和", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="UniChessKit 成对对局评测")
    ap.add_argument("config", help="JSON：{\"a\": EngineSpec, \"b\": EngineSpec, \"match\": {...}}")
    ap.add_argument("--out", required=True, help="结果 JSONL 路径（已存在则续跑）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    conf = json.loads(Path(args.config).read_text(encoding="utf-8"))
    unknown = set(conf) - {"a", "b", "match"}
    if unknown:
        raise SystemExit(f"配置有未知字段 {sorted(unknown)}")
    summary = run_match(MatchConfig.from_dict(conf["match"]),
                        spec_a=EngineSpec.from_dict(conf["a"]),
                        spec_b=EngineSpec.from_dict(conf["b"]),
                        out_path=args.out,
                        progress=None if args.quiet else _print_progress)
    brief = {k: summary[k] for k in ("a", "b", "games", "a_wins", "b_wins", "draws", "score_a",
                                     "elo", "elo_ci95", "pentanomial", "distinct_games",
                                     "termination", "batch", "elapsed_s")}
    if "sprt" in summary:
        brief["sprt"] = summary["sprt"]
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
