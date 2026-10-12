"""自对弈管线：一个 Player 执双方下完整盘，逐局交给引擎的 RecordSink 落盘。

口径移植自 S 的 ``tools/ssm_gumbel_selfplay.py``（Stage B 生成器），逐条对应：

- **开局注入**：第 g 局用开局库第 ``g % 库大小`` 条，裁到 ``book_plies``；这些着法**由 Player 照 book 走**
   （``GameStart.book``），Player 不搜索、直接给出这一步（没有访问分布，训练目标里跳过）。
   开局库**不去重**：裁切后重复的线各自保留序号（S 的 ``load_openings`` 如此；去重会改变
   开局分配，破坏与既有数据的可比性）。非法线丢弃并计数。
- **分片**：局序号是全局的（``first_game`` 起），开局、随机数流、局键都只取决于全局序号，
   因此把 N 局拆给多个进程（各取不相交区间、同一 seed）与单进程跑出的是同一批对局。
- **随机数**：管线把**全局 seed 原样**交给 Player，同时通过 ``GameStart.index`` 给出局序号；
    每局的随机数流由 Player 自己按 ``default_rng(SeedSequence(seed, spawn_key=(index,)))``
    派生（``SsmSelfPlayer`` 与 ``SearchPlayer`` 都是这个口径，与
    ``SeedSequence(seed).spawn(num_games)[index]`` 逐位同源）。**不要**改成在管线里派生后
    只传一个整数种子：S 的开局 π′ 缓存键是 ``(seed, book_id, ply)``，故意不含局序号，
    这样同一条开局在所有局里共享搜索结果；换了 per-game 种子缓存全部落空。
- **开局库缺失时的多样性**：没有开局文件时（``openings=None``）只有随机数流在区分各局，
    所以各局棋谱必然不同，但开局阶段完全一致。
- **裁决**：rules.StandardReferee（claim_draw 语义；max_plies 按整盘计，含 book）。
- 出错整批停止（CoroutinePool 语义），Sink 不会收到半局。

**双引擎（联盟自对弈）**：``run_selfplay_config`` 的配置里给可选的 ``engine_b``
（EngineSpec）时，两个 Player 各执一方——**颜色按局号分配**（偶数局 A 执白、奇数局
A 执黑，与 ``match`` 的成对换先同一目的），每步由行棋方搜索、双方都 observe、
两侧的 π′ 都进分片（``GameStart.record_visits``）。book 强制、种子派生（A 用全局
seed 原样、B 按局派生另一个流）、record 结构与单引擎同构；单引擎路径逐字节不变。

可复现性：并发 1 时与 S 原生成器逐字节一致；并发 > 1 时批组成随完成顺序变化，
``SeqModel.step`` 的输出随批大小有 ~1e-5 浮点差，两边都只在统计意义上一致。
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess
import numpy as np

from ..api.types import GameStart, MoveDecision, PlayerError, SearchBudget
from ..rules.openings import parse_line
from ..rules.referee import StandardReferee
from ..runtime.batcher import Batcher, CoroutinePool, ThreadedCoroutinePool


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int
    seed: int = 0
    max_plies: int = 300
    concurrency: int = 128
    openings: Optional[str] = None      # 开局文件路径；None = 全部从初始局面开始
    book_plies: int = 6
    min_book_plies: Optional[int] = None # 开局随机截断最小深度；None = 固定 book_plies
    simulations: Optional[int] = None   # 覆盖 Player 默认模拟数
    first_game: int = 0                 # 全局局序号起点（多进程分片：各进程取不相交区间）
    workers: int = 1                    # 多进程分片：每 worker 一个进程 + 一个独立分片（.w{i}.sp.bin）
    threads: int = 1                    # 单进程内多线程池（共享一个 CUDA context；见 ThreadedCoroutinePool）

    def __post_init__(self):
        if (self.games < 0 or self.max_plies < 1 or self.concurrency < 1 or self.book_plies < 0
                or self.first_game < 0 or self.workers < 1 or self.threads < 1):
            raise ValueError("games / book_plies / first_game >= 0，max_plies / concurrency / "
                             "workers / threads >= 1")
        if self.min_book_plies is not None and (self.min_book_plies < 0 or self.min_book_plies > self.book_plies):
            raise ValueError(f"min_book_plies ({self.min_book_plies}) 必须在 [0, book_plies={self.book_plies}] 之间")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SelfPlayTask:
    game: int
    book: tuple = ()
    book_id: Optional[int] = None


def game_seed_sequence(seed: int, index: int) -> np.random.SeedSequence:
    """第 index 局的随机数种子序列（= ``SeedSequence(seed).spawn(n)[index]``）。"""
    return np.random.SeedSequence(int(seed), spawn_key=(int(index),))


def load_book_lines(path, book_plies: int) -> tuple:
    """开局文件 → (UCI 元组列表, 丢弃的非法行数)。裁到 book_plies，不去重（见模块说明）。"""
    lines, dropped = [], 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        toks = raw.split("#", 1)[0].split()
        if not toks:
            continue
        try:
            lines.append(parse_line(" ".join(toks[:book_plies])))
        except ValueError:
            dropped += 1
    return lines, dropped


def plan_selfplay(cfg: SelfPlayConfig, lines: Optional[list] = None) -> list:
    games = range(cfg.first_game, cfg.first_game + cfg.games)
    if not lines:
        return [SelfPlayTask(game=g) for g in games]
    tasks = []
    for g in games:
        full_line = tuple(lines[g % len(lines)])
        if cfg.min_book_plies is not None and cfg.min_book_plies < len(full_line):
            rng = np.random.default_rng(game_seed_sequence(cfg.seed, g))
            k = int(rng.integers(cfg.min_book_plies, len(full_line) + 1))
            tasks.append(SelfPlayTask(game=g, book=full_line[:k], book_id=g % len(lines)))
        else:
            tasks.append(SelfPlayTask(game=g, book=full_line, book_id=g % len(lines)))
    return tasks


def side_seed(seed: int, game: int, side: str) -> int:
    """双引擎对局里某一方的种子（``match._seed`` 的同款口径）。

    两个引擎的随机数流**必须不同**：同流会让两个模型的 Gumbel 噪声逐 ply 完全同步，
    多样性白丢。A 方（``engine``）不走这里——它原样用全局 seed，与单引擎路径同构
    （同一局号下 A 的噪声流不变）；B 方（``engine_b``）按 ``(seed, 局号, "B")`` 派生。
    """
    return zlib.crc32(f"{seed}:{game}:{side}".encode()) & 0x7FFFFFFF


def play_selfplay_game(task: SelfPlayTask, player, referee: StandardReferee,
                       budget: SearchBudget, seed: int, player_b=None):
    """一局自对弈协程 → (record, 终局 board, decisions)。结束时（含异常）关闭 Player。

    ``player_b`` 为 None：单引擎自对弈，一个 Player 执双方（既有口径，逐字节不变）。
    给出时：双引擎对局，两个 Player 各执一方，颜色按局号分配（见模块说明）；
    book 强制着法由行棋方照常搜索产出 π′ 再原样走出（与单引擎同一条规则），
    每步之后双方都 ``observe``。
    """
    t0 = time.perf_counter()
    board = chess.Board()
    decisions: list = []
    moves: list = []
    book = [chess.Move.from_uci(u) for u in task.book]
    dual = player_b is not None
    try:
        if dual:
            a_is_white = (task.game % 2 == 0)
            white, black = (player, player_b) if a_is_white else (player_b, player)
            yield from player.new_game(GameStart(
                color=a_is_white, seed=seed, game_id=f"g{task.game}",
                book=tuple(task.book), book_id=task.book_id, index=task.game,
                record_visits=True))
            yield from player_b.new_game(GameStart(
                color=not a_is_white, seed=side_seed(seed, task.game, "B"),
                game_id=f"g{task.game}", book=tuple(task.book), book_id=task.book_id,
                index=task.game, record_visits=True))
        else:
            white = black = player
            yield from player.new_game(GameStart(color=chess.WHITE, seed=seed, game_id=f"g{task.game}",
                                                 both_sides=True, book=tuple(task.book),
                                                 book_id=task.book_id, index=task.game))
        while True:
            verdict = referee.verdict(board)
            if verdict is not None:
                break
            ply = len(board.move_stack)
            mover = white if board.turn == chess.WHITE else black
            decision = yield from mover.choose(board.copy(), budget)
            if not isinstance(decision, MoveDecision):
                raise PlayerError(f"{mover.name}.choose 应返回 MoveDecision，"
                                  f"收到 {type(decision).__name__}")
            if ply < len(book) and decision.move != book[ply]:
                raise PlayerError(f"{mover.name} 在第 {ply} ply 应走 book 着法 {book[ply]}，"
                                  f"却走了 {decision.move}（第 {task.game} 局）")
            if decision.move not in board.legal_moves:
                raise PlayerError(f"{mover.name} 走了非法着法 {decision.move} @ {board.fen()}"
                                  f"（第 {task.game} 局）")
            board.push(decision.move)
            moves.append(decision.move.uci())
            decisions.append(decision)
            if dual:
                for p in (white, black):
                    yield from p.observe(board.copy(), decision.move)
            else:
                yield from player.observe(board.copy(), decision.move)
    finally:
        for p in ((player, player_b) if dual else (player,)):
            try:
                p.close()
            except Exception as e:  # noqa: close 失败不应掩盖真正的异常
                print(f"[selfplay] {getattr(p, 'name', p)}.close() 失败：{e}", file=sys.stderr)
    record = {
        "type": "game", "game": task.game, "book_id": task.book_id,
        "book_plies": min(len(book), len(moves)), "moves": moves,
        "result": verdict.result, "termination": verdict.termination,
        "plies": len(board.move_stack), "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    return record, board, decisions


def run_selfplay(cfg: SelfPlayConfig, make_player: Callable, sink,
                 progress: Optional[Callable[[dict, int], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 skip_games: Optional[set] = None,
                 make_player_b: Optional[Callable] = None) -> dict:
    """跑 cfg.games 局自对弈，逐局调 ``sink.on_game_end``，返回汇总。

    make_player 为 PlayerFactory（每局一个 Player）；跨局共享的状态（如 book ply 的搜索缓存）
    由引擎放在工厂里。``make_player_b`` 给出时走双引擎对局（两个工厂各执一方、颜色按局号
    分配，见 ``play_selfplay_game``）；为 None 时是既有的单引擎口径。
    """
    lines, dropped = (load_book_lines(cfg.openings, cfg.book_plies) if cfg.openings
                      else ([], 0))
    if cfg.openings and not lines:
        raise ValueError(f"开局文件 {cfg.openings} 没有有效开局")
    tasks = plan_selfplay(cfg, lines)
    if skip_games:                         # 续跑：已落盘的局跳过（每局只取决于全局局号）
        tasks = [t for t in tasks if t.game not in skip_games]
    referee = StandardReferee(max_plies=cfg.max_plies)
    budget = SearchBudget(simulations=cfg.simulations, add_noise=True)
    batcher = Batcher()
    if cfg.threads > 1:
        # 单进程多线程：共享一个 CUDA context（多进程的 context 切换反而更慢），
        # 各线程一个 Batcher 并行喂 GPU。on_done 会被多线程调用 ⇒ 用锁串行化收尾。
        pool = ThreadedCoroutinePool(cfg.threads, cfg.concurrency)
        done_lock: Optional[threading.Lock] = threading.Lock()
    else:
        pool = CoroutinePool(cfg.concurrency, batcher)
        done_lock = None
    totals = {"games": 0, "plies": 0, "book_plies": 0, "termination": {}}
    t0 = time.perf_counter()

    def _start(t):
        # Player 在 job 真正开始时才建（池子占一个坑起一局），与既有口径一致
        a = make_player()
        b = make_player_b() if make_player_b is not None else None
        return play_selfplay_game(t, a, referee, budget, cfg.seed, player_b=b)

    def _finalize(record, board, decisions):
        sink.on_game_end(record, board, decisions)
        totals["games"] += 1
        totals["plies"] += record["plies"]
        totals["book_plies"] += record["book_plies"]
        term = totals["termination"]
        term[record["termination"]] = term.get(record["termination"], 0) + 1
        if progress is not None:
            progress(record, totals["games"])

    def on_done(_gid, out):
        record, board, decisions = out
        if done_lock is not None:
            with done_lock:
                _finalize(record, board, decisions)
        else:
            _finalize(record, board, decisions)

    jobs = ((t.game, (lambda t=t: _start(t))) for t in tasks)
    pool.run(jobs, on_done, should_stop=should_stop or (lambda: False))
    elapsed = time.perf_counter() - t0
    if cfg.threads > 1:
        batch_stats: dict = {}
        for b in pool.batchers:
            for k, v in b.stats.as_dict().items():
                if isinstance(v, (int, float)):
                    batch_stats[k] = batch_stats.get(k, 0) + v
    else:
        batch_stats = batcher.stats.as_dict()
    return {**totals, "termination": dict(sorted(totals["termination"].items())),
            "games_planned": cfg.games, "opening_lines": len(lines),
            "opening_lines_dropped": dropped, "elapsed_s": round(elapsed, 1),
            "batch": batch_stats, "config": cfg.to_dict(),
            "dual_engine": make_player_b is not None}


# ------------------------------------------------------------------ 命令行

def run_selfplay_config(conf: dict, should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """``{"engine": EngineSpec, "engine_b": EngineSpec?, "selfplay": SelfPlayConfig, "sink": {...}}``。

    ``engine_b`` 可选：给出时跑双引擎对局（两个 Player 各执一方、颜色按局号分配，
    见 ``play_selfplay_game``）——联盟自对弈用它把历史冠军拉进来当对手。

    sink 工厂返回带 ``on_game_end`` 的对象；若它有 ``done_games()``，已落盘的局跳过（续跑）。
    ``selfplay.workers > 1`` 时走多进程分片（见 ``run_selfplay_workers``），``engine_b``
    原样透传给每个子进程。
    """
    from ..registry import EngineSpec, build_player_factory, load_object

    unknown = set(conf) - {"engine", "engine_b", "selfplay", "sink"}
    if unknown:
        raise ValueError(f"自对弈配置有未知字段 {sorted(unknown)}")
    spec = EngineSpec.from_dict(conf["engine"])
    spec_b = (EngineSpec.from_dict(conf["engine_b"]) if conf.get("engine_b") is not None
              else None)
    cfg = SelfPlayConfig(**conf["selfplay"])
    if cfg.workers > 1:
        return run_selfplay_workers(conf, cfg, spec, should_stop=should_stop)
    sink_conf = conf["sink"]
    sink = load_object(sink_conf["factory"], spec.root)(**sink_conf.get("kwargs", {}))
    done = sink.done_games() if hasattr(sink, "done_games") else set()
    make_player = build_player_factory(spec)
    if spec_b is None:
        make_player_b = None
    elif conf["engine"] == conf.get("engine_b"):
        # 镜像自对弈（engine_b 与 engine 是同一个引擎）：复用同一个工厂，两个 Player
        # 共享一个 evaluator。各建一个的话，两个 evaluator 对象同 model_key，Batcher 会以
        # "会被错误地拼进同一次前向"为由报错——那个检查是对的，只是这里本就不该建两个。
        make_player_b = make_player
    else:
        make_player_b = build_player_factory(spec_b)
    summary = run_selfplay(cfg, make_player, sink, should_stop=should_stop, skip_games=done,
                           make_player_b=make_player_b)
    summary["skipped_done"] = len(done & set(range(cfg.first_game, cfg.first_game + cfg.games)))
    close = getattr(sink, "close", None)
    if close:
        close()
    return summary


def _worker_sink_path(path: Path, i: int) -> Path:
    """``selfplay.sp.bin`` -> ``selfplay.w{i}.sp.bin``（结尾仍是 .sp.bin，
    loop 的 ``{selfplay_files}`` 用 ``*.sp.bin`` glob 收分片）。"""
    return path.with_name(path.name[:-len(".sp.bin")] + f".w{i}.sp.bin")


def run_selfplay_workers(conf: dict, cfg: SelfPlayConfig, spec,
                         should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """多进程自对弈：``games`` 按**连续区间**切给 workers 个子进程，各自写 ``.w{i}.sp.bin``。

    区间连续 ⇒ 全局局号不变 ⇒ 每局种子/开局与单进程一致（只有批组成差异，见模块说明）。
    续跑按各分片自己的 ``done_games`` 跳过。⚠ **改 workers 数量会让旧分片与新区间错位**
    （不同分片出现同一局号）——数量变了必须先归档旧分片再跑。

    父进程是"工头"：起子进程、收各自 summary、聚合返回；收到 SIGTERM 时先杀子进程再退。

    配置里可选的 ``engine_b``（双引擎）随 ``w = dict(conf)`` 原样进各子进程，由子进程的
    ``run_selfplay_config`` 校验并建两个工厂；子进程 cwd 取 A 方（``engine``）的 root，
    B 方的相对路径由它自己的工厂负责解析（``EngineSpec.root`` 会各自挂 sys.path）。
    """
    import signal
    import subprocess

    path = Path(dict(conf["sink"].get("kwargs", {}))["path"])
    base, rem = divmod(cfg.games, cfg.workers)
    procs = []
    start = cfg.first_game
    try:
        for i in range(cfg.workers):
            n = base + (1 if i < rem else 0)
            if n == 0:
                continue
            w = dict(conf)
            wsink = dict(conf["sink"])
            wsk = dict(wsink.get("kwargs", {}))
            wsk["path"] = str(_worker_sink_path(path, i))
            wsink["kwargs"] = wsk
            w["sink"] = wsink
            wcfg = dict(cfg.to_dict())
            wcfg.update(games=n, first_game=start, workers=1)
            w["selfplay"] = wcfg
            cpath = path.parent / f"{path.stem}.w{i}.json"
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(w, ensure_ascii=False, indent=1), encoding="utf-8")
            sfile = path.parent / f"{path.stem}.w{i}.summary.json"
            log = path.parent / f"{path.stem}.w{i}.log"
            fh = open(log, "w", encoding="utf-8")
            fh.write(f"$ {' '.join([sys.executable, '-m', 'Kit', 'selfplay', str(cpath)])}\n")
            fh.flush()
            p = subprocess.Popen(
                [sys.executable, "-m", "Kit", "selfplay", str(cpath),
                 "--summary-out", str(sfile)],
                cwd=str(spec.root), stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ))
            procs.append((i, p, fh, sfile, log, str(cpath)))
            start += n

        def _kill_children(*_a):
            for _i, p2, _fh, _s, _l, _c in procs:
                if p2.poll() is None:
                    p2.terminate()
            raise SystemExit(143)

        prev = signal.signal(signal.SIGTERM, _kill_children)
        try:
            summaries = []
            for i, p, fh, sfile, log, cpath in procs:
                rc = p.wait()
                fh.close()
                if rc != 0:
                    for _i2, p2, _fh2, _s2, _l2, _c2 in procs:
                        if p2.poll() is None:
                            p2.terminate()
                    raise RuntimeError(f"selfplay worker {i} 退出码 {rc}"
                                       f"（配置 {cpath}，日志 {log}）")
                summaries.append(json.loads(sfile.read_text(encoding="utf-8")))
        finally:
            signal.signal(signal.SIGTERM, prev)
    finally:
        for _i, p, fh, _s, _l, _c in procs:
            if p.poll() is None:
                p.terminate()
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass

    total = {"games": 0, "plies": 0, "book_plies": 0, "termination": {},
             "skipped_done": 0, "workers": len(procs)}
    for s in summaries:
        for k in ("games", "plies", "book_plies", "skipped_done"):
            total[k] += int(s.get(k, 0))
        for term, cnt in (s.get("termination") or {}).items():
            total["termination"][term] = total["termination"].get(term, 0) + int(cnt)
    return total


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="UniChessKit 自对弈")
    ap.add_argument("config", help="JSON：{\"engine\": EngineSpec, \"engine_b\": EngineSpec?, "
                                   "\"selfplay\": {...}, \"sink\": {...}}")
    ap.add_argument("--summary-out", default=None, help="汇总 JSON 另写到该文件（多进程 worker 用）")
    args = ap.parse_args(argv)
    # 路径一律绝对化：调用方（如 SSM 的 run_gpu_server_selfplay.py）以 import 根为 cwd 起
    # 子进程，相对路径会被解释到另一个目录上去（2026-10-11 踩过：worker 找不到 config.json）。
    conf = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    summary = run_selfplay_config(conf)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary_out:
        Path(args.summary_out).resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
