"""换代循环：自对弈 → 训练 → arena 换代，每代状态落盘，随时可中断续跑。

    python -m Kit loop <loop.json>

每一代 g（目录 ``<out>/gen_XXXX/``）三个阶段，各自作为**子进程**运行（显存随进程释放，
阶段内部各自可续跑：自对弈 sink 跳过已写的局、训练器从 ``latest.pt`` 续、match 结果文件续跑）：

1. **selfplay**：冠军权重自对弈 ``games`` 局 → ``gen_XXXX/selfplay/``。全局局号从
   ``g * games`` 起，不同代的随机数流与开局分配互不重复。
2. **train**：从冠军权重初始化训练（训练配置模板里的占位符被替换，见下）→ ``gen_XXXX/train/``。
3. **arena**：候选（训练导出的权重）对冠军 → ``gen_XXXX/arena.jsonl``；
   ``gate.kind = "sprt"``（match 配置须带 sprt，裁决 H1 才换代）或 ``"score"``（``score_a ≥ min_score``）。

配置::

    {"out": "runs/loop_p4", "generations": 10, "initial": "<冠军权重路径>",
     "games": 2000, "window": 4,
     "engine":   EngineSpec 模板（自对弈与 arena 双方共用），
     "selfplay": SelfPlayConfig 字段（games / first_game 由循环填），
     "sink":     {"factory": ..., "kwargs": {...}}   （path 等可用占位符）,
     "train":    TrainConfig 模板（out 由循环填）,
     "export":   "final.pt"   （训练目录里作为候选的导出文件名）,
     "arena":    {"match": MatchConfig 字段, "gate": {"kind": "sprt"} | {"kind": "score", "min_score": 0.55}}}

模板占位符（字符串整体等于占位符时替换为对应值，可为列表；否则做子串替换）::

    {weights}          当前冠军权重（自对弈、训练初始化、arena 的 B 方）
    {candidate}        本代候选权重（仅 arena 的 A 方）
    {gen}              代号（整数）
    {gen_dir}          本代目录
    {selfplay_dir}     本代自对弈目录
    {selfplay_files}   最近 window 代（含本代）的全部自对弈分片路径列表

状态文件 ``<out>/loop_state.json``：``{"generation", "phase", "champion", "history": [...]}``。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .. import IMPORT_ROOT
from ..runtime.locks import FileLock

PHASES = ("selfplay", "train", "arena")


def _subst(obj, mapping: dict):
    if isinstance(obj, str):
        if obj in mapping:
            return copy.deepcopy(mapping[obj])
        for k, v in mapping.items():
            if isinstance(v, (str, int, float)) and k in obj:
                obj = obj.replace(k, str(v))
        return obj
    if isinstance(obj, list):
        return [_subst(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst(v, mapping) for k, v in obj.items()}
    return obj


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class Loop:
    def __init__(self, conf: dict, base_dir: Path, *, python: str = sys.executable):
        unknown = set(conf) - {"out", "generations", "initial", "games", "window", "engine",
                               "selfplay", "sink", "train", "export", "arena"}
        if unknown:
            raise ValueError(f"loop 配置有未知字段 {sorted(unknown)}")
        self.conf = conf
        out = Path(conf["out"])
        self.out = out if out.is_absolute() else (base_dir / out).resolve()
        self.python = python
        self.state_path = self.out / "loop_state.json"
        gate = conf["arena"].get("gate", {"kind": "sprt"})
        if gate["kind"] not in ("sprt", "score"):
            raise ValueError("arena.gate.kind 应为 sprt 或 score")
        if gate["kind"] == "sprt" and not conf["arena"]["match"].get("sprt"):
            raise ValueError("gate = sprt 时 arena.match 必须配置 sprt")

    # ---------------------------------------------------------------- 状态
    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"generation": 0, "phase": "selfplay", "champion": str(self.conf["initial"]),
                "history": []}

    def gen_dir(self, g: int) -> Path:
        return self.out / f"gen_{g:04d}"

    def mapping(self, g: int, champion: str) -> dict:
        window = int(self.conf.get("window", 1))
        files = []
        for h in range(max(0, g - window + 1), g + 1):
            d = self.gen_dir(h) / "selfplay"
            files += sorted(str(p) for p in d.glob("*.sp.bin")) if d.exists() else []
        gd = self.gen_dir(g)
        return {"{weights}": champion, "{gen}": g, "{gen_dir}": str(gd),
                "{selfplay_dir}": str(gd / "selfplay"), "{selfplay_files}": files,
                "{candidate}": str(gd / "train" / self.conf.get("export", "final.pt"))}

    # ---------------------------------------------------------------- 子进程
    def _run(self, subcmd: str, cfg_path: Path, log_path: Path, extra=()) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(IMPORT_ROOT)] +
                                            [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                             if p])
        cmd = [self.python, "-m", "Kit", subcmd, str(cfg_path), *extra]
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=str(self.out)).returncode
        if rc != 0:
            raise RuntimeError(f"{subcmd} 子进程退出码 {rc}，见 {log_path}")

    def phase_selfplay(self, g: int, m: dict) -> None:
        gd = self.gen_dir(g)
        games = int(self.conf["games"])
        sp = dict(self.conf["selfplay"], games=games, first_game=g * games)
        conf = {"engine": _subst(self.conf["engine"], m), "selfplay": sp,
                "sink": _subst(self.conf["sink"], m)}
        path = gd / "selfplay.json"
        _write_json(path, conf)
        self._run("selfplay", path, gd / "selfplay.log")

    def phase_train(self, g: int, m: dict) -> None:
        gd = self.gen_dir(g)
        conf = _subst(self.conf["train"], m)
        conf["out"] = str(gd / "train")
        path = gd / "train.json"
        _write_json(path, conf)
        self._run("train", path, gd / "train.log")
        cand = Path(m["{candidate}"])
        if not cand.exists():
            raise RuntimeError(f"训练结束但没有导出 {cand}")

    def phase_arena(self, g: int, m: dict) -> dict:
        gd = self.gen_dir(g)
        a = _subst(self.conf["engine"], {**m, "{weights}": m["{candidate}"]})
        a["label"] = f"gen{g}"
        b = _subst(self.conf["engine"], m)
        b["label"] = "champion"
        path = gd / "arena.json"
        _write_json(path, {"a": a, "b": b, "match": self.conf["arena"]["match"]})
        self._run("match", path, gd / "arena.log", ("--out", str(gd / "arena.jsonl"), "--quiet"))
        return json.loads((gd / "arena.jsonl.summary.json").read_text(encoding="utf-8"))

    def promote(self, summary: dict) -> bool:
        gate = self.conf["arena"].get("gate", {"kind": "sprt"})
        if gate["kind"] == "sprt":
            return (summary.get("sprt") or {}).get("verdict") == "H1"
        score = summary.get("score_a")
        return score is not None and score >= float(gate.get("min_score", 0.55))

    # ---------------------------------------------------------------- 主循环
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.out / "loop.lock")
        if not lock.acquire(timeout=0):
            raise RuntimeError(f"{self.out} 正被另一个 loop 进程使用")
        try:
            st = self.load_state()
            _write_json(self.out / "loop_config.json", self.conf)
            while st["generation"] < int(self.conf["generations"]):
                g = st["generation"]
                self.gen_dir(g).mkdir(parents=True, exist_ok=True)
                m = self.mapping(g, st["champion"])
                t0 = time.time()
                if st["phase"] == "selfplay":
                    self.phase_selfplay(g, m)
                    st["phase"] = "train"
                    _write_json(self.state_path, st)
                    m = self.mapping(g, st["champion"])      # 本代分片现在才存在
                if st["phase"] == "train":
                    self.phase_train(g, m)
                    st["phase"] = "arena"
                    _write_json(self.state_path, st)
                if st["phase"] == "arena":
                    summary = self.phase_arena(g, m)
                    promoted = self.promote(summary)
                    rec = {"generation": g, "promoted": promoted, "candidate": m["{candidate}"],
                           "champion_before": st["champion"], "score_a": summary.get("score_a"),
                           "elo": summary.get("elo"), "games": summary.get("games"),
                           "sprt": (summary.get("sprt") or {}).get("verdict"),
                           "sec": round(time.time() - t0, 1)}
                    if promoted:
                        st["champion"] = m["{candidate}"]
                    st["history"].append(rec)
                    st["generation"] = g + 1
                    st["phase"] = "selfplay"
                    _write_json(self.state_path, st)
                    with open(self.out / "loop.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(json.dumps(rec, ensure_ascii=False), flush=True)
            return st
        finally:
            lock.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="UniChessKit 换代循环")
    ap.add_argument("config")
    args = ap.parse_args(argv)
    path = Path(args.config).resolve()
    conf = json.loads(path.read_text(encoding="utf-8"))
    st = Loop(conf, path.parent).run()
    print(json.dumps({"generation": st["generation"], "champion": st["champion"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
