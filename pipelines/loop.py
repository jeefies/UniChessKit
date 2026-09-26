"""换代循环：自对弈 → 训练 → arena 换代，每代状态落盘，随时可中断续跑。

    python -m Kit loop <loop.json>

每一代 g（目录 ``<out>/gen_XXXX/``）三个阶段，各自作为**子进程**运行（显存随进程释放，
阶段内部各自可续跑：自对弈 sink 跳过已写的局、训练器从 ``latest.pt`` 续、match 结果文件续跑）：

1. **selfplay**：冠军权重自对弈 ``games`` 局 → ``gen_XXXX/selfplay/``。全局局号从
   ``g * games`` 起，不同代的随机数流与开局分配互不重复。
2. **train** 或 **search**：默认只有一个训练配置模板（``gen_XXXX/train/``）。
   若 ``train.variants`` 非空则改为**枚举搜索**，见下。
3. **arena**：候选（胜出变体导出的权重）对冠军 → ``gen_XXXX/arena.jsonl``；
   ``gate.kind = "sprt"``（match 配置须带 sprt，裁决 H1 才换代）或 ``"score"``（``score_a ≥ min_score``）。

**枚举搜索**（``train.variants`` 非空时启用，取代单一训练）::

    "train": {<TrainConfig 模板>, "variants": [{"label": "...", ...覆盖字段...}, ...],
              "screen": {"pairs": 64, ...MatchConfig 字段...}}

逐个变体：先用模板与 variant **递归合并**（variant 只写要改的键，如
``{"optimizer": {"lr": 1e-5}, "steps": 800}``）→ 训练到 ``gen_XXXX/train_<label>/``
→ 与当前冠军跑一场筛选赛 → ``gen_XXXX/screen_<label>.jsonl``。
全部跑完后按 **筛选赛 score_a** 取最高者（同分比 elo、再比局数），
它的导出成为本代候选，进最终 arena；``rec["search"]`` 里记录所有变体的成绩。
续跑粒度到变体：训练导出与筛选赛汇总都在就跳过，中断不重跑已花的 GPU。

注意（胜者诅咒）：筛选用小赛场选最优，选出来的那个的 score_a **系统性偏高**，
所以最终能否换代仍由 ``arena`` 的完整预算判定，不认筛选赛的分数。

配置::

     {"out": "runs/loop_p4", "generations": 10, "initial": "<冠军权重路径>",
      "games": 2000, "window": 4,
      "engine":   EngineSpec 模板（自对弈与 arena 双方共用），
      "selfplay": SelfPlayConfig 字段（games / first_game 由循环填），
      "sink":     {"factory": ..., "kwargs": {...}}   （path 等可用占位符）,
      "train":    TrainConfig 模板（out 由循环填）；variants 非空时进入枚举搜索,
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

PHASES = ("selfplay", "train", "search", "arena")


def _merge(base, over):
    """递归合并：variant 只写要覆盖的键（如 ``{"optimizer": {"lr": 1e-5}}``）。"""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


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
        self.variants = list(conf["train"].get("variants") or [])
        if self.variants:
            labels = [v.get("label") for v in self.variants]
            if any(not lab for lab in labels):
                raise ValueError("train.variants 每项都要有 label")
            if len(set(labels)) != len(labels):
                raise ValueError(f"train.variants 的 label 重复：{labels}")
            screen = conf["train"].get("screen") or {}
            if not screen.get("pairs"):
                raise ValueError("用 variants 时必须给 train.screen.pairs（候选对冠军的筛选赛场数）")

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

    # ------------------------------------------------------------ 枚举搜索
    def _variant_train_conf(self, variant: dict, m: dict) -> dict:
        """base train 模板与 variant 递归合并后做占位符替换；``out`` 按 label 分开。"""
        base = copy.deepcopy(self.conf["train"])
        base.pop("variants", None)
        base.pop("screen", None)
        conf = _merge(base, variant)
        conf.pop("label", None)
        conf["out"] = str(Path(m["{gen_dir}"]) / f"train_{variant['label']}")
        return _subst(conf, m)

    def _screen_conf(self, m: dict, label: str) -> dict:
        """一个候选对当前冠军的筛选赛：a=候选，b=冠军，match 取自 ``train.screen``。"""
        gd = Path(m["{gen_dir}"])
        cand = gd / f"train_{label}" / self.conf.get("export", "final.pt")
        a = _subst(self.conf["engine"], {**m, "{weights}": str(cand)})
        a["label"] = label
        b = _subst(self.conf["engine"], m)
        b["label"] = "champion"
        return {"a": a, "b": b, "match": dict(self.conf["train"]["screen"])}

    # ---------------------------------------------------------------- 子进程
    def _search_results(self, g: int) -> dict:
        """本代枚举搜索的结果表（含 ``_selected``）；没有则返回空。"""
        path = self.gen_dir(g) / "search.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

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

    def _search_all(self, g: int, m: dict) -> str:
        """逐个变体：训练 → 对冠军筛选赛 → 记录结果。返回得分最高变体的 label。

        每个变体的产物落在 ``gen_XXXX/train_<label>/`` 与 ``gen_XXXX/screen_<label>.jsonl``；
        两者都存在即视为已做完（续跑跳过），中断后不必重跑已花的 GPU。
        """
        gd = self.gen_dir(g)
        gd.mkdir(parents=True, exist_ok=True)
        done_path = gd / "search.json"
        results = (json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists()
                   else {})
        results.pop("_selected", None)
        for variant in self.variants:
            label = variant["label"]
            if label in results:
                continue
            train_dir = gd / f"train_{label}"
            export = train_dir / self.conf.get("export", "final.pt")
            if not export.exists():
                path = gd / f"train_{label}.json"
                _write_json(path, self._variant_train_conf(variant, m))
                self._run("train", path, gd / f"train_{label}.log")
            if not export.exists():
                raise RuntimeError(f"变体 {label} 训练结束但没有导出 {export}")
            scr = gd / f"screen_{label}.jsonl"
            if not scr.with_name(scr.name + ".summary.json").exists():
                path = gd / f"screen_{label}.json"
                _write_json(path, self._screen_conf(m, label))
                self._run("match", path, gd / f"screen_{label}.log",
                          ("--out", str(scr), "--quiet"))
            summary = json.loads(scr.with_name(scr.name + ".summary.json")
                                 .read_text(encoding="utf-8"))
            rec = {"label": label,
                   "config": {k: v for k, v in variant.items() if k != "label"},
                   "games": summary.get("games"), "score_a": summary.get("score_a"),
                   "elo": summary.get("elo"), "a_wins": summary.get("a_wins"),
                   "draws": summary.get("draws"), "b_wins": summary.get("b_wins"),
                   "distinct_games": summary.get("distinct_games"),
                   "elapsed_s": summary.get("elapsed_s")}
            train_log = train_dir / "train.jsonl"
            if train_log.exists():
                rows = [json.loads(l) for l in
                        train_log.read_text(encoding="utf-8").splitlines() if '"loss"' in l]
                if rows:
                    rec["train"] = {"steps": len(rows), "first_loss": rows[0]["loss"],
                                    "last_loss": rows[-1]["loss"],
                                    "last_policy": rows[-1].get("policy"),
                                    "last_value": rows[-1].get("value")}
            results[label] = rec
            _write_json(done_path, results)
        if len(results) != len(self.variants):
            raise RuntimeError(f"变体结果不齐：有 {len(results)}，应有 {len(self.variants)}")
        best = max(results.values(),
                   key=lambda r: (r["score_a"] or 0.0, r["elo"] or 0.0, r["games"] or 0))
        _write_json(done_path, {**results, "_selected": best["label"]})
        return best["label"]

    def phase_search(self, g: int, m: dict) -> str:
        """跑完所有变体并选出一个；``{candidate}`` 改指获胜变体的导出。"""
        label = self._search_all(g, m)
        gd = self.gen_dir(g)
        cand = gd / f"train_{label}" / self.conf.get("export", "final.pt")
        self.selected_variant = label
        return str(cand)

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
                    st["phase"] = "search" if self.variants else "train"
                    _write_json(self.state_path, st)
                    m = self.mapping(g, st["champion"])      # 本代分片现在才存在
                if st["phase"] == "search":
                    label = self._search_all(g, m)
                    picked = self.gen_dir(g) / f"train_{label}" / self.conf.get("export",
                                                                               "final.pt")
                    m = {**m, "{candidate}": str(picked)}
                    st["variant"] = label
                    st["phase"] = "arena"
                    _write_json(self.state_path, st)
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
                    if self.variants:
                        rec["variant"] = st.get("variant")
                        rec["search"] = {k: {"score_a": v["score_a"], "elo": v["elo"],
                                             "games": v["games"]}
                                         for k, v in self._search_results(g).items()
                                         if not k.startswith("_")}
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
