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

**只枚举前 N 代**（``enumerate_generations: 3``）：前 N 代照常枚举，选中变体的覆盖字段
记进 ``loop_state.json`` 的 ``train_variant``；第 N 代起直接用那份配置训练
（产物落 ``gen_XXXX/train/``，跳过全部筛选赛）。每代省下的就是 N 场筛选赛的钱。
不写这个字段 = 每代都枚举。

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
     "enumerate_generations": 可选，只枚举前 N 代（见「枚举搜索」一节）

模板占位符（字符串整体等于占位符时替换为对应值，可为列表；否则做子串替换）::

    {weights}          当前冠军权重（自对弈、训练初始化、arena 的 B 方）
    {candidate}        本代候选权重（仅 arena 的 A 方）
    {gen}              代号（整数）
    {gen_dir}          本代目录
    {selfplay_dir}     本代自对弈目录
    {selfplay_files}   最近 window 代（含本代）的全部自对弈分片路径列表

**启动时的配置体检**（只告警、不拦，写进日志/终端）：
循环不认识的 ``{token}``（拼错的占位符，dict 键里的占位符——键本来就不替换）；
用 ``train.variants`` 时 ``train.screen`` 与 ``arena.match`` 的开局库一边有一个没有
（``MatchConfig.openings`` 缺省 ``"bundled"``，一边一个 = 筛选赛与 arena 的开局分布不同）；
锁定配置来自已不在 ``train.variants`` 里的变体（网格换过了）。

状态文件 ``<out>/loop_state.json``：``{"generation", "phase", "champion", "history": [...]}``
（``phase == "search"`` 时带 ``variant`` / ``train_variant``，中断续跑靠它们找胜者路径）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from .. import IMPORT_ROOT
from ..runtime.locks import FileLock

PHASES = ("selfplay", "train", "search", "arena")

# 循环自己认识的模板占位符。别的花括号 token（如训练导出名里的 {step}）留给子进程解释；
# 拼错的占位符（{selfplay_dirr}）不会被替换，却会原样带进子进程配置里，直到要写文件
# 那一刻才炸——所以启动时对配置里的 token 做一次体检，只告警不拦。
PLACEHOLDERS = ("{weights}", "{candidate}", "{gen}", "{gen_dir}", "{selfplay_dir}",
                "{selfplay_files}")
_OTHER_TOKENS = ("{step}",)          # Kit/train 的 export.every_name 用
_TOKEN = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _merge(base, over):
    """递归合并：variant 只写要覆盖的键（如 ``{"optimizer": {"lr": 1e-5}}``）。"""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def _apply_overrides(base: dict, over: dict) -> dict:
    """训练模板 base + 覆盖字段 over；``schedule`` 整体替换而不是递归合并。

    ``schedule`` 是自含的小字典，合并会留下对方的键（例如基座
    ``{"kind":"onecycle","pct_start":0.25}`` 加上覆盖的 ``{"kind":"constant"}`` 就成了
    ``{kind: constant, pct_start: 0.25}``，``build_schedule`` 会以「未知参数」为由报错）。
    两个调用方（变体训练、枚举用尽后按锁定配置训练）必须是同一套语义。
    """
    base, over = copy.deepcopy(base), copy.deepcopy(over)
    base_sched, over_sched = base.pop("schedule", None), over.pop("schedule", None)
    conf = _merge(base, over)
    sched = over_sched if over_sched is not None else base_sched
    if sched is not None:                      # 都没有时不写这个键：保持 TrainConfig 缺省
        conf["schedule"] = copy.deepcopy(sched)
    return conf


def _unknown_tokens(obj, found=None) -> list:
    """配置里循环不认识的 ``{token}``（含字符串值、dict 键、列表元素）。

    dict 的键不做占位符替换，所以键里的占位符会整个失效；一并报出来。
    """
    if found is None:
        found = []
    if isinstance(obj, str):
        for tok in _TOKEN.findall(obj):
            if tok not in PLACEHOLDERS and tok not in _OTHER_TOKENS and tok not in found:
                found.append(tok)
    elif isinstance(obj, list):
        for x in obj:
            _unknown_tokens(x, found)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _unknown_tokens(k, found)
            _unknown_tokens(v, found)
    return found


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
                               "selfplay", "sink", "train", "export", "arena",
                               "enumerate_generations", "seed_base"}
        if unknown:
            raise ValueError(f"loop 配置有未知字段 {sorted(unknown)}")
        tokens = _unknown_tokens(conf)
        if tokens:
            warnings.warn(f"loop 配置里有循环不认识的占位符 {tokens}：循环只替换 "
                          f"{list(PLACEHOLDERS)}（dict 的键不做替换），拼错的占位符会"
                          f"原样带进子进程配置里", stacklevel=2)
        self.conf = conf
        out = Path(conf["out"])
        self.out = out if out.is_absolute() else (base_dir / out).resolve()
        self.python = python
        self.state_path = self.out / "loop_state.json"
        self.enumerate_generations = conf.get("enumerate_generations")
        if self.enumerate_generations is not None:
            self.enumerate_generations = int(self.enumerate_generations)
            if self.enumerate_generations < 0:
                raise ValueError("enumerate_generations 应 >= 0")
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
            self._warn_screen_arena_openings(screen, conf["arena"]["match"])

    def _warn_screen_arena_openings(self, screen: dict, arena_match: dict) -> None:
        """筛选赛与最终 arena 的开局库必须一致（一边给了另一边没给就喊出来）。

        ``MatchConfig.openings`` 缺省是 ``"bundled"``（Kit 的 34 条开局），而 arena.match
        通常显式指向合成开局库：这时 ``train.screen`` 不写 ``openings`` 就会**静默**用
        bundled 跑筛选赛。2026-09-27 loop_p4_v2 gen 0 正是如此——旧进程按内存里的旧配置
        写出 ``openings: "bundled"``，十场筛选赛实际跑在 34 条开局上（duplicate_rate
        0.27~0.375、只摊到 32 条线路），选出的是「按别的开局分布」最优的变体，arena 却
        用合成库，两边的结论互不相干。两者差一个，筛选就等于白跑。
        """
        screen_book, arena_book = screen.get("openings"), arena_match.get("openings")
        if bool(screen_book) != bool(arena_book):
            warnings.warn(f"train.screen.openings={screen_book!r} 与 arena.match.openings="
                          f"{arena_book!r} 不一致（MatchConfig 缺省 bundled）：筛选赛与最终"
                          f"arena 的开局分布不同，筛选选出的变体未必适合 arena", stacklevel=2)

    # ---------------------------------------------------------------- 状态
    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"generation": 0, "phase": "selfplay", "champion": str(self.conf["initial"]),
                "history": []}

    def gen_dir(self, g: int) -> Path:
        return self.out / f"gen_{g:04d}"

    def candidate_path(self, g: int, variant: Optional[str]) -> str:
        """本代候选权重的落地路径。

        search 代的胜者在 ``train_<label>/`` 下（``phase_search`` 一行挑出来）；
        非 search 代（``enumerate_generations`` 用尽后）以及还没选变体时在 ``train/``
        下（``phase_train`` 的产物）。两者不可混用——续跑时 ``mapping`` 只会给后者。
        """
        export = self.conf.get("export", "final.pt")
        if variant and self.uses_search(g):
            return str(self.gen_dir(g) / f"train_{variant}" / export)
        return str(self.gen_dir(g) / "train" / export)

    def mapping(self, g: int, champion: str) -> dict:
        window = int(self.conf.get("window", 1))
        files = []
        for h in range(max(0, g - window + 1), g + 1):
            d = self.gen_dir(h) / "selfplay"
            files += sorted(str(p) for p in d.glob("*.sp.bin")) if d.exists() else []
        gd = self.gen_dir(g)
        return {"{weights}": champion, "{gen}": g, "{gen_dir}": str(gd),
                "{selfplay_dir}": str(gd / "selfplay"), "{selfplay_files}": files,
                "{candidate}": self.candidate_path(g, None)}

    # ------------------------------------------------------------ 枚举搜索
    def uses_search(self, g: int) -> bool:
        """本代是否跑枚举搜索：有 variants，且代数没超过 ``enumerate_generations``。

        ``enumerate_generations = None``（默认）= 每代都枚举；``= 3`` = 只枚举前 3 代，
        之后按上一代选中变体的配置训练（不再打筛选赛）。
        """
        if not self.variants:
            return False
        if self.enumerate_generations is None:
            return True
        return g < self.enumerate_generations

    def _variant_train_conf(self, variant: dict, m: dict) -> dict:
        """base train 模板与 variant 递归合并后做占位符替换；``out`` 按 label 分开。

        ``schedule`` 是**整体替换**而不是合并：它是自含的小字典，合并会留下对方的键
        （例如基座 ``{"kind":"onecycle","pct_start":0.25}`` 加上变体的
        ``{"kind":"constant"}`` 就成了 ``{kind: constant, pct_start: 0.25}``，
        ``build_schedule`` 会以「未知参数」为由报错）。锁定配置那一侧同理，见
        ``_apply_overrides``。
        """
        over = copy.deepcopy(variant)
        label = over.pop("label")
        conf = _apply_overrides(self._base_train(), over)
        conf["out"] = str(Path(m["{gen_dir}"]) / f"train_{label}")
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
            raise RuntimeError(f"{subcmd} 子进程退出码 {rc}（配置 {cfg_path}，"
                               f"日志 {log_path}）")

    # ---------------------------------------------------------------- 每代种子
    def _gen_seed(self, g: int, which: int) -> Optional[int]:
        """按代派生种子；配置没给 ``seed_base`` 时返回 None（保持原行为）。

        为什么需要它：arena 连着六代用同一个 ``match.seed``，开局与配色
        都由那个固定流决定，等于每次都在**同一批固定局面**上判决——
        那是测试集不是随机样本，系统性偏袒哪一边都不奇怪。
        训练的固定 seed 也让静态语料的洗牌顺序每代一模一样
        （``Kit/planes19/task._mix_batches`` 只认 data 的 ``seed``）。

        派生是确定性的：中断续跑同一代会算出同一个种子，续跑不会换种子；
        跨代则必然不同。换了 ``seed_base`` 就等于换整套种子序列，
        想完全复现某一代就把它固定回去。
        """
        base = self.conf.get("seed_base")
        if base is None:
            return None
        return (int(base) + which * 7919 + g * 104729) % (2 ** 31 - 1)

    def _subst_seeds(self, g: int) -> None:
        """把本代种子写回 self.conf 的 train / selfplay / arena.match 三处（原地）。

        幂等：同一代重复调用得到同一结果，所以 ``run()`` 每轮开头调一次即可。
        """
        for which, keys in enumerate((("train",), ("selfplay",), ("arena", "match")), start=1):
            seed = self._gen_seed(g, which)
            if seed is None:
                return
            node = self.conf
            for k in keys:
                node = node.get(k)
                if not isinstance(node, dict):
                    break
            if isinstance(node, dict):
                node["seed"] = seed

    def phase_selfplay(self, g: int, m: dict) -> None:
        gd = self.gen_dir(g)
        games = int(self.conf["games"])
        sp = dict(self.conf["selfplay"], games=games, first_game=g * games)
        conf = {"engine": _subst(self.conf["engine"], m), "selfplay": sp,
                "sink": _subst(self.conf["sink"], m)}
        path = gd / "selfplay.json"
        _write_json(path, conf)
        self._run("selfplay", path, gd / "selfplay.log")

    def _fixed_train_conf(self, m: dict, overrides: dict) -> dict:
        """枚举次数用完后，按锁定的变体配置训练（无筛选赛，产物落 ``gen_XXXX/train/``）。"""
        conf = _apply_overrides(self._base_train(), overrides)
        conf["out"] = str(Path(m["{gen_dir}"]) / "train")
        return _subst(conf, m)

    def phase_train(self, g: int, m: dict, overrides: Optional[dict] = None) -> None:
        """单路径训练：``overrides`` 给出时按锁定的变体配置训练（枚举已结束）。

        不检查候选是否已存在、每次都拉起训练子进程：``Trainer`` 自己会按
        ``latest.pt`` 的 step 与配置哈希判断续跑还是重训，训完再按 ``export``
        落盘（2026-09-30 补：``step >= steps`` 的提前返回路径也要导出，
        否则 loop 重入本方法时会拿不到候选文件）。
        """
        gd = self.gen_dir(g)
        conf = (self._fixed_train_conf(m, overrides) if overrides
                else _subst(self._base_train(), m))
        conf["out"] = str(gd / "train")
        path = gd / "train.json"
        _write_json(path, conf)
        self._run("train", path, gd / "train.log")
        cand = Path(m["{candidate}"])
        if not cand.exists():
            raise RuntimeError(f"训练结束但没有导出 {cand}")

    def _base_train(self) -> dict:
        """train 模板去掉枚举搜索专用字段（variants / screen 不是 TrainConfig 的键）。"""
        base = copy.deepcopy(self.conf["train"])
        base.pop("variants", None)
        base.pop("screen", None)
        return base

    def _selected_variant_config(self, g: int) -> dict:
        """本代选中变体的覆盖字段（去掉 label），供后续代锁定使用。"""
        res = self._search_results(g)
        label = res.get("_selected")
        if not label:
            return {}
        return {k: v for k, v in res[label]["config"].items()}

    def _drop_stale_results(self, results: dict, done_path: Path) -> dict:
        """丢掉不属于当前 ``train.variants`` 网格的结果条目。

        2026-09-27 loop_p4_v2 的实况：gen 0 的 ``search.json`` 是旧网格（10 个恒定 lr
        变体）写的，换成新网格的配置再重启 loop → 新 label 不在 results 里，得全部重跑
        （1.5h GPU）；跑完 ``len(results)`` = 旧 10 + 新 8 = 18 ≠ 8，撞「变体结果不齐」
        报错。而旧 label 还留在文件里，之后再重启也永远过不了这个校验——loop 被永久卡死
        在这一代，只能手工删 ``search.json``（把新变体已跑出的成绩一起丢掉）。

        网格换过了，旧 label 的成绩对新网格没有意义，扔掉重跑才对。
        """
        labels = {v["label"] for v in self.variants}
        stale = sorted(k for k in results if k not in labels)
        if not stale:
            return results
        for k in stale:
            results.pop(k)
        _write_json(done_path, results)
        warnings.warn(f"search.json 里有 {stale} 条不在 train.variants 里的结果"
                      f"（网格换过了）：已丢弃并按新网格重跑这些变体", stacklevel=2)
        return results

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
        results = self._drop_stale_results(results, done_path)
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
        # 判决必须可复现：arena 两侧**强制确定性选着**（temperature=0）。
        # ``engine.kwargs`` 是自对弈与 arena 共用的，自对弈可能需要
        # temperature>0 来让对局有变化（否则 argmax(访问数) + ε=0 会让对局
        # 塌缩成确定性树），但那不能渗进判决——否则同一局能下出不同结果，
        # SPRT 的方差白涨。match 的 SearchBudget 默认 add_noise=False，
        # 所以 Dirichlet 本来就不进 arena，只有 temperature 需要在这里挡掉。
        for side in (a, b):
            side["kwargs"]["temperature"] = 0
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
                # 每代重新掷种子（opt-in：配了 seed_base 才生效）。放在循环体最前面，
                # 保证 train / selfplay / arena.match 三个子进程都吃到本代的种子。
                self._subst_seeds(g)
                self.gen_dir(g).mkdir(parents=True, exist_ok=True)
                m = self.mapping(g, st["champion"])
                # 续跑纠正候选路径：中断后 phase 停在 arena 时，mapping 只会给
                # phase_train 用的 train/ 路径，而 search 代的胜者在 train_<label>/ 下。
                # （2026-09-27 实测：在 arena 阶段重启 loop → FileNotFoundError:
                #   .../gen_0000/train/final.pt）
                # variant 优先读状态文件；老版本 / 手改过的状态文件可能没写这个字段，
                # 退回本代 search.json 的 _selected（胜者的权威记录在那）。
                if st["phase"] == "arena":
                    variant = st.get("variant") or self._search_results(g).get("_selected")
                    m = {**m, "{candidate}": self.candidate_path(g, variant)}
                t0 = time.time()
                if st["phase"] == "selfplay":
                    self.phase_selfplay(g, m)
                    st["phase"] = "search" if self.uses_search(g) else "train"
                    _write_json(self.state_path, st)
                    m = self.mapping(g, st["champion"])      # 本代分片现在才存在
                if st["phase"] == "search":
                    label = self._search_all(g, m)
                    m = {**m, "{candidate}": self.candidate_path(g, label)}
                    st["variant"] = label
                    st["train_variant"] = self._selected_variant_config(g)
                    st["phase"] = "arena"
                    _write_json(self.state_path, st)
                if st["phase"] == "train":
                    # 枚举已结束的代：按上一代选中变体的配置训练（search 里不再跑筛选赛）
                    locked = st.get("train_variant") or None
                    if locked:
                        src = st.get("variant")
                        if src is not None and src not in {v["label"] for v in self.variants}:
                            warnings.warn(
                                f"gen {g} 的锁定配置来自变体 {src!r}，但它已不在 train.variants"
                                f" 里（网格换过了）：仍按旧网格胜者 {sorted(locked)} 训练，"
                                f"不重新枚举。要用新网格就删掉 loop_state.json 的 "
                                f"train_variant 字段再续跑", stacklevel=2)
                    self.phase_train(g, m, locked)
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
                    # 本代实际用的种子一并留档：换了 seed_base 后要能对上是哪一套序列
                    seeds = {k: self.conf[k]["seed"] for k in ("train", "selfplay")
                             if isinstance(self.conf.get(k), dict) and "seed" in self.conf[k]}
                    mk = (self.conf.get("arena") or {}).get("match") or {}
                    if "seed" in mk:
                        seeds["match"] = mk["seed"]
                    if seeds:
                        rec["seeds"] = seeds
                    if self.variants:
                        if self.uses_search(g):
                            rec["variant"] = st.get("variant")
                        else:
                            # 锁定代没跑枚举：variant 是上一次 search 的胜者，
                            # 不能冒充本代选出来的；把锁定配置一并记下来备查
                            rec["variant"] = None
                            rec["locked_train_variant"] = st.get("train_variant") or None
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
