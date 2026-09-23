# UniChessKit AGENTS.md

> 面向 AI 编码 agent。项目群总览与隐私纪律见上级目录 `../AGENTS.md`（最高优先级规则同样适用）。

## 定位

S/T/R 共享的管线库。设计与分阶段计划：`~/.claude/plans/s-ply-dazzling-hearth.md`（P1 = 本仓库骨架 + T/R 接入）。
依赖只能自上而下：`api` ← `rules`/`stats`/`runtime` ← `search` ← `players` ← `pipelines`。
`api`、`rules`、`stats`、`runtime`、`search` 不得 import torch；引擎专有编码不得进入 kit（T/R 共用的放 `contrib/planes19`）。

## 不变量（每条都对应过去某个仓库踩过的坑，改动前先看对应测试）

- 模拟撞终局节点也计入预算（R 访问数暴涨到 7661 的 bug）——`test_puct.test_terminal_sims_consume_budget`。
- 树复用前核对 EPD，换棋路必须重建——`test_players.test_tree_not_reused_for_other_game`。
- 确定性双方逐局重复必须在汇总里暴露（S `365685b`）——`duplicate_rate`。
- worker 静默退出算错误（S `d87371c`）——`test_runtime`。
- 出错整批停止，不留"运行中"假象；非法着法抛 `PlayerError`。
- 裁决统一 `claim_draw=True`；超 `max_plies` 记 truncated，按和棋计分但单独统计。
- 续跑拒绝配置哈希不同的结果文件；中间行损坏报错，末尾半行丢弃。
- 配置哈希只用 `EngineSpec.identity()`（不含 `runtime`）；会改变结果的参数必须放 `kwargs`——`test_registry`。
- `PUCT` 与 R `search/mcts.py` 逐节点一致（parity 测试），改搜索先让 parity 失败有理由。

## 开发

- 本机：`C:\ProgramData\miniconda3\python.exe -m unittest discover -s tests -t .`（设 `PYTHONIOENCODING=utf-8`）。
- 远端：`/home/jeefy/miniconda3/envs/unichess/bin/python`，同样用 unittest。
- 提交信息中文；结果文件、权重、日志一律不入库（`.gitignore` 已配）。
