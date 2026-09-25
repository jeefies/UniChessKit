# UniChessKit AGENTS.md

> 面向 AI 编码 agent。项目群总览与隐私纪律见上级目录 `../AGENTS.md`（最高优先级规则同样适用）。

## 角色

维护 S/T/R 共享的后端管线库（包名 `Kit`），向外提供辅助模型推理的接口与功能。
扁平布局（2026-09-25）后，本仓库**根目录就是包**，import 根是 `~/UniChess`。
不修改 `SSM/`、`ResNet/`、`Transformer/` 的内容，只通过 `EngineSpec` + `PlayerFactory`
契约消费其能力；确实需要引擎侧配合时，改完两边都要跑对方的测试。

## 定位

分层依赖（`api` 不依赖 torch）：`api` ← `rules` / `stats` / `runtime` ← `search` ←
`players` ← `pipelines`。`planes19` 放 T/R 共用的编码（引擎专有代码不得进入 kit，
S 的编码在 `SSM/` 自己那边）。

## 不变式（每条都对应过去某个仓库踩过的坑，改动前先看对应测试）

- 模拟撞终局节点也计入预算（R 访问数暴涨到 7661 的 bug）——`test_puct.test_terminal_sims_consume_budget`。
- 树复用前核对 EPD，换棋路必须重建——`test_players.test_tree_not_reused_for_other_game`。
- 确定性双方逐局重复必须在汇总里暴露——`duplicate_rate`。
- worker 静默退出算错误——`test_runtime`。
- 出错整批停止，不留"运行中"假象；非法着法抛 `PlayerError`。
- 装死判定统一 `claim_draw=True`；超 `max_plies` 记 `truncated`，按和棋计分但单独统计。
- 续跑拒绝配置哈希不同的结果文件；中间行损坏报错，不吞尾半行。
- 配置哈希只用 `EngineSpec.identity()`（不含 `runtime`）；会改结构的参数必须进 `kwargs`——`test_registry`。
- `PUCT` 与 R `search/mcts.py` 逐节点一致（parity 测试）。**R 已扁平化**，旧引用路径
  `unichess_r/search/mcts.py` 已不存在；parity 的依据改为 `ResNet/` 仓的 git 历史与该测试。
  改搜索先解释 parity 为什么没失败。
- `PUCTCpp` 与 `PUCT` 整树逐位一致——`test_puct_cpp`（着法顺序、整树对照、整层对照）。
  改 `PUCT` 必须同步改 `search/_native/puct_native.cpp`，否则 parity 测试会失败；
  C++ 编译失败必须报错，不得静默回退 Python。

## 环境与依赖

- Python >= 3.10；依赖 `chess>=1.9`、`numpy>=1.24`，**不依赖 torch**（torch 只在引擎侧）。
- 远端 conda 环境：`/home/jeefy/miniconda3/envs/unichess/bin/python`。
- C++ PUCT 首次使用时 `g++` 编译，缓存在 `$UNICHESS_KIT_NATIVE_CACHE`
  （默认 `~/.cache/kit/native/<哈希>/`）；环境变量 `CXX` 可换编译器。
  编译失败直接抛 `NativeBuildError`，不静默回退 Python。
- GPU 租约目录：`$UNICHESS_GPU_LEASE_DIR`（默认 `~/.cache/unichess/gpu_leases`）。
- import 根：`$UNICHESS_IMPORT_ROOT`（旧名 `UNICHESS_KIT_ROOT` 仍兼容）。**扁平布局后它指的是
  `~/UniChess`，不是本仓库目录**——挂在 `sys.path` 上才能 `import Kit` / `import ResNet`。
- 引擎插件协议版本 `SPI_VERSION = 1`；引擎方在模块级声明 `KIT_SPI_VERSION`
  （不一致则拒绝加载）。现状：**ResNet 声明 2**（扁平化时提升过，见 `ResNet/kit.py`），
  Transformer / SSM 仍是 1。

## 测试

标准库 `unittest`，无 pytest。

```bash
# 全量（cwd = ~/UniChess，-t 指向 import 根；tests 是命名空间包，-t 必须给对）
cd ~/UniChess && export UNICHESS_IMPORT_ROOT=~/UniChess
python -m unittest discover -s Kit/tests -t ~/UniChess

# 单文件 / 单用例
python -m unittest Kit.tests.test_puct -v
python -m unittest Kit.tests.test_puct.test_terminal_sims_consume_budget -v
```

`test_gpu.py` 依赖 GPU，无 GPU 的机器上会跳过。269 项（2026-09-25）。

## 开发

- 本机：`C:\ProgramData\miniconda3\python.exe -m unittest discover -s tests -t .`
  （设 `PYTHONIOENCODING=utf-8`）。
- 提交信息中文；结果文件、权重、日志、`.so` 产物一律不入库（`.gitignore` 已配）。
- 训练能力在 `Kit/train/`：`Trainer` + `TrainConfig`，`steps=0` 表示由任务的
  `auto_steps(accum)` 从数据量定步数；`loss()` 可声明 `total_steps` 形参接收训练总步数
  （做退火用，见 `train/trainer.py`）。

## 引擎接入

引擎仓库提供工厂函数，返回无参 `PlayerFactory`（每局新建一个 Player）。扁平布局下工厂
地址是 `<包名>.kit:make_player_factory`：

```python
KIT_SPI_VERSION = 1

def make_player_factory(checkpoint, simulations=256, **kw):
    evaluator = ...                      # 引擎自己的 evaluator
    cfg = GumbelConfig(simulations=simulations)
    return lambda: SearchPlayer("S", Planes19Expander(evaluator), cfg=cfg, **kw)
```

现成例子：`ResNet/kit.py`、`Transformer/kit.py`、`SSM/kit.py`。
实现后用 `Kit.testing.PlayerContract` / `ExpanderContract` 跑契约测试。

批量对弈配置示例：

```bash
python -m Kit match match.json --out runs/t_vs_r/results.jsonl
```

```json
{"a": {"factory": "ResNet.kit:make_player_factory", "root": "/home/jeefy/UniChess",
       "kwargs": {"checkpoint": "ResNet/runs/...", "preset": "..."}},
 "b": {"factory": "Transformer.kit:make_player_factory", "root": "/home/jeefy/UniChess",
       "kwargs": {"checkpoint": "Transformer/runs/..."}},
 "match": {"pairs": 100, "seed": 20260925, "max_plies": 400}}
```

## C++ PUCT 接入

`make_search_player_factory(..., planes_evaluator=..., search_impl="auto")`：
`planes_evaluator` 的负载是 `(19, 8, 8) float32`；R/T 的 `kit.py` 已提供
`engine.evaluate_planes`（扁平化后由 `ResNet/kit.py`、`Transformer/kit.py` 的
`make_evaluators` 给出），默认走 C++。`search_impl="python"` 切回 Python 逐位对照，
放在 `runtime` 里不改配置哈希。

## 关键设计点

- `Think = Generator[EvalRequest, Any, R]`：所有网络前向操作都是生成器，`yield EvalRequest`，
  调度器按 `model_key` 拼批；单层同步调用 `runtime.run_sync(think)`。
- `EngineSpec(root, factory, kwargs, runtime)`：`root` 做路径隔离，同名顶层包来自别处时
  直接报错（扁平布局下 `root` 应为 `~/UniChess`）；`runtime` 只影响执行方式、不影响结果，
  不进配置哈希，与 `kwargs` 同名报错。
- 批量对弈：`python -m Kit match match.json --out runs/.../results.jsonl`；续跑同一命令
  自动接上，丢了后半行不报错。
- 后台 job：`python -m Kit.jobs <job_dir>`，进程存活期间持锁 <job_dir>/lock 文件；
  调用方只读写 `data/jobs/<id>/`，不直连进程。退出码：0 完成、1 出错、2 显存不足、
  3 被停、4 已在运行。
