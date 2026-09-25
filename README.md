`PUCTCpp`：同一算法的 C++ 实现（见下），与 `PUCT` 整树逐位一致；# UniChessKit

UniChess 各引擎（S / T / R）共享的管线库。引擎只实现少量协议，对弈、搜索、裁决、统计由本库统一提供，
避免同一类 bug 在多个仓库里反复出现。

## 包结构（依赖自上而下，`api` 不依赖 torch）

| 模块 | 内容 |
|---|---|
| `api` | 协议与数据类型：`Player`、`Expander`、`BatchEvaluator`、`EvalRequest`、`SearchBudget`、`GameStart`、`Verdict` … |
| `runtime` | `Batcher`（按 `model_key` 跨局攒批，一模型一拍一次前向）、`CoroutinePool`、`WorkerPool`（spawn；只有显式 done/error 才算结束）、`GpuLease`、`FileLock` |
| `rules` | `StandardReferee`（`claim_draw=True` 语义，超 `max_plies` 记 truncated）、`OpeningBook`、`TablebaseOracle` |
| `stats` | Elo±CI、LOS、三项/五项 GSPRT（单一实现） |
| `search` | `PUCT`：R `search/mcts.py` 的协程化移植，逐节点一致（有 parity 测试），C++ 版 `PUCTCpp` 与 Python 整树逐位一致；`Gumbel`：顺序减半搜索的**唯一实现**（S 的旧 `stateseq/gumbel.py` 已删，黄金口径在这里），与 S 逐字节一致（`test_gumbel_parity`） |
| `players` | `SearchPlayer`（残局表 → 开局库 → 搜索 → 策略）、`RandomPlayer`、`UciPlayer` |
| `pipelines.match` | 配对换色 + 开局 + SPRT 早停 + JSONL 断点续跑 + 多进程 |
| `contrib.planes19` | T/R 共用的 19 平面编码与 `Planes19Expander` |
| `serving` / `jobs` | Server 六方法适配、后台 job 进程（见下） |
| `testing` | Player / Expander 契约测试、fake 评估器 |

## 协程协议

所有需要网络前向的操作都是生成器 `Think = Generator[EvalRequest, Any, R]`：
`yield EvalRequest(evaluator, payloads)`，调度器把同一 `model_key` 的请求拼成一批前向后 `send` 回结果。
单局同步调用用 `runtime.run_sync(think)`。

## 接入引擎

引擎仓库提供一个工厂函数，返回无参的 `PlayerFactory`（每局新建一个 Player）：

```python
# Transformer/kit.py
KIT_SPI_VERSION = 1

def make_player_factory(checkpoint, simulations=800, **kw):
    evaluator = BatchFnEvaluator("T:" + checkpoint, load_engine(checkpoint).evaluate_batch)
    return lambda: SearchPlayer("T", Planes19Expander(evaluator), simulations=simulations, **kw)
```

实现后用 `testing.PlayerContract` / `ExpanderContract` 跑契约测试。

## 批量对弈

```bash
python -m Kit match match.json --out runs/t_vs_r/results.jsonl
```

```json
{
  "a": {"factory": "Transformer.kit:make_player_factory", "root": "/home/jeefy/UniChess/Transformer",
        "kwargs": {"checkpoint": "runs/transformer_20m/best_model.pt"}, "label": "T-20M"},
  "b": {"factory": "ResNet.kit:make_player_factory", "root": "/home/jeefy/UniChess/ResNet",
        "kwargs": {"checkpoint": "runs/stage1/ckpt_00187578.pt"}, "label": "R"},
  "match": {"pairs": 32, "seed": 0, "max_plies": 400, "concurrency": 8, "workers": 1,
            "openings": "bundled", "sprt": null}
}
```

- 第 2p 局 A 执白、第 2p+1 局 B 执白，同一对共用开局；结果按模型 A/B 计分，同时给出五项 Elo。
- `root` 做路径隔离：模块文件必须在该目录下，同名顶层包来自别处时直接报错。
- `runtime`（可选）：只影响运行方式、不影响结果的工厂参数（如 S 的 `pool_slots`、临时 GPU 服务目录），传给工厂但不进配置哈希；与 `kwargs` 重名报错。
- 结果 JSONL 首行是含 `config_hash` 的表头，每局一行 fsync；中断后同一命令续跑，末尾半行自动丢弃。
- 汇总写到 `<out>.summary.json`，含 `duplicate_rate`（确定性双方逐局重复时会暴露出来）。

`run_match(..., observer=f, should_stop=g)`：`observer` 逐步收到 `game_start` / `move` 事件（多进程时由
worker 转发），`should_stop` 返回真时不再开新局。

## 后台 job（Server 用）

```bash
python -m Kit.jobs <job_dir>      # job_dir/job.json 描述任务
```

`job.json`：`{"kind": "match"|"game", "a": EngineSpec, "b": EngineSpec, "names": {"A":…, "B":…},
"match": {...MatchConfig}, "game": {"max_plies", "seed", "opening"}, "gpu_mib": 预算, "lease_dir": 可选}`。
目录内容：`status.json`（starting → running → completed / error / stopped / gpu_busy，原子写）、
`live.json`（进行中各局着法与每步 source/ms/info，0.2 s 刷新）、`results.jsonl`、`job.log`、`lock`（进程存活期间持有）。
调用方用 `JobHandle.submit(...)` 启动（独立进程组），`state()` 在进程死了却没写终态时返回 `died`，
`stop()` SIGTERM 整组、超时 SIGKILL，`wait()` 等进程完全退出。退出码：0 完成、1 出错、2 显存不足、3 被停止、4 已在运行。

## GPU 租约

`runtime.GpuLease(budget_mib, name)`：在 `~/.cache/unichess/gpu_leases`（或 `UNICHESS_GPU_LEASE_DIR`）登记预算并持有文件锁。
可用量 = 总显存 − 512 MiB 余量 − 非租约进程占用 − Σ 各存活租约的 max(预算, 实际占用)；不够就抛 `GpuBusyError`
（训练进程占卡时 job 直接以 `gpu_busy` 结束，不排队）。进程死掉锁自动释放，残留的 `.lease` 文件下次申请时清理。
没有 `nvidia-smi` 的机器上总是批准。

## C++ PUCT（`search/puct_cpp.py`）

`PUCT` 的下探、棋规、19 平面编码、回传全部搬进 C++（`search/_native/puct_native.cpp`，ctypes 调用、期间释放 GIL），
Python 只剩调度：`collect`（C++ 选叶子并直接写出编码）→ `yield EvalRequest` → `apply`（C++ 算先验、展开、回传）。

- **逐位一致**：同一评估器下整棵树（moves / P / N / W / VL / sum_N / 终局值）与 `last_metrics` 都和 `PUCT` 相同
  （`tests/test_puct_cpp.py`）。为此 C++ 复刻了 python-chess 1.11 的合法着法顺序、`is_repetition` 回退、终局判定顺序，
  以及 numpy 2 的类型提升与 pairwise 求和；编译带 `-ffp-contract=off`（不合并 FMA）。
- **残局表**仍在 Python：叶子子力 <= `oracle.max_pieces` 时 C++ 暂停，把路径交回，Python 在原棋盘（含走子栈）上重放后探测、回填。
- **编译**：首次使用时 `g++` 编译，按源码 + 编译选项哈希缓存到 `~/.cache/unichess_kit/native/`（`UNICHESS_KIT_NATIVE_CACHE` 可改）；
  编译失败直接报错，不静默回退 Python（T 的 C++ MCTS 曾静默回退，见 T `5f81219`）。
- **接入**：`make_search_player_factory(..., planes_evaluator=..., search_impl="auto")`，`planes_evaluator` 的负载是
  `(19, 8, 8) float32`；T/R 的 `kit_adapter` 已提供（`engine.evaluate_planes`），默认走 C++。
  `search_impl="python"` 切回 Python 版对照——结果相同，放在 `runtime` 里不改配置哈希。
- **实测**（5070 Ti，P1 验收口径：T strat-p4 vs R stage1，各 200 模拟，32 对 64 局，concurrency 16）：
  Python 685 s → C++ 77 s（8.9×）；两边 64/64 局逐字节一致，批次计数（ticks / forwards / positions）也相同，
  与 P1 结果一致（T +11 =53 −0，Elo +60）。

## Serving（Server 六方法契约）

- `serving.make_game_engine(player_factory_loader, kit_factory=...)`：把 PlayerFactory 包成 Server 的
  `GameEngine`（setup / human_move / engine_move / state / undo / cleanup），同参数的模型全进程只加载一次，终局判定用 `classify`。
- `serving.game_engine_player_factory(engine_path, engine_kwargs)`：反过来把现有 `GameEngine` 包成 kit `Player`，
  供没有声明 `KIT_FACTORY` 的引擎（T 的 C++ MCTS、M6）进入 job 管线。

## 测试

标准库 unittest（远端无 pytest）：

```bash
python -m unittest discover -s tests -t .
```

与 R/T 的 parity 测试会自动在同级目录 `../ResNet`、`../Transformer` 查找（或设
`UNICHESS_R_ROOT` / `UNICHESS_T_ROOT`），找不到时跳过。
