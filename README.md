# UniChessKit

UniChess 各引擎（S / T / R）共享的管线库。引擎只实现少量协议，对弈、搜索、裁决、统计由本库统一提供，
避免同一类 bug 在多个仓库里反复出现。

## 包结构（依赖自上而下，`api` 不依赖 torch）

| 模块 | 内容 |
|---|---|
| `api` | 协议与数据类型：`Player`、`Expander`、`BatchEvaluator`、`EvalRequest`、`SearchBudget`、`GameStart`、`Verdict` … |
| `runtime` | `Batcher`（按 `model_key` 跨局攒批，一模型一拍一次前向）、`CoroutinePool`、`WorkerPool`（spawn；只有显式 done/error 才算结束） |
| `rules` | `StandardReferee`（`claim_draw=True` 语义，超 `max_plies` 记 truncated）、`OpeningBook`、`TablebaseOracle` |
| `stats` | Elo±CI、LOS、三项/五项 GSPRT（单一实现） |
| `search` | `PUCT`：R `search/mcts.py` 的协程化移植，逐节点一致（有 parity 测试） |
| `players` | `SearchPlayer`（残局表 → 开局库 → 搜索 → 策略）、`RandomPlayer`、`UciPlayer` |
| `pipelines.match` | 配对换色 + 开局 + SPRT 早停 + JSONL 断点续跑 + 多进程 |
| `contrib.planes19` | T/R 共用的 19 平面编码与 `Planes19Expander` |
| `testing` | Player / Expander 契约测试、fake 评估器 |

## 协程协议

所有需要网络前向的操作都是生成器 `Think = Generator[EvalRequest, Any, R]`：
`yield EvalRequest(evaluator, payloads)`，调度器把同一 `model_key` 的请求拼成一批前向后 `send` 回结果。
单局同步调用用 `runtime.run_sync(think)`。

## 接入引擎

引擎仓库提供一个工厂函数，返回无参的 `PlayerFactory`（每局新建一个 Player）：

```python
# unichess_t/kit_adapter.py
KIT_SPI_VERSION = 1

def make_player_factory(checkpoint, simulations=800, **kw):
    evaluator = BatchFnEvaluator("T:" + checkpoint, load_engine(checkpoint).evaluate_batch)
    return lambda: SearchPlayer("T", Planes19Expander(evaluator), simulations=simulations, **kw)
```

实现后用 `testing.PlayerContract` / `ExpanderContract` 跑契约测试。

## 批量对弈

```bash
python -m unichess_kit.match match.json --out runs/t_vs_r/results.jsonl
```

```json
{
  "a": {"factory": "unichess_t.kit_adapter:make_player_factory", "root": "/home/jeefy/UniChess/Transformer",
        "kwargs": {"checkpoint": "runs/transformer_20m/best_model.pt"}, "label": "T-20M"},
  "b": {"factory": "unichess_r.kit_adapter:make_player_factory", "root": "/home/jeefy/UniChess/ResNet",
        "kwargs": {"checkpoint": "runs/stage1/ckpt_00187578.pt"}, "label": "R"},
  "match": {"pairs": 32, "seed": 0, "max_plies": 400, "concurrency": 8, "workers": 1,
            "openings": "bundled", "sprt": null}
}
```

- 第 2p 局 A 执白、第 2p+1 局 B 执白，同一对共用开局；结果按模型 A/B 计分，同时给出五项 Elo。
- `root` 做路径隔离：模块文件必须在该目录下，同名顶层包来自别处时直接报错。
- 结果 JSONL 首行是含 `config_hash` 的表头，每局一行 fsync；中断后同一命令续跑，末尾半行自动丢弃。
- 汇总写到 `<out>.summary.json`，含 `duplicate_rate`（确定性双方逐局重复时会暴露出来）。

## 测试

标准库 unittest（远端无 pytest）：

```bash
python -m unittest discover -s tests -t .
```

与 R/T 的 parity 测试会自动在同级目录 `../ResNet`、`../Transformer` 查找（或设
`UNICHESS_R_ROOT` / `UNICHESS_T_ROOT`），找不到时跳过。
