"""Kit：S / T / R 共用的对局、搜索、评测与训练框架。

仓库根目录即包 ``Kit``，import 根是它的父目录（远端 ``~/UniChess``）：
所有入口在 import 根下运行 ``python -m Kit ...``，多进程 spawn 的子进程也按包名导入。

分层（依赖只能自上而下；``api`` / ``rules`` / ``stats`` / ``runtime`` / ``search`` 不依赖 torch）::

    api        协议与数据类型
    rules      裁判、开局库、残局表
    stats      Elo / LOS / SPRT
    runtime    协程攒批驱动、多进程 worker 池、GPU 租约
    search     搜索算法（PUCT / PUCTCpp / Gumbel）
    players    Player 实现（搜索、随机、UCI）
    pipelines  对局编排（match / selfplay / loop）
    planes19   T/R 共用：19 平面编码、96 字节记录、数据集、损失、数据构建
    train      统一训练器（TrainTask 协议）
    testing    契约测试与假引擎
"""
from pathlib import Path

__version__ = "0.2.0"

# 引擎插件协议（SPI）版本。协议有不兼容改动时加一，
# registry 加载引擎时比对引擎声明的 KIT_SPI_VERSION，防止新旧版本混用。
SPI_VERSION = 2

# import 根：Kit 目录的父目录。jobs 子进程经 PYTHONPATH 继承它。
IMPORT_ROOT = Path(__file__).resolve().parent.parent
