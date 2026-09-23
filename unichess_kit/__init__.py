"""UniChessKit：S / T / R 共用的对局、搜索、评测管线。

分层（依赖只能自上而下，``api`` 不依赖 torch）::

    api        协议与数据类型
    rules      裁判、开局库、残局表
    stats      Elo / LOS / SPRT
    runtime    协程攒批驱动、多进程 worker 池
    search     搜索算法（PUCT）
    players    Player 实现（搜索、随机、UCI）
    pipelines  对局编排（match）
    contrib    可选组件（T/R 共用的 19 平面编码）
    testing    契约测试与假引擎
"""

__version__ = "0.1.0"

# 引擎插件协议（SPI）版本。协议有不兼容改动时加一，
# registry 加载引擎时比对引擎声明的 KIT_SPI_VERSION，防止新旧版本混用。
SPI_VERSION = 1
