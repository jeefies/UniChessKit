"""Maia Chess 的 PyTorch 移植：lc0 ``net.proto`` 权重格式的人类 Elo 档位引擎。

定位
----
给 SSM/Transformer/ResNet 的 arena 提供**人类 Elo 锚点**。Stockfish 的 ``UCI_Elo``
限力档实测被压缩 2.3~3.9 倍、名义值不可当 Elo 读（见 ``SSM/docs/representation-and-strength-audit.md``
§5.4）；Maia 的 9 个档位是按 Lichess 真实分段（1100~1900，间隔 100）分别训练的，
档位标签本身就是人类 Elo，而且它下人类开局、是神经网络，能同时解决"标尺未校准"和
"开局库偏差"两个问题。

权重与格式
----------
- 来源：``CSSLab/maia-chess`` v1.0 release 的 ``maia-<elo>.pb.gz``（9 档，各约 1.3MB）
- 格式：lc0 ``Net`` protobuf，``weights`` 字段是 ``pblczero.Weights``；
  每层 ``LINEAR16`` 量化（``min_val``/``max_val`` + uint16 载荷）
- 结构（从 ``Format.network_format`` 读出的实测值）::

      input               = INPUT_CLASSICAL_112_PLANE
      output              = OUTPUT_WDL
      network             = NETWORK_SE_WITH_HEADFORMAT
      policy              = POLICY_CONVOLUTION
      value               = VALUE_WDL
      moves_left          = 缺失（无该头）
      default_activation  = 缺失 => DEFAULT_ACTIVATION_RELU

- 实测维度（maia-1500）：input 卷积 112→64，**6 个残差块**（64→64×2），
  SE 隐藏 8；policy1 64→64，policy 64→**80**（只用前 73 个平面）；
  value 1×1 卷积 64→32，FC 2048→128→3（WDL）

对拍依据（lc0 v0.29.0 源码，逐行照抄而非猜测）
--------------------------------------------
- 前向与 BN 折叠：``src/neural/blas/network_blas.cc``、``src/neural/network_legacy.cc``
- SE 单元：``src/neural/blas/se_unit.cc``（注意 lc0 的 ``beta`` 里
  ``gamma * conv2_bias`` 会出现两次，必须照抄）
- LINEAR16 解码：``src/utils/weights_adapter.cc``（``u16 / 65535 * (max-min) + min``）
- 112 输入平面：``src/neural/encoder.cc``（``kMoveHistory=8``、``kPlanesPerBoard=13``）
- 1858 policy 索引表：``src/chess/bitboard.cc`` 的 ``kIdxToMove``（纯数据：UCI 着法串
  按索引顺序排列）。**马升变不在表内**（lc0 的已知缺口，66 个升变全是 q/r/b）
- 卷积 policy 平面映射：由着法表算法化推导（8 方向×7 距离 + 8 马步 + 9 升变平面），
  已与 lc0 ``src/neural/shared/policy_map.h`` 的 ``kConvPolicyMap`` 逐项对拍一致

关键约定：**棋盘始终按行棋方视角**（"我方"在底部）。lc0 的 ``ChessBoard`` 对黑方行棋
的局面做了纵向翻转，所以黑方行棋时输入平面要 rank 镜像，着法索引也要先翻到行棋方
坐标系再查表。
"""
from __future__ import annotations

import gzip
import struct
from typing import Optional

import numpy as np

__all__ = [
    'MOVES', 'MOVE_TO_IDX', 'NUM_POLICY', 'NUM_PLANES',
    'MaiaWeights', 'MaiaNet', 'load_maia', 'encode_input',
    'move_to_index', 'index_to_move', 'MaiaError',
]

NUM_POLICY = 1858
NUM_PLANES = 112
MOVE_HISTORY = 8
PLANES_PER_BOARD = 13
AUX_PLANE_BASE = PLANES_PER_BOARD * MOVE_HISTORY      # 104
CHANNELS = 64
RESIDUAL_BLOCKS = 6
SE_HIDDEN = 8
POLICY_PLANES = 73           # 卷积 policy 头实际使用的平面数（权重里有 80，多的是 padding）
POLICY_PADDED_PLANES = 80
VALUE_PLANES = 32
VALUE_HIDDEN = 128

# lc0 的 8 个滑动方向，顺序即 plane 组序（每组 7 个距离）
_SLIDE_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
# lc0 的 8 个马步，顺序即 plane 56..63
_KNIGHT_DIRS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
# 升变平面 = 64 + (df+1)*3 + piece_idx；piece 顺序是 r, b, q（不是 q, r, b）
_PROMO_PIECE = {'r': 0, 'b': 1, 'q': 2}

# 我方/对方的棋子平面顺序（P,N,B,R,Q,K）
_PIECE_ORDER = (1, 2, 3, 4, 5, 6)


class MaiaError(RuntimeError):
    """Maia 权重或编码不支持的显式报错（不静默降级）。"""


# --------------------------------------------------------------------------
# policy 索引表
# --------------------------------------------------------------------------
_MOVES_RAW = """
    a1b1 a1c1 a1d1 a1e1 a1f1 a1g1 a1h1 a1a2 a1b2 a1c2 a1a3 a1b3
    a1c3 a1a4 a1d4 a1a5 a1e5 a1a6 a1f6 a1a7 a1g7 a1a8 a1h8 b1a1
    b1c1 b1d1 b1e1 b1f1 b1g1 b1h1 b1a2 b1b2 b1c2 b1d2 b1a3 b1b3
    b1c3 b1d3 b1b4 b1e4 b1b5 b1f5 b1b6 b1g6 b1b7 b1h7 b1b8 c1a1
    c1b1 c1d1 c1e1 c1f1 c1g1 c1h1 c1a2 c1b2 c1c2 c1d2 c1e2 c1a3
    c1b3 c1c3 c1d3 c1e3 c1c4 c1f4 c1c5 c1g5 c1c6 c1h6 c1c7 c1c8
    d1a1 d1b1 d1c1 d1e1 d1f1 d1g1 d1h1 d1b2 d1c2 d1d2 d1e2 d1f2
    d1b3 d1c3 d1d3 d1e3 d1f3 d1a4 d1d4 d1g4 d1d5 d1h5 d1d6 d1d7
    d1d8 e1a1 e1b1 e1c1 e1d1 e1f1 e1g1 e1h1 e1c2 e1d2 e1e2 e1f2
    e1g2 e1c3 e1d3 e1e3 e1f3 e1g3 e1b4 e1e4 e1h4 e1a5 e1e5 e1e6
    e1e7 e1e8 f1a1 f1b1 f1c1 f1d1 f1e1 f1g1 f1h1 f1d2 f1e2 f1f2
    f1g2 f1h2 f1d3 f1e3 f1f3 f1g3 f1h3 f1c4 f1f4 f1b5 f1f5 f1a6
    f1f6 f1f7 f1f8 g1a1 g1b1 g1c1 g1d1 g1e1 g1f1 g1h1 g1e2 g1f2
    g1g2 g1h2 g1e3 g1f3 g1g3 g1h3 g1d4 g1g4 g1c5 g1g5 g1b6 g1g6
    g1a7 g1g7 g1g8 h1a1 h1b1 h1c1 h1d1 h1e1 h1f1 h1g1 h1f2 h1g2
    h1h2 h1f3 h1g3 h1h3 h1e4 h1h4 h1d5 h1h5 h1c6 h1h6 h1b7 h1h7
    h1a8 h1h8 a2a1 a2b1 a2c1 a2b2 a2c2 a2d2 a2e2 a2f2 a2g2 a2h2
    a2a3 a2b3 a2c3 a2a4 a2b4 a2c4 a2a5 a2d5 a2a6 a2e6 a2a7 a2f7
    a2a8 a2g8 b2a1 b2b1 b2c1 b2d1 b2a2 b2c2 b2d2 b2e2 b2f2 b2g2
    b2h2 b2a3 b2b3 b2c3 b2d3 b2a4 b2b4 b2c4 b2d4 b2b5 b2e5 b2b6
    b2f6 b2b7 b2g7 b2b8 b2h8 c2a1 c2b1 c2c1 c2d1 c2e1 c2a2 c2b2
    c2d2 c2e2 c2f2 c2g2 c2h2 c2a3 c2b3 c2c3 c2d3 c2e3 c2a4 c2b4
    c2c4 c2d4 c2e4 c2c5 c2f5 c2c6 c2g6 c2c7 c2h7 c2c8 d2b1 d2c1
    d2d1 d2e1 d2f1 d2a2 d2b2 d2c2 d2e2 d2f2 d2g2 d2h2 d2b3 d2c3
    d2d3 d2e3 d2f3 d2b4 d2c4 d2d4 d2e4 d2f4 d2a5 d2d5 d2g5 d2d6
    d2h6 d2d7 d2d8 e2c1 e2d1 e2e1 e2f1 e2g1 e2a2 e2b2 e2c2 e2d2
    e2f2 e2g2 e2h2 e2c3 e2d3 e2e3 e2f3 e2g3 e2c4 e2d4 e2e4 e2f4
    e2g4 e2b5 e2e5 e2h5 e2a6 e2e6 e2e7 e2e8 f2d1 f2e1 f2f1 f2g1
    f2h1 f2a2 f2b2 f2c2 f2d2 f2e2 f2g2 f2h2 f2d3 f2e3 f2f3 f2g3
    f2h3 f2d4 f2e4 f2f4 f2g4 f2h4 f2c5 f2f5 f2b6 f2f6 f2a7 f2f7
    f2f8 g2e1 g2f1 g2g1 g2h1 g2a2 g2b2 g2c2 g2d2 g2e2 g2f2 g2h2
    g2e3 g2f3 g2g3 g2h3 g2e4 g2f4 g2g4 g2h4 g2d5 g2g5 g2c6 g2g6
    g2b7 g2g7 g2a8 g2g8 h2f1 h2g1 h2h1 h2a2 h2b2 h2c2 h2d2 h2e2
    h2f2 h2g2 h2f3 h2g3 h2h3 h2f4 h2g4 h2h4 h2e5 h2h5 h2d6 h2h6
    h2c7 h2h7 h2b8 h2h8 a3a1 a3b1 a3c1 a3a2 a3b2 a3c2 a3b3 a3c3
    a3d3 a3e3 a3f3 a3g3 a3h3 a3a4 a3b4 a3c4 a3a5 a3b5 a3c5 a3a6
    a3d6 a3a7 a3e7 a3a8 a3f8 b3a1 b3b1 b3c1 b3d1 b3a2 b3b2 b3c2
    b3d2 b3a3 b3c3 b3d3 b3e3 b3f3 b3g3 b3h3 b3a4 b3b4 b3c4 b3d4
    b3a5 b3b5 b3c5 b3d5 b3b6 b3e6 b3b7 b3f7 b3b8 b3g8 c3a1 c3b1
    c3c1 c3d1 c3e1 c3a2 c3b2 c3c2 c3d2 c3e2 c3a3 c3b3 c3d3 c3e3
    c3f3 c3g3 c3h3 c3a4 c3b4 c3c4 c3d4 c3e4 c3a5 c3b5 c3c5 c3d5
    c3e5 c3c6 c3f6 c3c7 c3g7 c3c8 c3h8 d3b1 d3c1 d3d1 d3e1 d3f1
    d3b2 d3c2 d3d2 d3e2 d3f2 d3a3 d3b3 d3c3 d3e3 d3f3 d3g3 d3h3
    d3b4 d3c4 d3d4 d3e4 d3f4 d3b5 d3c5 d3d5 d3e5 d3f5 d3a6 d3d6
    d3g6 d3d7 d3h7 d3d8 e3c1 e3d1 e3e1 e3f1 e3g1 e3c2 e3d2 e3e2
    e3f2 e3g2 e3a3 e3b3 e3c3 e3d3 e3f3 e3g3 e3h3 e3c4 e3d4 e3e4
    e3f4 e3g4 e3c5 e3d5 e3e5 e3f5 e3g5 e3b6 e3e6 e3h6 e3a7 e3e7
    e3e8 f3d1 f3e1 f3f1 f3g1 f3h1 f3d2 f3e2 f3f2 f3g2 f3h2 f3a3
    f3b3 f3c3 f3d3 f3e3 f3g3 f3h3 f3d4 f3e4 f3f4 f3g4 f3h4 f3d5
    f3e5 f3f5 f3g5 f3h5 f3c6 f3f6 f3b7 f3f7 f3a8 f3f8 g3e1 g3f1
    g3g1 g3h1 g3e2 g3f2 g3g2 g3h2 g3a3 g3b3 g3c3 g3d3 g3e3 g3f3
    g3h3 g3e4 g3f4 g3g4 g3h4 g3e5 g3f5 g3g5 g3h5 g3d6 g3g6 g3c7
    g3g7 g3b8 g3g8 h3f1 h3g1 h3h1 h3f2 h3g2 h3h2 h3a3 h3b3 h3c3
    h3d3 h3e3 h3f3 h3g3 h3f4 h3g4 h3h4 h3f5 h3g5 h3h5 h3e6 h3h6
    h3d7 h3h7 h3c8 h3h8 a4a1 a4d1 a4a2 a4b2 a4c2 a4a3 a4b3 a4c3
    a4b4 a4c4 a4d4 a4e4 a4f4 a4g4 a4h4 a4a5 a4b5 a4c5 a4a6 a4b6
    a4c6 a4a7 a4d7 a4a8 a4e8 b4b1 b4e1 b4a2 b4b2 b4c2 b4d2 b4a3
    b4b3 b4c3 b4d3 b4a4 b4c4 b4d4 b4e4 b4f4 b4g4 b4h4 b4a5 b4b5
    b4c5 b4d5 b4a6 b4b6 b4c6 b4d6 b4b7 b4e7 b4b8 b4f8 c4c1 c4f1
    c4a2 c4b2 c4c2 c4d2 c4e2 c4a3 c4b3 c4c3 c4d3 c4e3 c4a4 c4b4
    c4d4 c4e4 c4f4 c4g4 c4h4 c4a5 c4b5 c4c5 c4d5 c4e5 c4a6 c4b6
    c4c6 c4d6 c4e6 c4c7 c4f7 c4c8 c4g8 d4a1 d4d1 d4g1 d4b2 d4c2
    d4d2 d4e2 d4f2 d4b3 d4c3 d4d3 d4e3 d4f3 d4a4 d4b4 d4c4 d4e4
    d4f4 d4g4 d4h4 d4b5 d4c5 d4d5 d4e5 d4f5 d4b6 d4c6 d4d6 d4e6
    d4f6 d4a7 d4d7 d4g7 d4d8 d4h8 e4b1 e4e1 e4h1 e4c2 e4d2 e4e2
    e4f2 e4g2 e4c3 e4d3 e4e3 e4f3 e4g3 e4a4 e4b4 e4c4 e4d4 e4f4
    e4g4 e4h4 e4c5 e4d5 e4e5 e4f5 e4g5 e4c6 e4d6 e4e6 e4f6 e4g6
    e4b7 e4e7 e4h7 e4a8 e4e8 f4c1 f4f1 f4d2 f4e2 f4f2 f4g2 f4h2
    f4d3 f4e3 f4f3 f4g3 f4h3 f4a4 f4b4 f4c4 f4d4 f4e4 f4g4 f4h4
    f4d5 f4e5 f4f5 f4g5 f4h5 f4d6 f4e6 f4f6 f4g6 f4h6 f4c7 f4f7
    f4b8 f4f8 g4d1 g4g1 g4e2 g4f2 g4g2 g4h2 g4e3 g4f3 g4g3 g4h3
    g4a4 g4b4 g4c4 g4d4 g4e4 g4f4 g4h4 g4e5 g4f5 g4g5 g4h5 g4e6
    g4f6 g4g6 g4h6 g4d7 g4g7 g4c8 g4g8 h4e1 h4h1 h4f2 h4g2 h4h2
    h4f3 h4g3 h4h3 h4a4 h4b4 h4c4 h4d4 h4e4 h4f4 h4g4 h4f5 h4g5
    h4h5 h4f6 h4g6 h4h6 h4e7 h4h7 h4d8 h4h8 a5a1 a5e1 a5a2 a5d2
    a5a3 a5b3 a5c3 a5a4 a5b4 a5c4 a5b5 a5c5 a5d5 a5e5 a5f5 a5g5
    a5h5 a5a6 a5b6 a5c6 a5a7 a5b7 a5c7 a5a8 a5d8 b5b1 b5f1 b5b2
    b5e2 b5a3 b5b3 b5c3 b5d3 b5a4 b5b4 b5c4 b5d4 b5a5 b5c5 b5d5
    b5e5 b5f5 b5g5 b5h5 b5a6 b5b6 b5c6 b5d6 b5a7 b5b7 b5c7 b5d7
    b5b8 b5e8 c5c1 c5g1 c5c2 c5f2 c5a3 c5b3 c5c3 c5d3 c5e3 c5a4
    c5b4 c5c4 c5d4 c5e4 c5a5 c5b5 c5d5 c5e5 c5f5 c5g5 c5h5 c5a6
    c5b6 c5c6 c5d6 c5e6 c5a7 c5b7 c5c7 c5d7 c5e7 c5c8 c5f8 d5d1
    d5h1 d5a2 d5d2 d5g2 d5b3 d5c3 d5d3 d5e3 d5f3 d5b4 d5c4 d5d4
    d5e4 d5f4 d5a5 d5b5 d5c5 d5e5 d5f5 d5g5 d5h5 d5b6 d5c6 d5d6
    d5e6 d5f6 d5b7 d5c7 d5d7 d5e7 d5f7 d5a8 d5d8 d5g8 e5a1 e5e1
    e5b2 e5e2 e5h2 e5c3 e5d3 e5e3 e5f3 e5g3 e5c4 e5d4 e5e4 e5f4
    e5g4 e5a5 e5b5 e5c5 e5d5 e5f5 e5g5 e5h5 e5c6 e5d6 e5e6 e5f6
    e5g6 e5c7 e5d7 e5e7 e5f7 e5g7 e5b8 e5e8 e5h8 f5b1 f5f1 f5c2
    f5f2 f5d3 f5e3 f5f3 f5g3 f5h3 f5d4 f5e4 f5f4 f5g4 f5h4 f5a5
    f5b5 f5c5 f5d5 f5e5 f5g5 f5h5 f5d6 f5e6 f5f6 f5g6 f5h6 f5d7
    f5e7 f5f7 f5g7 f5h7 f5c8 f5f8 g5c1 g5g1 g5d2 g5g2 g5e3 g5f3
    g5g3 g5h3 g5e4 g5f4 g5g4 g5h4 g5a5 g5b5 g5c5 g5d5 g5e5 g5f5
    g5h5 g5e6 g5f6 g5g6 g5h6 g5e7 g5f7 g5g7 g5h7 g5d8 g5g8 h5d1
    h5h1 h5e2 h5h2 h5f3 h5g3 h5h3 h5f4 h5g4 h5h4 h5a5 h5b5 h5c5
    h5d5 h5e5 h5f5 h5g5 h5f6 h5g6 h5h6 h5f7 h5g7 h5h7 h5e8 h5h8
    a6a1 a6f1 a6a2 a6e2 a6a3 a6d3 a6a4 a6b4 a6c4 a6a5 a6b5 a6c5
    a6b6 a6c6 a6d6 a6e6 a6f6 a6g6 a6h6 a6a7 a6b7 a6c7 a6a8 a6b8
    a6c8 b6b1 b6g1 b6b2 b6f2 b6b3 b6e3 b6a4 b6b4 b6c4 b6d4 b6a5
    b6b5 b6c5 b6d5 b6a6 b6c6 b6d6 b6e6 b6f6 b6g6 b6h6 b6a7 b6b7
    b6c7 b6d7 b6a8 b6b8 b6c8 b6d8 c6c1 c6h1 c6c2 c6g2 c6c3 c6f3
    c6a4 c6b4 c6c4 c6d4 c6e4 c6a5 c6b5 c6c5 c6d5 c6e5 c6a6 c6b6
    c6d6 c6e6 c6f6 c6g6 c6h6 c6a7 c6b7 c6c7 c6d7 c6e7 c6a8 c6b8
    c6c8 c6d8 c6e8 d6d1 d6d2 d6h2 d6a3 d6d3 d6g3 d6b4 d6c4 d6d4
    d6e4 d6f4 d6b5 d6c5 d6d5 d6e5 d6f5 d6a6 d6b6 d6c6 d6e6 d6f6
    d6g6 d6h6 d6b7 d6c7 d6d7 d6e7 d6f7 d6b8 d6c8 d6d8 d6e8 d6f8
    e6e1 e6a2 e6e2 e6b3 e6e3 e6h3 e6c4 e6d4 e6e4 e6f4 e6g4 e6c5
    e6d5 e6e5 e6f5 e6g5 e6a6 e6b6 e6c6 e6d6 e6f6 e6g6 e6h6 e6c7
    e6d7 e6e7 e6f7 e6g7 e6c8 e6d8 e6e8 e6f8 e6g8 f6a1 f6f1 f6b2
    f6f2 f6c3 f6f3 f6d4 f6e4 f6f4 f6g4 f6h4 f6d5 f6e5 f6f5 f6g5
    f6h5 f6a6 f6b6 f6c6 f6d6 f6e6 f6g6 f6h6 f6d7 f6e7 f6f7 f6g7
    f6h7 f6d8 f6e8 f6f8 f6g8 f6h8 g6b1 g6g1 g6c2 g6g2 g6d3 g6g3
    g6e4 g6f4 g6g4 g6h4 g6e5 g6f5 g6g5 g6h5 g6a6 g6b6 g6c6 g6d6
    g6e6 g6f6 g6h6 g6e7 g6f7 g6g7 g6h7 g6e8 g6f8 g6g8 g6h8 h6c1
    h6h1 h6d2 h6h2 h6e3 h6h3 h6f4 h6g4 h6h4 h6f5 h6g5 h6h5 h6a6
    h6b6 h6c6 h6d6 h6e6 h6f6 h6g6 h6f7 h6g7 h6h7 h6f8 h6g8 h6h8
    a7a1 a7g1 a7a2 a7f2 a7a3 a7e3 a7a4 a7d4 a7a5 a7b5 a7c5 a7a6
    a7b6 a7c6 a7b7 a7c7 a7d7 a7e7 a7f7 a7g7 a7h7 a7a8 a7b8 a7c8
    b7b1 b7h1 b7b2 b7g2 b7b3 b7f3 b7b4 b7e4 b7a5 b7b5 b7c5 b7d5
    b7a6 b7b6 b7c6 b7d6 b7a7 b7c7 b7d7 b7e7 b7f7 b7g7 b7h7 b7a8
    b7b8 b7c8 b7d8 c7c1 c7c2 c7h2 c7c3 c7g3 c7c4 c7f4 c7a5 c7b5
    c7c5 c7d5 c7e5 c7a6 c7b6 c7c6 c7d6 c7e6 c7a7 c7b7 c7d7 c7e7
    c7f7 c7g7 c7h7 c7a8 c7b8 c7c8 c7d8 c7e8 d7d1 d7d2 d7d3 d7h3
    d7a4 d7d4 d7g4 d7b5 d7c5 d7d5 d7e5 d7f5 d7b6 d7c6 d7d6 d7e6
    d7f6 d7a7 d7b7 d7c7 d7e7 d7f7 d7g7 d7h7 d7b8 d7c8 d7d8 d7e8
    d7f8 e7e1 e7e2 e7a3 e7e3 e7b4 e7e4 e7h4 e7c5 e7d5 e7e5 e7f5
    e7g5 e7c6 e7d6 e7e6 e7f6 e7g6 e7a7 e7b7 e7c7 e7d7 e7f7 e7g7
    e7h7 e7c8 e7d8 e7e8 e7f8 e7g8 f7f1 f7a2 f7f2 f7b3 f7f3 f7c4
    f7f4 f7d5 f7e5 f7f5 f7g5 f7h5 f7d6 f7e6 f7f6 f7g6 f7h6 f7a7
    f7b7 f7c7 f7d7 f7e7 f7g7 f7h7 f7d8 f7e8 f7f8 f7g8 f7h8 g7a1
    g7g1 g7b2 g7g2 g7c3 g7g3 g7d4 g7g4 g7e5 g7f5 g7g5 g7h5 g7e6
    g7f6 g7g6 g7h6 g7a7 g7b7 g7c7 g7d7 g7e7 g7f7 g7h7 g7e8 g7f8
    g7g8 g7h8 h7b1 h7h1 h7c2 h7h2 h7d3 h7h3 h7e4 h7h4 h7f5 h7g5
    h7h5 h7f6 h7g6 h7h6 h7a7 h7b7 h7c7 h7d7 h7e7 h7f7 h7g7 h7f8
    h7g8 h7h8 a8a1 a8h1 a8a2 a8g2 a8a3 a8f3 a8a4 a8e4 a8a5 a8d5
    a8a6 a8b6 a8c6 a8a7 a8b7 a8c7 a8b8 a8c8 a8d8 a8e8 a8f8 a8g8
    a8h8 b8b1 b8b2 b8h2 b8b3 b8g3 b8b4 b8f4 b8b5 b8e5 b8a6 b8b6
    b8c6 b8d6 b8a7 b8b7 b8c7 b8d7 b8a8 b8c8 b8d8 b8e8 b8f8 b8g8
    b8h8 c8c1 c8c2 c8c3 c8h3 c8c4 c8g4 c8c5 c8f5 c8a6 c8b6 c8c6
    c8d6 c8e6 c8a7 c8b7 c8c7 c8d7 c8e7 c8a8 c8b8 c8d8 c8e8 c8f8
    c8g8 c8h8 d8d1 d8d2 d8d3 d8d4 d8h4 d8a5 d8d5 d8g5 d8b6 d8c6
    d8d6 d8e6 d8f6 d8b7 d8c7 d8d7 d8e7 d8f7 d8a8 d8b8 d8c8 d8e8
    d8f8 d8g8 d8h8 e8e1 e8e2 e8e3 e8a4 e8e4 e8b5 e8e5 e8h5 e8c6
    e8d6 e8e6 e8f6 e8g6 e8c7 e8d7 e8e7 e8f7 e8g7 e8a8 e8b8 e8c8
    e8d8 e8f8 e8g8 e8h8 f8f1 f8f2 f8a3 f8f3 f8b4 f8f4 f8c5 f8f5
    f8d6 f8e6 f8f6 f8g6 f8h6 f8d7 f8e7 f8f7 f8g7 f8h7 f8a8 f8b8
    f8c8 f8d8 f8e8 f8g8 f8h8 g8g1 g8a2 g8g2 g8b3 g8g3 g8c4 g8g4
    g8d5 g8g5 g8e6 g8f6 g8g6 g8h6 g8e7 g8f7 g8g7 g8h7 g8a8 g8b8
    g8c8 g8d8 g8e8 g8f8 g8h8 h8a1 h8h1 h8b2 h8h2 h8c3 h8h3 h8d4
    h8h4 h8e5 h8h5 h8f6 h8g6 h8h6 h8f7 h8g7 h8h7 h8a8 h8b8 h8c8
    h8d8 h8e8 h8f8 h8g8 a7a8q a7a8r a7a8b a7b8q a7b8r a7b8b b7a8q b7a8r
    b7a8b b7b8q b7b8r b7b8b b7c8q b7c8r b7c8b c7b8q c7b8r c7b8b c7c8q c7c8r
    c7c8b c7d8q c7d8r c7d8b d7c8q d7c8r d7c8b d7d8q d7d8r d7d8b d7e8q d7e8r
    d7e8b e7d8q e7d8r e7d8b e7e8q e7e8r e7e8b e7f8q e7f8r e7f8b f7e8q f7e8r
    f7e8b f7f8q f7f8r f7f8b f7g8q f7g8r f7g8b g7f8q g7f8r g7f8b g7g8q g7g8r
    g7g8b g7h8q g7h8r g7h8b h7g8q h7g8r h7g8b h7h8q h7h8r h7h8b
    """

MOVES = _MOVES_RAW.split()
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES)}

assert len(MOVES) == NUM_POLICY, f'policy 表应有 {NUM_POLICY} 项，实得 {len(MOVES)}'
assert len(set(MOVES)) == NUM_POLICY, 'policy 表有重复项'

# 推导 (plane, from_square) -> policy index，即 lc0 的 kConvPolicyMap。
_CONV_MAP = [-1] * (POLICY_PLANES * 64)


def _sq(name: str) -> int:
    return (ord(name[0]) - 97) + 8 * (ord(name[1]) - 49)


def _build_conv_map() -> None:
    for idx, uci in enumerate(MOVES):
        f, t = _sq(uci[:2]), _sq(uci[2:4])
        df, dr = (t % 8) - (f % 8), (t // 8) - (f // 8)
        if len(uci) == 5:
            plane = 64 + (df + 1) * 3 + _PROMO_PIECE[uci[4]]
        elif (abs(df), abs(dr)) in ((1, 2), (2, 1)):
            plane = 56 + _KNIGHT_DIRS.index((df, dr))
        else:
            unit = (df // abs(df) if df else 0, dr // abs(dr) if dr else 0)
            plane = _SLIDE_DIRS.index(unit) * 7 + max(abs(df), abs(dr)) - 1
        pos = plane * 64 + f
        if _CONV_MAP[pos] != -1:
            raise MaiaError(f'conv policy 平面 {plane} 的起始格 {f} 重复赋值')
        _CONV_MAP[pos] = idx


_build_conv_map()
assert sum(1 for x in _CONV_MAP if x >= 0) == NUM_POLICY


def index_to_uci(idx: int) -> str:
    return MOVES[idx]


def _flip_square(s: int) -> int:
    """行棋方坐标系的 rank 镜像：黑方行棋时真实格 -> lc0 行棋方格。"""
    return (7 - s // 8) * 8 + s % 8


def move_to_index(move, board) -> Optional[int]:
    """python-chess 着法 -> lc0 policy 索引；表外着法（马升变）返回 None。"""
    import chess
    if board.is_castling(move):
        # lc0 用"王吃自家车"表示易位；行棋方坐标系里王恒在 e1、车在 h1/a1
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        f_m, t_m = 4, (7 if kingside else 0)
    else:
        f_m, t_m = move.from_square, move.to_square
        if board.turn == chess.BLACK:
            f_m, t_m = _flip_square(f_m), _flip_square(t_m)
    promo = ''
    if move.promotion:
        promo = {chess.QUEEN: 'q', chess.ROOK: 'r',
                 chess.BISHOP: 'b', chess.KNIGHT: 'n'}[move.promotion]
    uci = chess.square_name(f_m) + chess.square_name(t_m) + promo
    return MOVE_TO_IDX.get(uci)


def index_to_move(idx: int, board):
    """lc0 policy 索引 -> python-chess 着法；越界/不存在返回 None。"""
    import chess
    uci = MOVES[idx]
    f_m, t_m = _sq(uci[:2]), _sq(uci[2:4])
    if board.turn == chess.BLACK:
        f_m, t_m = _flip_square(f_m), _flip_square(t_m)
    if len(uci) == 5:
        move = chess.Move(f_m, t_m, promotion={'q': chess.QUEEN, 'r': chess.ROOK,
                                               'b': chess.BISHOP}[uci[4]])
    elif uci in ('e1h1', 'e1a1'):
        # lc0 用"王吃自家车"表示易位，行棋方坐标系里恒为 e1h1/e1a1；
        # 翻回真实坐标后王的目标格是同一横线的 g/c  file。
        # 王一步走不到 h1/a1，所以这两个串不可能是普通王着法。
        move = chess.Move(f_m, (f_m // 8) * 8 + (6 if uci[2] == 'h' else 2))
    else:
        move = chess.Move(f_m, t_m)
    return move if move in board.legal_moves else None


# --------------------------------------------------------------------------
# 112 输入平面
# --------------------------------------------------------------------------
# 字节 -> 8 个 float 的查表：一个 bitboard 字节覆盖 8 个格（一个横线），
# 比逐格 Python 循环快约一个数量级（Maia 每步要编 112 个平面）。
_BYTE_LUT = np.zeros((256, 8), dtype=np.float32)
for _v in range(256):
    for _j in range(8):
        _BYTE_LUT[_v, _j] = 1.0 if (_v >> _j) & 1 else 0.0


def _fill(plane: np.ndarray, mask: int, flip: bool) -> None:
    if not mask:
        return
    for i, byte in enumerate(mask.to_bytes(8, 'little')):
        if byte:
            plane[(7 - i) if flip else i] = _BYTE_LUT[byte]


def encode_input(board, history, repetition_counts=None) -> np.ndarray:
    """把一个局面编成 lc0 的 112 平面。

    ``board``    当前局面（行棋方即"我方"）
    ``history``   对局历史，``history[-1]`` 必须是 ``board``；只为前 8 步编码
    ``repetition_counts`` 可选，``{position_key: 出现次数}``（含当前这一次）；
                  不给则按历史列表现算
    """
    import chess
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK
    ours, theirs = board.turn, not board.turn

    if repetition_counts is None:
        repetition_counts = {}
        for b in history:
            k = b._transposition_key()
            repetition_counts[k] = repetition_counts.get(k, 0) + 1

    n_hist = min(len(history), MOVE_HISTORY)
    for i in range(n_hist):
        b = history[len(history) - 1 - i]
        base = i * PLANES_PER_BOARD
        # lc0 对每个历史局面都以"该局面自己的行棋方在底部"呈现（encoder.cc 里 flip
        # 每步交替，等价于按 b.turn 定向），所以翻转要按每个历史局面的行棋方算，
        # 不能对所有历史步用同一个翻转。
        bflip = b.turn == chess.BLACK
        for j, pt in enumerate(_PIECE_ORDER):
            _fill(planes[base + j], b.pieces_mask(pt, ours), bflip)
        for j, pt in enumerate(_PIECE_ORDER):
            _fill(planes[base + 6 + j], b.pieces_mask(pt, theirs), bflip)
        # 重复平面：该局面在此之前出现过
        if repetition_counts.get(b._transposition_key(), 1) >= 2:
            planes[base + 12, :, :] = 1.0

    # 辅助平面 104..111
    if board.has_queenside_castling_rights(ours):
        planes[AUX_PLANE_BASE + 0, :, :] = 1.0
    if board.has_kingside_castling_rights(ours):
        planes[AUX_PLANE_BASE + 1, :, :] = 1.0
    if board.has_queenside_castling_rights(theirs):
        planes[AUX_PLANE_BASE + 2, :, :] = 1.0
    if board.has_kingside_castling_rights(theirs):
        planes[AUX_PLANE_BASE + 3, :, :] = 1.0
    if flip:                                   # we_are_black
        planes[AUX_PLANE_BASE + 4, :, :] = 1.0
    planes[AUX_PLANE_BASE + 5, :, :] = float(board.halfmove_clock)
    # 106 以前是 movecount 平面，现在恒 0；107 恒 1 帮助网络找棋盘边界
    planes[AUX_PLANE_BASE + 7, :, :] = 1.0
    return planes


# --------------------------------------------------------------------------
# protobuf 极简解析（只为 net.proto，不引 protobuf 依赖）
# --------------------------------------------------------------------------
def _varint(buf: bytes, i: int):
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf: bytes):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
            yield f, wt, v
        elif wt == 2:
            ln, i = _varint(buf, i)
            yield f, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield f, wt, buf[i:i + 4]
            i += 4
        elif wt == 1:
            yield f, wt, buf[i:i + 8]
            i += 8
        else:
            raise MaiaError(f'net.proto 出现不支持的 wire type {wt}')


def _sub(buf: bytes, field: int) -> bytes:
    for f, wt, v in _fields(buf):
        if f == field and wt == 2:
            return v
    return b''


def _subs(buf: bytes, field: int):
    return [v for f, wt, v in _fields(buf) if f == field and wt == 2]


def _u32(buf: bytes, field: int, default: int = 0) -> int:
    for f, wt, v in _fields(buf):
        if f == field and wt == 0:
            return v
    return default


def _f32(buf: bytes, field: int, default: float = 0.0) -> float:
    for f, wt, v in _fields(buf):
        if f == field and wt == 5:
            return struct.unpack('<f', v)[0]
    return default


_LAYER_ENCODING = {0: 'UNKNOWN', 1: 'LINEAR16', 2: 'FLOAT16', 3: 'BFLOAT16'}


def _decode_layer(buf: bytes) -> Optional[np.ndarray]:
    """LINEAR16 -> float32；空层返回 None。"""
    if not buf:
        return None
    mn = _f32(buf, 1, 0.0)
    mx = _f32(buf, 2, 0.0)
    params = _sub(buf, 3)
    enc = _u32(buf, 4, 0)
    if not params:
        return None
    if enc not in (0, 1):
        raise MaiaError(f'Layer 编码 {_LAYER_ENCODING.get(enc, enc)} 未实现'
                        f'（只支持 LINEAR16）')
    u16 = np.frombuffer(params, dtype='<u2').astype(np.float32)
    return u16 / np.float32(65535.0) * np.float32(mx - mn) + np.float32(mn)


# --------------------------------------------------------------------------
# 权重容器 + BN 折叠
# --------------------------------------------------------------------------
_BN_EPS = 1e-5


class _ConvBlock:
    __slots__ = ('weights', 'biases')

    def __init__(self, buf: bytes):
        w = _decode_layer(_sub(buf, 1))
        b = _decode_layer(_sub(buf, 2))
        means = _decode_layer(_sub(buf, 3))
        stddivs = _decode_layer(_sub(buf, 4))
        gammas = _decode_layer(_sub(buf, 5))
        betas = _decode_layer(_sub(buf, 6))
        if w is None or w.size == 0:
            self.weights = None
            self.biases = None
            return
        if betas is None or betas.size == 0:
            betas = np.zeros_like(means) if means is not None else None
        if gammas is None or gammas.size == 0:
            gammas = np.ones_like(means) if means is not None else None
        if b is None or b.size == 0:
            b = np.zeros_like(means) if means is not None else None
        if means is None or means.size == 0:
            # 无 batch norm：weights/biases 原样用
            self.weights = w.astype(np.float32)
            self.biases = None if b is None else b.astype(np.float32)
            return
        # 折叠 batch norm（照抄 network_legacy.cc 的顺序）
        gammas = gammas / np.sqrt(stddivs + np.float32(_BN_EPS))
        means = means - b
        out = b.shape[0]
        per_out = w.size // out
        w = w.reshape(out, per_out) * gammas[:, None]
        b = -gammas * means + betas
        self.weights = w.reshape(-1).astype(np.float32)
        self.biases = b.astype(np.float32)


class _SE:
    __slots__ = ('w1', 'b1', 'w2', 'b2')

    def __init__(self, buf: bytes):
        self.w1 = _decode_layer(_sub(buf, 1))
        self.b1 = _decode_layer(_sub(buf, 2))
        self.w2 = _decode_layer(_sub(buf, 3))
        self.b2 = _decode_layer(_sub(buf, 4))


class _Residual:
    __slots__ = ('conv1', 'conv2', 'se')

    def __init__(self, buf: bytes):
        self.conv1 = _ConvBlock(_sub(buf, 1))
        self.conv2 = _ConvBlock(_sub(buf, 2))
        self.se = _SE(_sub(buf, 3))


class MaiaWeights:
    """从 ``.pb.gz`` 解析出的、已折叠 BN 的权重。"""

    def __init__(self, path: str):
        import os
        if not os.path.exists(path):
            raise MaiaError(f'Maia 权重不存在: {path}')
        raw = gzip.decompress(open(path, 'rb').read())
        magic = None
        for f, wt, v in _fields(raw):
            if f == 1 and wt == 5:
                magic = struct.unpack('<I', v)[0]
                break
        if magic != 0x1C0:
            raise MaiaError(f'{path} 不是 lc0 权重文件（magic={magic}，期望 448）')

        fmt = _sub(raw, 4)
        nf = _sub(fmt, 2)
        self.input_format = _u32(nf, 1, 0)
        self.output_format = _u32(nf, 2, 0)
        self.network_format = _u32(nf, 3, 0)
        self.policy_format = _u32(nf, 4, 0)
        self.value_format = _u32(nf, 5, 0)
        self.moves_left_format = _u32(nf, 6, 0)
        self.default_activation = _u32(nf, 7, 0)

        if self.input_format != 1:
            raise MaiaError(f'只支持 INPUT_CLASSICAL_112_PLANE，实得 {self.input_format}')
        if self.policy_format != 2:
            raise MaiaError(f'只支持 POLICY_CONVOLUTION，实得 {self.policy_format}')
        if self.value_format != 2:
            raise MaiaError(f'只支持 VALUE_WDL，实得 {self.value_format}')
        if self.moves_left_format != 0:
            raise MaiaError('该权重带 moves_left 头，本移植未实现')
        if self.default_activation != 0:
            raise MaiaError(f'只支持默认 RELU 激活，实得 {self.default_activation}'
                            f'（1=MISH）')

        w = _sub(raw, 10)
        if not w:
            raise MaiaError(f'{path} 没有 weights 字段（可能是 ONNX 权重）')

        self.input = _ConvBlock(_sub(w, 1))
        self.residual = [_Residual(r) for r in _subs(w, 2)]
        self.policy1 = _ConvBlock(_sub(w, 11))
        self.policy = _ConvBlock(_sub(w, 3))
        self.value = _ConvBlock(_sub(w, 6))
        self.ip1_val_w = _decode_layer(_sub(w, 7))
        self.ip1_val_b = _decode_layer(_sub(w, 8))
        self.ip2_val_w = _decode_layer(_sub(w, 9))
        self.ip2_val_b = _decode_layer(_sub(w, 10))

        # 形状自检（按实测维度，防止权重文件不匹配时静默算错）
        self.channels = int(self.input.biases.shape[0])
        if self.channels != CHANNELS:
            raise MaiaError(f'通道数 {self.channels} != 预期的 {CHANNELS}')
        if len(self.residual) != RESIDUAL_BLOCKS:
            raise MaiaError(f'残差块数 {len(self.residual)} != 预期的 {RESIDUAL_BLOCKS}')
        exp = {
            'input.weights': self.channels * NUM_PLANES * 9,
            'policy.weights': POLICY_PADDED_PLANES * self.channels * 9,
            'policy.biases': POLICY_PADDED_PLANES,
            'value.weights': VALUE_PLANES * self.channels,
            'ip1_val_w': VALUE_HIDDEN * VALUE_PLANES * 64,
            'ip2_val_w': 3 * VALUE_HIDDEN,
        }
        got = {
            'input.weights': self.input.weights.size,
            'policy.weights': self.policy.weights.size,
            'policy.biases': self.policy.biases.size,
            'value.weights': self.value.weights.size,
            'ip1_val_w': self.ip1_val_w.size,
            'ip2_val_w': self.ip2_val_w.size,
        }
        for k in exp:
            if got[k] != exp[k]:
                raise MaiaError(f'{k} 大小 {got[k]} != 预期 {exp[k]}')


# --------------------------------------------------------------------------
# PyTorch 网络
# --------------------------------------------------------------------------
class MaiaNet:
    """SE-ResNet（lc0 legacy 口径），batched 前向。

    ``forward(planes)`` 入参 ``(N, 112, 8, 8)`` float32，平面索引是
    ``[rank, file]``（rank 0 = 第 1 横线，即卷积图像的第 0 行）。
    返回 ``(policy_logits(N, 1858), wdl(N, 3))``；``wdl`` 已过 softmax，
    顺序是 win/draw/loss（行棋方视角）。
    """

    def __init__(self, weights: MaiaWeights, device='cpu'):
        import torch
        import torch.nn.functional as F
        # 435K 参数的小网：单线程最快，而且批量对弈时多个 worker 进程各自一份，
        # 默认开满物理核会把机器啃到负载 100+（实测 8 worker × 20 线程）。
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self._torch = torch
        self._F = F
        self.device = device
        C = weights.channels

        def conv(w, b, k):
            """lc0 卷积权重布局是 ``[out][in][k][k]``（见 winograd_filter.cc 的
            ``f[o * channels * 9 + c * 9 + ky * 3 + kx]``），与 PyTorch 一致。"""
            out = b.shape[0]
            per = w.size // out
            if per % (k * k):
                raise MaiaError(f'卷积权重 {w.size} 除不尽 out={out} k={k}')
            t = torch.from_numpy(w.reshape(out, per // (k * k), k, k).copy())
            return t.to(device), torch.from_numpy(b.copy()).to(device)

        def fc(w, b):
            t = torch.from_numpy(w.reshape(b.shape[0], w.size // b.shape[0])
                                 .T.copy()).to(device)
            return t, torch.from_numpy(b.copy()).to(device)

        self.in_w, self.in_b = conv(weights.input.weights, weights.input.biases, 3)
        self.blocks = []
        for r in weights.residual:
            se = r.se
            se_fc = int(se.b1.shape[0])
            self.blocks.append({
                'c1w': conv(r.conv1.weights, r.conv1.biases, 3),
                'c2w': conv(r.conv2.weights, r.conv2.biases, 3),
                'se_w1': fc(se.w1, se.b1),
                'se_w2': fc(se.w2, se.b2),
                'se_fc': se_fc,
            })
        self.p1_w, self.p1_b = conv(weights.policy1.weights, weights.policy1.biases, 3)
        self.pol_w, self.pol_b = conv(weights.policy.weights, weights.policy.biases, 3)
        self.val_w, self.val_b = conv(weights.value.weights, weights.value.biases, 1)
        self.v1_w, self.v1_b = fc(weights.ip1_val_w, weights.ip1_val_b)
        self.v2_w, self.v2_b = fc(weights.ip2_val_w, weights.ip2_val_b)

        # (plane, square) -> policy index，-1 处丢弃。
        # lc0 的 head_buffer 布局是 [plane][square]，square = rank*8 + file，
        # 与张量的 [plane, rank, file] 一致。
        flat = np.array(_CONV_MAP, dtype=np.int64).reshape(POLICY_PLANES, 64)
        valid = flat >= 0
        self._sel = torch.from_numpy(flat[valid]).to(device)     # (1858,)
        p_idx, s_idx = np.nonzero(valid)
        self._p_idx = torch.from_numpy(p_idx.astype(np.int64)).to(device)
        self._r_idx = torch.from_numpy((s_idx // 8).astype(np.int64)).to(device)
        self._f_idx = torch.from_numpy((s_idx % 8).astype(np.int64)).to(device)

    @staticmethod
    def _relu(x):
        return x.clamp_min(0.0)

    def forward(self, planes):
        torch, F = self._torch, self._F
        x = planes if isinstance(planes, torch.Tensor) else torch.as_tensor(planes)
        x = x.to(self.device, dtype=torch.float32)

        x = self._relu(F.conv2d(x, self.in_w, self.in_b, padding=1))
        for blk in self.blocks:
            res = x
            (c1w, c1b) = blk['c1w']
            y = self._relu(F.conv2d(x, c1w, c1b, padding=1))
            (c2w, c2b) = blk['c2w']
            y = F.conv2d(y, c2w, c2b, padding=1)
            # SE：全局平均池化（含 conv2 偏置）-> FC -> 2C 缩放，再加残差
            pooled = y.mean(dim=(2, 3)) + c2b
            (w1, b1) = blk['se_w1']
            h = self._relu(pooled @ w1 + b1)
            (w2, b2) = blk['se_w2']
            se = h @ w2 + b2
            gamma = torch.sigmoid(se[:, :self.channels()])
            beta = se[:, self.channels():] + gamma * c2b
            y = self._relu(gamma[:, :, None, None] * y
                           + beta[:, :, None, None] + res)
            x = y

        # policy 头：3x3 卷积 + 偏置 + RELU，再 3x3 卷积到 80 平面（无激活），
        # 然后按 (plane, square) -> policy index 收集
        y = self._relu(F.conv2d(x, self.p1_w, self.p1_b, padding=1))
        y = F.conv2d(y, self.pol_w, self.pol_b, padding=1)      # (N, 80, 8, 8)
        y = y[:, :POLICY_PLANES]                                # 丢掉 7 个 padding 平面
        policy = torch.full((y.shape[0], NUM_POLICY), float('-inf'),
                            device=self.device)
        policy[:, self._sel] = y[:, self._p_idx, self._r_idx, self._f_idx]

        # value 头：1x1 卷积 + 偏置 + RELU，flatten 后 FC -> 3，softmax
        v = self._relu(F.conv2d(x, self.val_w, self.val_b))
        v = v.flatten(1)
        v = self._relu(v @ self.v1_w + self.v1_b)
        wdl = F.softmax(v @ self.v2_w + self.v2_b, dim=1)
        return policy, wdl

    def channels(self):
        return self.in_w.shape[0]

    def eval(self):
        return self

    def to(self, device):
        """把全部张量搬到另一设备，返回新对象（原对象不变）。"""
        import torch
        if device == self.device:
            return self
        new = object.__new__(MaiaNet)
        new._torch, new._F, new.device = self._torch, self._F, device
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                setattr(new, k, v.to(device))
            elif k == 'blocks':
                moved = []
                for blk in v:
                    moved.append({
                        'se_fc': blk['se_fc'],
                        'c1w': tuple(t.to(device) for t in blk['c1w']),
                        'c2w': tuple(t.to(device) for t in blk['c2w']),
                        'se_w1': tuple(t.to(device) for t in blk['se_w1']),
                        'se_w2': tuple(t.to(device) for t in blk['se_w2']),
                    })
                setattr(new, k, moved)
            else:
                setattr(new, k, v)
        return new


def load_maia(path: str, device='cpu') -> MaiaNet:
    """加载一个 Maia 档位权重并建好网络。"""
    return MaiaNet(MaiaWeights(path), device=device)
