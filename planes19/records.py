"""T/R 共用的定长训练记录与向量化解码。

两种记录，棋盘字段完全相同（前 70 字节），读取端统一：

**监督记录 ``RECORD_DTYPE``（96 字节，冻结）**——Stockfish 蒸馏分片 ``evals_*.bin`` / PGN 分片::

    pawns/knights/bishops/rooks/queens/kings  6 x u64   棋子类型占位
    occ_white / occ_black                     2 x u64   颜色占位
    castling  u8   低 4 位 = K Q k q（1/2/4/8）
    ep        u8   吃过路兵目标格 0-63，255 = 无
    halfmove  u8   五十步计数，255 封顶
    side      u8   0 = 白方行棋，1 = 黑方行棋
    rep       u8   重复次数 0/1/2
    promo     u8   最佳走法的升变索引 0-3，255 = 非升变
    policy_move  5 x u16   MultiPV 前 5 的走法索引（行棋方视角 from*64+to）
    policy_prob  5 x u16   对应概率，/65535
    wdl          3 x u16   胜/和/负概率（行棋方视角），/65535

**自对弈记录 ``SELFPLAY_DTYPE``（160 字节）**——kit 自对弈 sink 写出::

    棋盘字段      同上 70 字节
    visit_move   16 x u16   访问最多的 16 个着法索引（行棋方视角），按访问数降序
    visit_prob   16 x u16   对应访问占比，/65535（截断后可能和 < 1，解码时归一）
    z            i8         终局结果（行棋方视角）+1 / 0 / -1
    flags        u8         bit0 = 截断局（超 max_plies，z 记 0）
    q            f32        根节点搜索 Q（行棋方视角，[-1, 1]）
    wdl          3 x u16    搜索 WDL 估计（无则全 0）
    game         u32        局号
    ply          u16        局内半步序号
    _pad         8 x u8

分片文件名决定格式：``*.bin`` 为 96 字节监督记录，``*.sp.bin`` 为 160 字节自对弈记录
（``shard_dtype``）。两者都能 ``decode_batch``；``decode_targets`` 对自对弈记录返回
访问分布与由 z（可按 ``q_ratio`` 混入 q）构成的 WDL 目标。

历史坑：旧 ``build_evals.py`` 把 Lichess 评估库的易位写法 e1h1 / e1a1 直接算了索引，
``data/shards_evals`` 里 99.9% 的易位标签落在推理端永远读不到的 4*64+7 / 4*64+0。
R 训练在解码时修（``repair_castling=True``）；旧 T 训练没修。新配置一律 True，
逐位对照旧 T 轨迹时用 False。
"""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from .encoding import NUM_PLANES, POLICY_SIZE

_BOARD_FIELDS = [
    ("pawns",       "<u8"), ("knights", "<u8"), ("bishops", "<u8"),
    ("rooks",       "<u8"), ("queens",  "<u8"), ("kings",   "<u8"),
    ("occ_white",   "<u8"), ("occ_black", "<u8"),
    ("castling",    "u1"), ("ep",   "u1"), ("halfmove", "u1"),
    ("side",        "u1"), ("rep",  "u1"), ("promo",    "u1"),
]

RECORD_DTYPE = np.dtype(_BOARD_FIELDS + [
    ("policy_move", "<u2", (5,)),
    ("policy_prob", "<u2", (5,)),
    ("wdl",         "<u2", (3,)),
])
assert RECORD_DTYPE.itemsize == 96, RECORD_DTYPE.itemsize

SP_TOPK = 16
SELFPLAY_DTYPE = np.dtype(_BOARD_FIELDS + [
    ("visit_move", "<u2", (SP_TOPK,)),
    ("visit_prob", "<u2", (SP_TOPK,)),
    ("z",          "i1"),
    ("flags",      "u1"),
    ("q",          "<f4"),
    ("wdl",        "<u2", (3,)),
    ("game",       "<u4"),
    ("ply",        "<u2"),
    ("_pad",       "u1", (8,)),
])
assert SELFPLAY_DTYPE.itemsize == 160, SELFPLAY_DTYPE.itemsize

NO_EP = 255
NO_PROMO = 255
SCALE = 65535
FLAG_TRUNCATED = 1

# 平面偏移，与 encoding.encode 的布局一致
_P_OWN, _P_OPP = 0, 6
_P_CASTLE, _P_EP, _P_HALF, _P_REP = 12, 16, 17, 18


# ---------------------------------------------------------------- 单条记录 <-> 棋盘

def _fill_board(rec, board: chess.Board, rep: int, promo) -> None:
    rec["pawns"] = board.pawns
    rec["knights"] = board.knights
    rec["bishops"] = board.bishops
    rec["rooks"] = board.rooks
    rec["queens"] = board.queens
    rec["kings"] = board.kings
    rec["occ_white"] = board.occupied_co[chess.WHITE]
    rec["occ_black"] = board.occupied_co[chess.BLACK]
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    rec["castling"] = castling
    rec["ep"] = NO_EP if board.ep_square is None else board.ep_square
    rec["halfmove"] = min(board.halfmove_clock, 255)
    rec["side"] = 0 if board.turn == chess.WHITE else 1
    rec["rep"] = min(rep, 2)
    rec["promo"] = NO_PROMO if promo is None else promo


def _q(p: float) -> int:
    return int(round(max(0.0, min(1.0, p)) * SCALE))


def board_to_record(board: chess.Board, *, policy=None, wdl=(0.0, 1.0, 0.0),
                    promo=None, rep: int = 0) -> np.void:
    """局面 + 标签 → 一条 96 字节监督记录。policy 为 [(走法索引, 概率), ...] 最多 5 条。"""
    rec = np.zeros(1, dtype=RECORD_DTYPE)[0]
    _fill_board(rec, board, rep, promo)
    if policy:
        for i, (mv, pr) in enumerate(policy[:5]):
            rec["policy_move"][i] = mv
            rec["policy_prob"][i] = _q(pr)
    for i, v in enumerate(wdl):
        rec["wdl"][i] = _q(v)
    return rec


def selfplay_record(board: chess.Board, *, visits, z: int, q: float, game: int, ply: int,
                    promo=None, rep: int = 0, wdl=None, truncated: bool = False) -> np.void:
    """局面 + 搜索结果 → 一条 160 字节自对弈记录。

    visits：[(走法索引, 访问数或占比), ...]；按访问降序取前 16，占比按全部访问归一后再截断。
    """
    rec = np.zeros(1, dtype=SELFPLAY_DTYPE)[0]
    _fill_board(rec, board, rep, promo)
    items = sorted(((int(m), float(v)) for m, v in visits if v > 0), key=lambda t: (-t[1], t[0]))
    total = sum(v for _, v in items)
    for i, (mv, v) in enumerate(items[:SP_TOPK]):
        rec["visit_move"][i] = mv
        rec["visit_prob"][i] = _q(v / total)
    if z not in (-1, 0, 1):
        raise ValueError(f"z 应为 -1/0/1，得到 {z}")
    rec["z"] = z
    rec["flags"] = FLAG_TRUNCATED if truncated else 0
    rec["q"] = np.float32(q)
    if wdl is not None:
        for i, v in enumerate(wdl):
            rec["wdl"][i] = _q(v)
    rec["game"] = game
    rec["ply"] = ply
    return rec


def record_to_board(rec) -> chess.Board:
    """记录 → chess.Board（直接写位棋盘；两种记录都适用）。无走子栈，fullmove 记 1。"""
    board = chess.Board(None)
    board.pawns = int(rec["pawns"])
    board.knights = int(rec["knights"])
    board.bishops = int(rec["bishops"])
    board.rooks = int(rec["rooks"])
    board.queens = int(rec["queens"])
    board.kings = int(rec["kings"])
    board.occupied_co[chess.WHITE] = int(rec["occ_white"])
    board.occupied_co[chess.BLACK] = int(rec["occ_black"])
    board.occupied = int(rec["occ_white"]) | int(rec["occ_black"])
    board.promoted = 0
    c = int(rec["castling"])
    mask = 0
    if c & 1:
        mask |= chess.BB_H1
    if c & 2:
        mask |= chess.BB_A1
    if c & 4:
        mask |= chess.BB_H8
    if c & 8:
        mask |= chess.BB_A8
    board.castling_rights = mask
    ep = int(rec["ep"])
    board.ep_square = None if ep == NO_EP else ep
    board.halfmove_clock = int(rec["halfmove"])
    board.turn = chess.WHITE if int(rec["side"]) == 0 else chess.BLACK
    board.fullmove_number = 1
    return board


def record_policy(rec) -> list:
    """监督记录的策略软标签（去掉概率为 0 的空位）；自对弈记录返回访问分布。"""
    if "visit_move" in rec.dtype.names:
        pairs = zip(rec["visit_move"], rec["visit_prob"])
    else:
        pairs = zip(rec["policy_move"], rec["policy_prob"])
    return [(int(mv), float(pr) / SCALE) for mv, pr in pairs if pr]


def record_wdl(rec) -> tuple:
    return tuple(float(v) / SCALE for v in rec["wdl"])


# ---------------------------------------------------------------- 分片文件

def shard_dtype(path) -> np.dtype:
    return SELFPLAY_DTYPE if str(path).endswith(".sp.bin") else RECORD_DTYPE


def open_shard(path, mode: str = "r") -> np.memmap:
    """内存映射打开一个分片（格式由文件名决定）。大小不是记录长度整数倍时报错（写了一半的分片）。"""
    dt = shard_dtype(path)
    size = Path(path).stat().st_size
    if size % dt.itemsize:
        raise ValueError(f"{path} 大小 {size} 不是 {dt.itemsize} 的整数倍")
    return np.memmap(path, dtype=dt, mode=mode)


def shard_len(path) -> int:
    dt = shard_dtype(path)
    return Path(path).stat().st_size // dt.itemsize


# ---------------------------------------------------------------- 向量化解码

def decode_batch(recs: np.ndarray) -> np.ndarray:
    """一批记录 → (N, 19, 8, 8) float32，与 ``encoding.encode`` 逐位一致。纯 numpy 位运算。"""
    n = len(recs)
    planes = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)
    occ_w = recs["occ_white"].astype(np.uint64)
    occ_b = recs["occ_black"].astype(np.uint64)
    side = recs["side"].astype(bool)          # True = 黑方行棋
    own = np.where(side, occ_b, occ_w)
    opp = np.where(side, occ_w, occ_b)
    bits = np.arange(64, dtype=np.uint64)
    for i, field in enumerate(("pawns", "knights", "bishops", "rooks", "queens", "kings")):
        bb = recs[field].astype(np.uint64)
        present = ((bb[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        is_own = ((own[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        is_opp = ((opp[:, None] >> bits[None, :]) & np.uint64(1)).astype(bool)
        planes[:, _P_OWN + i] = (present & is_own).astype(np.float32)
        planes[:, _P_OPP + i] = (present & is_opp).astype(np.float32)
    c = recs["castling"].astype(np.uint8)
    own_k = np.where(side, (c >> 2) & 1, c & 1)
    own_q = np.where(side, (c >> 3) & 1, (c >> 1) & 1)
    opp_k = np.where(side, c & 1, (c >> 2) & 1)
    opp_q = np.where(side, (c >> 1) & 1, (c >> 3) & 1)
    for off, v in enumerate((own_k, own_q, opp_k, opp_q)):
        planes[:, _P_CASTLE + off] = v.astype(np.float32)[:, None]
    ep = recs["ep"].astype(np.int16)
    rows = np.nonzero(ep != NO_EP)[0]
    if len(rows):
        planes[rows, _P_EP, ep[rows]] = 1.0
    planes[:, _P_HALF] = (np.minimum(recs["halfmove"], 100) / 100.0).astype(np.float32)[:, None]
    planes[:, _P_REP] = (np.minimum(recs["rep"], 2) / 2.0).astype(np.float32)[:, None]
    planes = planes.reshape(n, NUM_PLANES, 8, 8)
    flip = np.nonzero(side)[0]                # 黑方行棋：上下翻转（颜色已对调）
    if len(flip):
        planes[flip] = planes[flip][:, :, ::-1, :]
    return planes


# 镜像后（永远是我方在下方）王在 e1 = 4 号格
_E1 = 4
_CASTLE_FIX = (
    (_E1 * 64 + 7, _E1 * 64 + 6),   # e1h1（王吃己车写法）→ e1g1
    (_E1 * 64 + 0, _E1 * 64 + 2),   # e1a1                → e1c1
)


def repair_castling(policy: np.ndarray, recs: np.ndarray) -> int:
    """把易位标签从「王吃己车」写法搬到「王走两格」写法，返回搬动条数（原地修改 policy）。

    判据是「我方王在 e1」：此时 4*64+7 只可能是易位（王一步到不了 h1）；王在别处时
    e1 车走 h1 也是这个索引，是合法普通着法，绝不能搬。新建的分片不会命中，自然成为空操作。
    """
    kings = recs["kings"].astype(np.uint64)
    white = recs["occ_white"].astype(np.uint64)
    black = recs["occ_black"].astype(np.uint64)
    is_white = recs["side"] == 0
    on_e1 = np.where(is_white,
                     (kings & white & np.uint64(1 << 4)) != 0,
                     (kings & black & np.uint64(1 << 60)) != 0)
    rows = np.flatnonzero(on_e1)
    moved = 0
    for src, dst in _CASTLE_FIX:
        r = rows[policy[rows, src] > 0]
        if r.size:
            policy[r, dst] += policy[r, src]
            policy[r, src] = 0.0
            moved += int(r.size)
    return moved


def _sparse_policy(moves: np.ndarray, probs: np.ndarray, recs, repair: bool) -> np.ndarray:
    n, k = moves.shape
    policy = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    mv = moves.astype(np.int32)
    pb = probs.astype(np.float32) / np.float32(SCALE)
    rows = np.repeat(np.arange(n), k)
    np.add.at(policy, (rows, mv.ravel()), pb.ravel() * (pb.ravel() > 0))
    if repair:
        repair_castling(policy, recs)
    s = policy.sum(axis=1, keepdims=True)
    np.divide(policy, s, out=policy, where=s > 0)
    return policy


def decode_targets(recs: np.ndarray, *, repair_castling: bool = True, q_ratio: float = 0.0):
    """记录 → (policy[N,4096], promo[N] int64（-100 = ignore）, wdl[N,3])。

    监督记录：MultiPV 软标签与 Stockfish WDL（各自归一）。
    自对弈记录：访问分布（归一）；WDL = (1-q_ratio)·onehot(z) + q_ratio·(q 折成的 W/L 二元分布)。
    """
    sp = "visit_move" in recs.dtype.names
    if sp:
        policy = _sparse_policy(recs["visit_move"], recs["visit_prob"], recs, False)
    else:
        policy = _sparse_policy(recs["policy_move"], recs["policy_prob"], recs, repair_castling)
    promo = recs["promo"].astype(np.int64)
    promo = np.where(promo == NO_PROMO, -100, promo)
    if sp:
        z = recs["z"].astype(np.int64)
        wdl = np.zeros((len(recs), 3), dtype=np.float32)
        wdl[np.arange(len(recs)), 1 - z] = 1.0                      # z=+1→W(0) 0→D(1) -1→L(2)
        if q_ratio:
            q = np.clip(recs["q"].astype(np.float32), -1.0, 1.0)
            wq = np.stack([(1 + q) / 2, np.zeros_like(q), (1 - q) / 2], axis=1)
            wdl = ((1.0 - q_ratio) * wdl + q_ratio * wq).astype(np.float32)
    else:
        wdl = recs["wdl"].astype(np.float32) / np.float32(SCALE)
        ws = wdl.sum(axis=1, keepdims=True)
        np.divide(wdl, ws, out=wdl, where=ws > 0)
    return policy, promo, wdl


def piece_counts(recs: np.ndarray) -> np.ndarray:
    """每条记录的子力总数（含王）。"""
    occ = recs["occ_white"].astype(np.uint64) | recs["occ_black"].astype(np.uint64)
    return np.unpackbits(occ.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.int64)


def piece_count_bucket(recs: np.ndarray, num_buckets: int) -> np.ndarray:
    """按子力数分桶（R 的 buckets 头）：32 子均分到 num_buckets 桶。"""
    cnt = piece_counts(recs)
    return np.clip((cnt - 1) * num_buckets // 32, 0, num_buckets - 1)
