"""T/R 共用的 19 平面组件（S 不使用）：编码、搜索扩展器、训练记录、数据流、自对弈 sink、数据构建。

本包顶层只导入不依赖 torch 的部分；训练相关（``losses`` / ``task``）按需 ``from Kit.planes19 import task``。
"""
from .dataset import (BatchShardDataset, CurriculumStream, Decoder, ShardSet, SpecialtyPool,
                      list_shards)
from .encoding import (INPUT_SHAPE, NUM_PLANES, POLICY_SIZE, PROMO_PIECES, PROMO_SIZE, encode,
                       index_to_move, move_to_index, move_to_promo_index, orient, orient_move,
                       priors_from_policy, repetitions_of, unorient_move)
from .expander import BatchFnEvaluator, Planes19Expander
from .factory import make_search_player_factory
from .records import (RECORD_DTYPE, SELFPLAY_DTYPE, board_to_record, decode_batch,
                      decode_targets, open_shard, record_policy, record_to_board, record_wdl,
                      selfplay_record, shard_dtype, shard_len)
from .sink import SelfPlayShardSink, game_records

__all__ = ["INPUT_SHAPE", "NUM_PLANES", "POLICY_SIZE", "PROMO_PIECES", "PROMO_SIZE", "encode",
           "index_to_move", "move_to_index", "move_to_promo_index", "orient", "orient_move",
           "priors_from_policy", "repetitions_of", "unorient_move",
           "BatchFnEvaluator", "Planes19Expander", "make_search_player_factory",
           "RECORD_DTYPE", "SELFPLAY_DTYPE", "board_to_record", "decode_batch", "decode_targets",
           "open_shard", "record_policy", "record_to_board", "record_wdl", "selfplay_record",
           "shard_dtype", "shard_len",
           "BatchShardDataset", "CurriculumStream", "Decoder", "ShardSet", "SpecialtyPool",
           "list_shards", "SelfPlayShardSink", "game_records"]
