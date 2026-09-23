"""T/R 共用的 19 平面编码（可选组件；S 不使用）。"""
from .encoding import (INPUT_SHAPE, NUM_PLANES, POLICY_SIZE, PROMO_PIECES, PROMO_SIZE, encode,
                       move_to_index, move_to_promo_index, orient, orient_move,
                       priors_from_policy, repetitions_of, unorient_move)
from .expander import BatchFnEvaluator, Planes19Expander

__all__ = ["INPUT_SHAPE", "NUM_PLANES", "POLICY_SIZE", "PROMO_PIECES", "PROMO_SIZE", "encode",
           "move_to_index", "move_to_promo_index", "orient", "orient_move",
           "priors_from_policy", "repetitions_of", "unorient_move",
           "BatchFnEvaluator", "Planes19Expander"]
