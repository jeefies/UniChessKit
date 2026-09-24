"""统一训练框架（允许 import torch）：``TrainTask`` 协议 + ``Trainer``。

    python -m Kit train <config.json>

配置字段见 ``config.py``，训练循环语义见 ``trainer.py``。
"""
from .config import TrainConfig
from .trainer import CKPT_FORMAT, TrainContext, Trainer, TrainTask, build_task, run_train

__all__ = ["CKPT_FORMAT", "TrainConfig", "TrainContext", "Trainer", "TrainTask", "build_task",
           "run_train"]
