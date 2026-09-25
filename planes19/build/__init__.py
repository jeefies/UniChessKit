"""T/R 训练数据构建：Lichess 评估库 / PGN → 96 字节分片，Stockfish MultiPV 标注，分片验收。

    python -m Kit data evals    --src lichess_db_eval.jsonl.zst --out <dir>
    python -m Kit data pgn      --src lichess_db_standard_rated_YYYY-MM.pgn.zst --out <dir>
    python -m Kit data annotate --fens fens.txt --out <dir> --stockfish <path>
    python -m Kit data check    <shard.bin> [...]
"""
