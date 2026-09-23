"""命令行入口：python -m unichess_kit.match CONFIG.json --out runs/x/results.jsonl"""
import sys

from .pipelines.match import main

if __name__ == "__main__":
    sys.exit(main())
