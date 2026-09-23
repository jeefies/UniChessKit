"""管线：只依赖 api 与 kit 内部组件，不 import 任何引擎。"""
from .match import GameTask, MatchConfig, SprtConfig, play_game, plan_games, run_match, summarize
from .selfplay import (SelfPlayConfig, SelfPlayTask, game_seed_sequence, load_book_lines,
                       plan_selfplay, play_selfplay_game, run_selfplay)

__all__ = ["GameTask", "MatchConfig", "SprtConfig", "play_game", "plan_games", "run_match",
           "summarize", "SelfPlayConfig", "SelfPlayTask", "game_seed_sequence", "load_book_lines",
           "plan_selfplay", "play_selfplay_game", "run_selfplay"]
