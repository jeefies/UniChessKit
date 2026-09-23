"""管线：只依赖 api 与 kit 内部组件，不 import 任何引擎。"""
from .match import GameTask, MatchConfig, SprtConfig, play_game, plan_games, run_match, summarize

__all__ = ["GameTask", "MatchConfig", "SprtConfig", "play_game", "plan_games", "run_match",
           "summarize"]
