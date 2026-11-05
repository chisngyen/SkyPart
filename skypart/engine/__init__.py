"""Training loop and per-dataset evaluators.

- evaluate_sues200   -- evaluator_sues200.py
- evaluate_uni1652   -- evaluator_uni1652.py
- evaluate_denseuav  -- evaluator_denseuav.py
- evaluate_weather   -- evaluator_weather.py (ten conditions over a base evaluator)
"""
from .trainer import EMAModel, set_seed, train_one_epoch, cosine_lr
from .evaluator_sues200 import evaluate as evaluate_sues200
from .evaluator_uni1652 import evaluate as evaluate_uni1652
from .evaluator_denseuav import evaluate as evaluate_denseuav
from .evaluator_weather import evaluate_weather
from ..utils import print_model_complexity

__all__ = ["EMAModel", "set_seed", "train_one_epoch", "cosine_lr",
           "evaluate_sues200", "evaluate_uni1652",
           "evaluate_denseuav", "evaluate_weather", "print_model_complexity"]
