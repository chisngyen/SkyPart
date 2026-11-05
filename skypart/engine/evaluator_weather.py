"""WeatherPrompt 10-condition evaluator (D2S + S2D, all 3 datasets).

Loops the 10 imgaug augmenters from `data/weather.py` over a base evaluator
function, returning per-condition R@1/AP for both directions plus the mean.

Protocol: drone images are corrupted in both directions; satellite stays clean.
"""
import torch

from ..config import CFG
from ..data.weather import WEATHER_CONDITIONS, build_weather_augmenters


@torch.no_grad()
def evaluate_weather(model, test_ds, device, base_evaluate_fn, cfg=CFG):
    """`base_evaluate_fn(model, test_ds, device, cfg, weather_drone_aug=...)` must
    accept the `weather_drone_aug` kwarg and return {'D2S': {...}, 'S2D': {...}}.

    Returns:
        {
            'per_condition': {cond: {'D2S': {...}, 'S2D': {...}}},
            'mean': {'D2S': {'R@1': ..., 'AP': ...},
                     'S2D': {'R@1': ..., 'AP': ...}},
        }
    """
    augs = build_weather_augmenters()
    assert len(augs) == len(WEATHER_CONDITIONS) == 10
    model.eval()

    per_cond = {}
    for cond, aug in zip(WEATHER_CONDITIONS, augs):
        print(f"\n  >> Weather condition: {cond}")
        res = base_evaluate_fn(model, test_ds, device, cfg, weather_drone_aug=aug)
        per_cond[cond] = res

    def _avg(direction, key):
        vals = [per_cond[c][direction][key] for c in WEATHER_CONDITIONS]
        return sum(vals) / len(vals)

    mean = {
        'D2S': {'R@1': _avg('D2S', 'R@1'), 'AP': _avg('D2S', 'AP')},
        'S2D': {'R@1': _avg('S2D', 'R@1'), 'AP': _avg('S2D', 'AP')},
    }

    print(f"\n  ============== Weather summary (10 conditions) ==============")
    print(f"  D2S Mean  R@1 {mean['D2S']['R@1']:6.2f}  AP {mean['D2S']['AP']:6.2f}")
    print(f"  S2D Mean  R@1 {mean['S2D']['R@1']:6.2f}  AP {mean['S2D']['AP']:6.2f}")
    for cond in WEATHER_CONDITIONS:
        d = per_cond[cond]['D2S']
        s = per_cond[cond]['S2D']
        print(f"  {cond:>10s}  D2S R@1 {d['R@1']:6.2f} AP {d['AP']:6.2f}"
              f"  |  S2D R@1 {s['R@1']:6.2f} AP {s['AP']:6.2f}")
    return {'per_condition': per_cond, 'mean': mean}
