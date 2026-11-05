"""Per-dataset dataloaders for SkyPart.

Each module follows the paper protocol:
- SUES-200: full 200-tile confusion gallery, 4 altitudes.
- University-1652: 4-folder D2S/S2D protocol, no altitude (mean-FiLM).
- DenseUAV: official full gallery (evaluator), DiscreteRotation on BOTH views (train).
- weather: WeatherPrompt 10-condition imgaug wrapper applied to drone images
  in both directions (satellite stays clean).
"""
from .sues200 import SUES200Dataset  # noqa: F401
from .university1652 import (  # noqa: F401
    University1652TrainDataset, University1652TestDataset,
)
from .denseuav import DenseUAVDataset  # noqa: F401
from .samplers import HardNegativeMiner, HardPKSampler, PKSampler  # noqa: F401
from .transforms import DiscreteRotation, get_drone_transforms, get_sat_transforms  # noqa: F401
from .weather import (  # noqa: F401
    WEATHER_CONDITIONS, build_weather_augmenters, apply_weather,
)
