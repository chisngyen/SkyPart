"""WeatherPrompt corruption wrapper.

Ten conditions identical to WeatherPrompt (Wen et al., NeurIPS 2025):
    normal / fog / rain / snow / dark / light / fog_rain / fog_snow / rain_snow / wind

Augmenter parameters are identical to the WeatherPrompt source code (paper
Appendix, Table A2 lists the augmenter for each condition). Augmenters are
applied to DRONE images only (satellite stays clean).

Usage:
    aug = build_weather_augmenters()[idx]   # idx 0..9, 0 == identity (Normal)
    pil_corrupted = apply_weather(pil_img, aug)
    # then feed `pil_corrupted` through the test transforms.
"""
import numpy as np
from PIL import Image

# imgaug compatibility shim for NumPy 2.x.
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int':   [np.int8, np.int16, np.int32, np.int64],
        'uint':  [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others':  [bool, object, bytes, str, np.void],
    }
for _alias, _real in [('complex', complex), ('float', float),
                      ('int', int), ('bool', bool)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

try:
    import imgaug.augmenters as iaa
except ImportError as e:
    raise ImportError(
        "WeatherPrompt evaluation requires `imgaug`. Install with: pip install imgaug"
    ) from e


WEATHER_CONDITIONS = [
    "normal", "fog", "rain", "snow", "dark", "light",
    "fog_rain", "fog_snow", "rain_snow", "wind",
]


def build_weather_augmenters():
    """Return a list of 10 imgaug augmenters aligned with WEATHER_CONDITIONS.

    Index 0 is `None` (identity / Normal). Parameters are identical to the
    WeatherPrompt source code.
    """
    return [
        None,
        # Fog
        iaa.Sequential([iaa.CloudLayer(
            intensity_mean=225, intensity_freq_exponent=-2,
            intensity_coarse_scale=2, alpha_min=1.0, alpha_multiplier=0.9,
            alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            density_multiplier=0.5, seed=35)]),
        # Rain
        iaa.Sequential([
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95)]),
        # Snow
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96)]),
        # Dark
        iaa.Sequential([
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100),
                           background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991)]),
        # Light (over-exposure)
        iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)]),
        # Fog + Rain
        iaa.Sequential([
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
                           intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10,
                           alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)]),
        # Fog + Snow
        iaa.Sequential([
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
                           intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10,
                           alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)]),
        # Rain + Snow
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74)]),
        # Wind (motion blur)
        iaa.Sequential([iaa.MotionBlur(15, seed=17)]),
    ]


def apply_weather(pil_img: Image.Image, augmenter) -> Image.Image:
    """Apply an imgaug augmenter to a PIL image. Returns a new PIL image."""
    if augmenter is None:
        return pil_img
    arr = np.asarray(pil_img.convert('RGB'))
    out = augmenter(image=arr)
    return Image.fromarray(out)
