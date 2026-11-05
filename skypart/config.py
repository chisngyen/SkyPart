"""Module-level configuration singleton.

Populated from a YAML file via `load_config(path)`. The same Config dataclass
is reused across SUES-200, University-1652, and DenseUAV by overriding only
the fields that differ (e.g. NUM_ALTITUDES, ALT_TO_IDX, IMG_SIZE).
"""
from dataclasses import dataclass, field, fields, asdict
from typing import List, Dict
import yaml


DEFAULT_DINOV2_REPO = (
    "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
DEFAULT_DINOV2_STUDENT_MODEL = "dinov2_vits14"
DEFAULT_DINOV2_TEACHER_MODEL = "dinov2_vitb14"


@dataclass
class Config:
    # ----- Dataset selector -----
    DATASET: str = "sues200"  # one of: sues200 / university1652 / denseuav
    DATA_ROOT: str = "./data/university1652"  # generic path used by uni1652 / denseuav

    # ----- SUES-200 -----
    SUES_ROOT: str = "./data/sues200"
    DRONE_DIR: str = "drone-view"
    SAT_DIR: str = "satellite-view"
    OUTPUT_DIR: str = "./outputs"
    ALTITUDES: List[str] = field(default_factory=lambda: ["150", "200", "250", "300"])
    ALT_TO_IDX: Dict[str, int] = field(default_factory=lambda: {"150": 0, "200": 1, "250": 2, "300": 3})
    ALT_HEIGHTS: List[int] = field(default_factory=lambda: [150, 200, 250, 300])
    NUM_ALTITUDES: int = 4
    TRAIN_LOCS: List[int] = field(default_factory=lambda: list(range(1, 121)))
    TEST_LOCS: List[int] = field(default_factory=lambda: list(range(121, 201)))
    NUM_CLASSES: int = 120

    # ----- DenseUAV-specific (override via YAML) -----
    DENSEUAV_ROOT: str = "./data/denseuav"
    HEIGHTS: List[str] = field(default_factory=lambda: ["H80", "H90", "H100"])
    DENSE_GPS_FILE: str = "Dense_GPS_ALL.txt"

    # ----- Model -----
    IMG_SIZE: int = 448
    N_PARTS: int = 12
    GATE_SPARSITY: float = 0.1
    PART_DIM: int = 256
    EMBED_DIM: int = 768
    TEACHER_DIM: int = 768
    CLUSTER_TEMP: float = 0.07
    UNFREEZE_BLOCKS: int = 6
    DINOV2_REPO: str = DEFAULT_DINOV2_REPO
    DINOV2_STUDENT_MODEL: str = DEFAULT_DINOV2_STUDENT_MODEL
    DINOV2_TEACHER_MODEL: str = DEFAULT_DINOV2_TEACHER_MODEL

    # ----- Training -----
    NUM_EPOCHS: int = 60
    EVAL_INTERVAL: int = 5
    EVAL_BATCH_SIZE: int = 64
    EVAL_NUM_WORKERS: int = 2
    P_CLASSES: int = 16
    K_SAMPLES: int = 4
    LR: float = 3e-4
    BACKBONE_LR: float = 3e-5
    WEIGHT_DECAY: float = 0.05
    WARMUP_EPOCHS: int = 5
    USE_AMP: bool = True
    SEED: int = 42
    NUM_WORKERS: int = 2

    # ----- Patch NCE -----
    PATCH_NCE_WEIGHT: float = 0.3
    PATCH_NCE_SAMPLES: int = 64

    # ----- VICReg -----
    VICREG_WEIGHT: float = 0.5
    VICREG_SIM_COEFF: float = 25.0
    VICREG_STD_COEFF: float = 25.0
    VICREG_COV_COEFF: float = 1.0

    # ----- Circle -----
    CIRCLE_MARGIN: float = 0.25
    CIRCLE_GAMMA: int = 256

    # ----- GeoPartLoss -----
    PROXY_MARGIN: float = 0.1
    PROXY_ALPHA: int = 32
    EMA_DECAY: float = 0.996
    LABEL_SMOOTHING: float = 0.1
    MASK_RATIO: float = 0.30
    RECON_WARMUP: int = 10
    DISTILL_TEMP: float = 4.0
    NUM_LOSS_GROUPS: int = 4

    # ----- Graph -----
    GAT_HEADS: int = 4
    GAT_HIDDEN: int = 128
    GRAPH_EMBED_DIM: int = 256
    GRAPH_LOSS_WEIGHT: float = 0.5

    # ----- Hard negative mining -----
    MINE_INTERVAL: int = 10
    HARD_RATIO_START: float = 0.3
    HARD_RATIO_END: float = 0.8
    TOP_K_HARD: int = 10

    # ----- Weather (eval only) -----
    WEATHER_CONDITIONS: List[str] = field(default_factory=lambda: [
        "normal", "fog", "rain", "snow", "dark", "light",
        "fog_rain", "fog_snow", "rain_snow", "wind",
    ])


CFG = Config()


def load_config(path: str) -> Config:
    """Overwrite CFG fields from a YAML file. Returns the singleton."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    known_fields = {config_field.name for config_field in fields(Config)}
    for key in data:
        if key not in known_fields:
            raise KeyError(f"Unknown config key: {key}")
    defaults = Config()
    for config_field in fields(Config):
        setattr(CFG, config_field.name, getattr(defaults, config_field.name))
    for k, v in data.items():
        setattr(CFG, k, v)
    return CFG


def dump_config() -> dict:
    return asdict(CFG)
