"""Evaluate a SkyPart checkpoint (clean retrieval).

Dispatched by `cfg.DATASET`. Reports D->S and S->D R@1/R@5/R@10/AP, plus
per-altitude breakdown for SUES-200, and SDM@{1,3,5} for DenseUAV (when
Dense_GPS_ALL.txt is present).

Usage:
    python scripts/eval.py --config configs/sues200/default.yaml --ckpt <pth>
    python scripts/eval.py --config configs/university1652/default.yaml --ckpt <pth>
    python scripts/eval.py --config configs/denseuav/default.yaml      --ckpt <pth>
"""
import argparse
import json
import os

import torch

from skypart.checkpoints import load_checkpoint, load_model_state
from skypart.config import load_config
from skypart.data import (
    SUES200Dataset, University1652TestDataset, DenseUAVDataset,
    get_drone_transforms,
)
from skypart.engine import (
    evaluate_sues200, evaluate_uni1652, evaluate_denseuav,
    print_model_complexity,
)
from skypart.models import SkyPartModel


def _build_test_ds(cfg):
    if cfg.DATASET == 'sues200':
        return (SUES200Dataset(cfg.SUES_ROOT, 'test',
                                transform=get_drone_transforms("test")),
                evaluate_sues200)
    if cfg.DATASET == 'university1652':
        return (University1652TestDataset(cfg.DATA_ROOT or cfg.SUES_ROOT),
                evaluate_uni1652)
    if cfg.DATASET == 'denseuav':
        return (DenseUAVDataset(cfg.DENSEUAV_ROOT or cfg.DATA_ROOT, mode='test',
                                 transform=get_drone_transforms("test")),
                evaluate_denseuav)
    raise ValueError(f"Unknown cfg.DATASET={cfg.DATASET!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Allow metadata-free checkpoints with YAML-owned configuration.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(
        args.ckpt,
        cfg,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    test_ds, eval_fn = _build_test_ds(cfg)

    model = SkyPartModel(checkpoint.num_classes, cfg=cfg).to(device)
    load_model_state(model, checkpoint)
    model.eval()

    print_model_complexity(model, device)
    metrics = eval_fn(model, test_ds, device, cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(cfg.OUTPUT_DIR, f'skypart_{cfg.DATASET}_eval.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Eval JSON -> {out}")


if __name__ == "__main__":
    main()
