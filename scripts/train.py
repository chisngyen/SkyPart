"""Train SkyPart on SUES-200 / University-1652 / DenseUAV.

Dispatched by `cfg.DATASET` (set in YAML):
    DATASET: sues200          -> SUES200Dataset, evaluate_sues200
    DATASET: university1652   -> University1652TrainDataset/TestDataset, evaluate_uni1652
    DATASET: denseuav         -> DenseUAVDataset, evaluate_denseuav

Optional weather online aug: pass `--weather` to corrupt drone images per batch
with one of the 10 WeatherPrompt conditions (paper Sec.4.4 fine-tune protocol).

Usage:
    python scripts/train.py --config configs/sues200/default.yaml
    python scripts/train.py --config configs/sues200/weather.yaml --weather \
        --init-ckpt outputs/skypart_sues200_final.pth
"""
import argparse
import datetime
import json
import os
import random

import torch
from torch.utils.data import DataLoader

from skypart.checkpoints import build_checkpoint, load_checkpoint, load_model_state
from skypart.config import load_config, dump_config
from skypart.data import (
    SUES200Dataset, University1652TrainDataset, University1652TestDataset,
    DenseUAVDataset, HardNegativeMiner, HardPKSampler, PKSampler,
    get_drone_transforms, get_sat_transforms,
    build_weather_augmenters, apply_weather,
)
from skypart.engine import (
    EMAModel, set_seed, train_one_epoch, cosine_lr,
    evaluate_sues200, evaluate_uni1652, evaluate_denseuav,
    print_model_complexity,
)
from skypart.losses import GeoPartLoss
from skypart.models import SkyPartModel, DINOv2Teacher


# ---------------------------------------------------------------------------
# Online weather augmentation wrapper
# ---------------------------------------------------------------------------
class _WeatherDroneTransform:
    """Wraps a torchvision Compose so that, before tensorisation, a random
    weather corruption is applied to the drone PIL image (sat untouched).
    Used during paper Sec.4.4 weather fine-tune training."""

    def __init__(self, base_transform):
        self.base = base_transform
        self.augs = build_weather_augmenters()

    def __call__(self, pil_img):
        aug = random.choice(self.augs)
        if aug is not None:
            pil_img = apply_weather(pil_img, aug)
        return self.base(pil_img)


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------
def _build_sues200(cfg, drone_tf, sat_tf):
    train_ds = SUES200Dataset(cfg.SUES_ROOT, 'train',
                               drone_transform=drone_tf, sat_transform=sat_tf)
    test_ds = SUES200Dataset(cfg.SUES_ROOT, 'test',
                              transform=get_drone_transforms("test"))
    return train_ds, test_ds, evaluate_sues200


def _build_uni1652(cfg, drone_tf, sat_tf):
    root = cfg.DATA_ROOT or cfg.SUES_ROOT
    train_ds = University1652TrainDataset(root, drone_transform=drone_tf,
                                           sat_transform=sat_tf)
    test_ds = University1652TestDataset(root)
    return train_ds, test_ds, evaluate_uni1652


def _build_denseuav(cfg, drone_tf, sat_tf):
    root = cfg.DENSEUAV_ROOT or cfg.DATA_ROOT
    train_ds = DenseUAVDataset(root, mode='train',
                                drone_transform=drone_tf, sat_transform=sat_tf)
    test_ds = DenseUAVDataset(root, mode='test',
                               transform=get_drone_transforms("test"))
    return train_ds, test_ds, evaluate_denseuav


_BUILDERS = {
    'sues200': _build_sues200,
    'university1652': _build_uni1652,
    'denseuav': _build_denseuav,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="Override CFG.SEED (paper used 42, 123, 2024).")
    ap.add_argument("--weather", action='store_true',
                    help="Online weather aug on drone (paper Sec.4.4 fine-tune).")
    ap.add_argument("--init-ckpt", default=None,
                    help="Clean checkpoint to fine-tune from (weather runs).")
    ap.add_argument("--allow-legacy-checkpoint", action='store_true',
                    help="Accept a metadata-free --init-ckpt.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.SEED = args.seed
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.DATASET not in _BUILDERS:
        raise ValueError(f"Unknown cfg.DATASET={cfg.DATASET!r}; "
                          f"expected one of {list(_BUILDERS)}")
    print("=" * 72)
    print(f"  SkyPart -- {cfg.DATASET} ({cfg.NUM_EPOCHS} ep, seed {cfg.SEED}, {device}"
          f"{', weather-online' if args.weather else ''})")
    print("=" * 72)

    # Rotation is DenseUAV-only: its satellite crops have arbitrary basemap
    # orientation; SUES-200 and University-1652 pairs are canonically oriented.
    rotate = cfg.DATASET == 'denseuav'
    drone_tf = get_drone_transforms("train", rotate=rotate)
    sat_tf = get_sat_transforms("train", rotate=rotate)
    if args.weather:
        drone_tf = _WeatherDroneTransform(drone_tf)

    train_ds, test_ds, eval_fn = _BUILDERS[cfg.DATASET](cfg, drone_tf, sat_tf)
    cfg.NUM_CLASSES = train_ds.num_classes

    model = SkyPartModel(cfg.NUM_CLASSES, cfg=cfg).to(device)
    if args.init_ckpt:
        init = load_checkpoint(args.init_ckpt, cfg,
                               allow_legacy_checkpoint=args.allow_legacy_checkpoint)
        if init.num_classes != cfg.NUM_CLASSES:
            raise SystemExit(f"--init-ckpt has {init.num_classes} classes but this "
                             f"dataset has {cfg.NUM_CLASSES}; use a checkpoint "
                             f"trained on {cfg.DATASET}.")
        load_model_state(model, init)
        print(f"  Initialised from {args.init_ckpt}")
    teacher = None
    try:
        teacher = DINOv2Teacher(
            repo=cfg.DINOV2_REPO,
            model=cfg.DINOV2_TEACHER_MODEL,
        ).to(device); teacher.eval()
    except Exception as e:
        print(f"  [WARN] teacher not loaded: {e}")
    ema = EMAModel(model, decay=cfg.EMA_DECAY)

    # Sampler: SUES-200 + DenseUAV use HardPKSampler; Uni-1652 uses plain PKSampler.
    if cfg.DATASET in ('sues200', 'denseuav'):
        miner = HardNegativeMiner(train_ds, model, device, top_k=cfg.TOP_K_HARD)
        sampler = HardPKSampler(train_ds, cfg.P_CLASSES, cfg.K_SAMPLES,
                                miner=miner, hard_ratio=cfg.HARD_RATIO_START)
    else:
        miner = None
        sampler = PKSampler(train_ds, cfg.P_CLASSES, cfg.K_SAMPLES)

    loader = DataLoader(train_ds, batch_sampler=sampler,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True,
                        persistent_workers=cfg.NUM_WORKERS > 0)
    stats = print_model_complexity(model, device)
    geo_loss = GeoPartLoss(cfg.NUM_CLASSES, cfg.EMBED_DIM,
                            num_groups=cfg.NUM_LOSS_GROUPS, cfg=cfg).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.BACKBONE_LR},
        {'params': head_params, 'lr': cfg.LR},
        {'params': geo_loss.parameters(), 'lr': cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_AMP)

    history = []
    train_start = datetime.datetime.now().isoformat()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        scale = cosine_lr(epoch, cfg)
        for g in optimizer.param_groups:
            base = cfg.BACKBONE_LR if g is optimizer.param_groups[0] else cfg.LR
            g['lr'] = base * scale

        if miner is not None and epoch % cfg.MINE_INTERVAL == 0 and epoch > cfg.WARMUP_EPOCHS:
            progress = (epoch - cfg.WARMUP_EPOCHS) / max(cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS, 1)
            ratio = cfg.HARD_RATIO_START + progress * (cfg.HARD_RATIO_END - cfg.HARD_RATIO_START)
            miner.mine()
            sampler = HardPKSampler(train_ds, cfg.P_CLASSES, cfg.K_SAMPLES,
                                    miner=miner, hard_ratio=ratio)
            loader = DataLoader(train_ds, batch_sampler=sampler,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True,
                                persistent_workers=cfg.NUM_WORKERS > 0)

        avg_loss, ld = train_one_epoch(model, teacher, ema, loader, geo_loss,
                                        optimizer, scaler, device, epoch, cfg)
        print(f"Ep {epoch:3d}/{cfg.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"Align {ld.get('G_align',0):.3f}  Dist {ld.get('G_distill',0):.3f}  "
              f"Part {ld.get('G_part',0):.3f}  Alt {ld.get('G_alt',0):.3f}")

        # Periodic evaluation is logged only; it never selects the checkpoint.
        if epoch % cfg.EVAL_INTERVAL == 0 or epoch == cfg.NUM_EPOCHS:
            metrics = eval_fn(model, test_ds, device, cfg)
            history.append({'epoch': epoch, **metrics})
            print(f"  D->S R@1 {metrics['D2S']['R@1']:.2f}%")

    # The reported model is the final-epoch checkpoint (no test-set selection).
    final_metrics = history[-1]
    final_r1 = final_metrics['D2S']['R@1']
    ckpt_name = f'skypart_{cfg.DATASET}{"_weather" if args.weather else ""}_final.pth'
    torch.save(
        build_checkpoint(
            model_state_dict=model.state_dict(),
            epoch=cfg.NUM_EPOCHS,
            metrics={k: v for k, v in final_metrics.items() if k != 'epoch'},
            config=dump_config(),
        ),
        os.path.join(cfg.OUTPUT_DIR, ckpt_name),
    )

    summary = {
        'experiment': os.path.basename(args.config),
        'dataset': cfg.DATASET,
        'weather_finetune': args.weather,
        'timestamp': {'start': train_start, 'end': datetime.datetime.now().isoformat()},
        'model_complexity': stats,
        'config': dump_config(),
        'final_epoch': cfg.NUM_EPOCHS,
        'final_d2s_r1': final_r1,
        'eval_history': history,
    }
    out = os.path.join(cfg.OUTPUT_DIR,
                        f'skypart_{cfg.DATASET}{"_weather" if args.weather else ""}_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Done. Final-epoch D->S R@1 {final_r1:.2f}% ({ckpt_name}). Summary -> {out}")


if __name__ == "__main__":
    main()
