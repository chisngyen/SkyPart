"""SkyPart smoke test (no dataset required).

Runs the full forward+backward path on a synthetic batch for SUES-200 config
and verifies everything imports + computes without crashing. Useful before
downloading the datasets.

Usage:
    python scripts/smoke_test.py

Pass criteria: prints "SMOKE TEST OK" with no exception.
"""
import os
import sys
import traceback

# Make `skypart` importable without `pip install -e .`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from skypart.config import load_config, CFG
from skypart.models import SkyPartModel, DINOv2Teacher
from skypart.losses import GeoPartLoss
from skypart.engine import EMAModel, set_seed


def _synthetic_batch(B, sz, num_classes, num_alts, device):
    return {
        'drone':     torch.randn(B, 3, sz, sz, device=device),
        'satellite': torch.randn(B, 3, sz, sz, device=device),
        'label':     torch.randint(0, num_classes, (B,), device=device),
        'alt_idx':   torch.randint(0, num_alts, (B,), device=device),
        'alt_norm':  torch.rand(B, device=device),
    }


def main():
    print("=" * 60)
    print("  SkyPart smoke test (synthetic data, no dataset needed)")
    print("=" * 60)

    cfg = load_config(os.path.join(_REPO_ROOT, "configs", "sues200", "default.yaml"))
    cfg.NUM_CLASSES = 8     # tiny for smoke
    cfg.NUM_EPOCHS = 1
    cfg.RECON_WARMUP = 0    # exercise MAR path
    cfg.IMG_SIZE = 224      # smaller, faster
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}, image size: {cfg.IMG_SIZE}, K_max: {cfg.N_PARTS}")

    print("\n[1/5] Building SkyPartModel ...")
    model = SkyPartModel(cfg.NUM_CLASSES, cfg=cfg).to(device)

    print("[2/5] Loading DINOv2 teacher ...")
    teacher = None
    try:
        teacher = DINOv2Teacher(
            repo=cfg.DINOV2_REPO,
            model=cfg.DINOV2_TEACHER_MODEL,
        ).to(device); teacher.eval()
    except Exception as e:
        print(f"  [WARN] teacher not loaded ({e}); continuing without distillation.")

    print("[3/5] Building EMAModel + GeoPartLoss ...")
    ema = EMAModel(model, decay=cfg.EMA_DECAY)
    geo = GeoPartLoss(cfg.NUM_CLASSES, cfg.EMBED_DIM,
                       num_groups=cfg.NUM_LOSS_GROUPS, cfg=cfg).to(device)

    print("[4/5] Forward + backward on synthetic batch ...")
    batch = _synthetic_batch(B=4, sz=cfg.IMG_SIZE,
                              num_classes=cfg.NUM_CLASSES,
                              num_alts=cfg.NUM_ALTITUDES, device=device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(geo.parameters()), lr=1e-4)
    model.train()
    optimizer.zero_grad()
    d_out = model(batch['drone'], alt_idx=batch['alt_idx'], return_parts=True)
    s_out = model(batch['satellite'], alt_idx=None, return_parts=True)
    loss, ld = geo(d_out, s_out, batch['label'], model, teacher, ema,
                    batch['drone'], batch['satellite'],
                    batch['alt_idx'], batch['alt_norm'].float(),
                    epoch=1, device=device)
    loss.backward()
    optimizer.step()
    ema.update(model)
    print(f"  loss={loss.item():.4f}  groups: "
          f"align={ld.get('G_align',0):.3f} dist={ld.get('G_distill',0):.3f} "
          f"part={ld.get('G_part',0):.3f} alt={ld.get('G_alt',0):.3f}")

    print("[5/5] Eval extract_embedding (single-pass, no TTA) ...")
    model.eval()
    with torch.no_grad():
        emb = model.extract_embedding(batch['drone'][:2], alt_idx=None)  # mean-FiLM
    assert emb.shape == (2, cfg.EMBED_DIM), f"expected ({2}, {cfg.EMBED_DIM}), got {emb.shape}"
    print(f"  embedding shape: {tuple(emb.shape)}  norm: {emb.norm(dim=-1).mean():.4f}")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n!!! SMOKE TEST FAILED !!!")
        traceback.print_exc()
        sys.exit(1)
