"""Train loop, EMA wrapper, seed util."""
import copy
import math
import random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from ..config import CFG


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class EMAModel:
    def __init__(self, model, decay: float = 0.996):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


def cosine_lr(epoch: int, cfg=CFG) -> float:
    if epoch <= cfg.WARMUP_EPOCHS:
        scale = epoch / max(cfg.WARMUP_EPOCHS, 1)
    else:
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS, 1)
        scale = 0.5 * (1 + math.cos(math.pi * progress))
    return max(scale, 0.01)


def train_one_epoch(model, teacher, ema, loader, geo_loss, optimizer,
                    scaler, device, epoch: int, cfg=CFG):
    model.train()
    if teacher is not None:
        teacher.eval()
    total_sum = 0.0
    n = 0
    loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone = batch['drone'].to(device)
        sat = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        alt_norm = batch['alt_norm'].to(device).float()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=cfg.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)
            loss, ld = geo_loss(d_out, s_out, labels, model, teacher, ema,
                                drone, sat, alt_idx, alt_norm, epoch, device)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        total_sum += float(loss.item())
        n += 1
        for k, v in ld.items():
            loss_sums[k] += v

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}
