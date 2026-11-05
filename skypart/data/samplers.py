"""PK / hard-PK samplers + hard-negative miner.

PK sampler: P classes x K samples per batch. Hard-PK augments each batch with
top-K hard negatives mined from current embeddings every MINE_INTERVAL epochs.
"""
import random

import torch
import torch.nn.functional as F
from PIL import Image

from ..config import CFG
from .transforms import get_drone_transforms


class HardNegativeMiner:
    def __init__(self, dataset, model, device, top_k: int = 10):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.top_k = top_k
        self.hard_negatives = {}

    @torch.no_grad()
    def mine(self):
        self.model.eval()
        tf = get_drone_transforms("test")
        all_embs, all_labels = [], []
        for idx in range(len(self.dataset)):
            drone_path, _, label, height_name = self.dataset.samples[idx]
            try:
                img = tf(Image.open(drone_path).convert('RGB')).unsqueeze(0).to(self.device)
            except Exception:
                continue
            alt_idx_t = torch.tensor([CFG.ALT_TO_IDX.get(height_name, 1)], device=self.device)
            emb = self.model.extract_embedding(img, alt_idx=alt_idx_t).cpu()
            all_embs.append(emb)
            all_labels.append(label)
        if not all_embs:
            return
        all_embs = torch.cat(all_embs)
        all_labels = torch.tensor(all_labels)
        self.hard_negatives = {}
        for lbl in all_labels.unique():
            lbl = lbl.item()
            pos_mask = (all_labels == lbl)
            neg_mask = ~pos_mask
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            class_emb = F.normalize(all_embs[pos_mask].mean(0, keepdim=True), dim=-1)
            neg_embs = F.normalize(all_embs[neg_mask], dim=-1)
            sim = (class_emb @ neg_embs.T).squeeze(0)
            k = min(self.top_k, sim.size(0))
            _, topk_idx = sim.topk(k)
            neg_indices = torch.where(neg_mask)[0]
            self.hard_negatives[lbl] = neg_indices[topk_idx].tolist()
        print(f"  [HardMiner] Mined for {len(self.hard_negatives)} classes (top-{self.top_k})")
        self.model.train()


class PKSampler:
    def __init__(self, ds, p: int, k: int):
        self.ds = ds
        self.p = p
        self.k = k
        self.labels = list(ds.drone_by_label.keys())

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)
        batch = []
        for l in labels:
            idx = self.ds.drone_by_label[l]
            if len(idx) < self.k:
                pool = (idx * (self.k // len(idx) + 1))[:self.k]
            else:
                pool = random.sample(idx, self.k)
            batch.extend(pool)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]

    def __len__(self):
        return len(self.labels) // self.p


class HardPKSampler:
    def __init__(self, ds, p: int, k: int, miner=None, hard_ratio: float = 0.3):
        self.ds = ds
        self.p = p
        self.k = k
        self.labels = list(ds.drone_by_label.keys())
        self.miner = miner
        self.hard_ratio = hard_ratio

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)
        batch = []
        has_hard = (self.miner is not None and len(self.miner.hard_negatives) > 0)
        for l in labels:
            idx = self.ds.drone_by_label[l]
            if len(idx) < self.k:
                pool = (idx * (self.k // len(idx) + 1))[:self.k]
            else:
                pool = random.sample(idx, self.k)
            batch.extend(pool)
            if has_hard and l in self.miner.hard_negatives and random.random() < self.hard_ratio:
                hard_indices = self.miner.hard_negatives[l]
                batch.extend(random.sample(hard_indices, min(self.k, len(hard_indices))))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]

    def __len__(self):
        return len(self.labels) // self.p
