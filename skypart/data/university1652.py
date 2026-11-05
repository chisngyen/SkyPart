"""University-1652 datasets.

Protocol (as in the paper, 4-folder asymmetric):
    train/{drone,satellite}/<bid>/*.jpg                 -- training pairs
    test/query_drone/<bid>/*.jpg                        -- D->S queries
    test/gallery_satellite/<bid>/*.jpg                  -- D->S gallery
    test/query_satellite/<bid>/*.jpg                    -- S->D queries
    test/gallery_drone/<bid>/*.jpg                      -- S->D gallery

No altitude metadata: training passes alt_idx=0 (the single bin); inference
passes alt_idx=None (mean-FiLM).
NUM_ALTITUDES is set to 1 in the YAML so the FiLM bank has a single bin that
the marginalized mean reduces to identity.
"""
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset

from ..config import CFG


class University1652TrainDataset(Dataset):
    def __init__(self, root: str, drone_transform=None, sat_transform=None):
        self.root = root
        self.drone_transform = drone_transform
        self.sat_transform = sat_transform
        drone_dir = os.path.join(root, "train", "drone")
        sat_dir = os.path.join(root, "train", "satellite")

        building_ids = sorted([
            d for d in os.listdir(drone_dir)
            if os.path.isdir(os.path.join(drone_dir, d))
            and os.path.isdir(os.path.join(sat_dir, d))
        ])
        self.bid_to_idx = {b: i for i, b in enumerate(building_ids)}
        self.num_classes = len(building_ids)
        self.samples = []
        self.drone_by_label = defaultdict(list)

        for bid in building_ids:
            idx = self.bid_to_idx[bid]
            dp = os.path.join(drone_dir, bid)
            sp = os.path.join(sat_dir, bid)
            sat_imgs = sorted([os.path.join(sp, f) for f in os.listdir(sp)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not sat_imgs:
                continue
            sat_img = sat_imgs[0]
            for f in sorted(os.listdir(dp)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(dp, f), sat_img, idx))
                    self.drone_by_label[idx].append(len(self.samples) - 1)
        print(f"  [train] {len(self.samples)} samples | {self.num_classes} buildings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, label = self.samples[idx]
        try:
            d = Image.open(dp).convert('RGB')
            s = Image.open(sp).convert('RGB')
        except Exception:
            sz = CFG.IMG_SIZE
            d = s = Image.new('RGB', (sz, sz), (128, 128, 128))
        if self.drone_transform:
            d = self.drone_transform(d)
        if self.sat_transform:
            s = self.sat_transform(s)
        # No altitude metadata -- dummies for batch homogeneity (single FiLM bin;
        # the altitude loss group is skipped when NUM_ALTITUDES == 1).
        return {'drone': d, 'satellite': s, 'label': label,
                'altitude': 0, 'alt_idx': 0, 'alt_norm': 0.0}


def _collect_uni1652_split(split_dir: str):
    samples, labels = [], []
    if not os.path.isdir(split_dir):
        return samples, labels
    for bid in sorted(os.listdir(split_dir)):
        bp = os.path.join(split_dir, bid)
        if not os.path.isdir(bp):
            continue
        for f in sorted(os.listdir(bp)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append(os.path.join(bp, f))
                labels.append(int(bid))
    return samples, labels


class University1652TestDataset:
    """Collects the four test folders. Embedding is delegated to the evaluator."""

    def __init__(self, root: str):
        test_root = os.path.join(root, "test")
        self.d2s_query_samples, self.d2s_query_labels = _collect_uni1652_split(
            os.path.join(test_root, "query_drone"))
        self.d2s_gallery_samples, self.d2s_gallery_labels = _collect_uni1652_split(
            os.path.join(test_root, "gallery_satellite"))
        self.s2d_query_samples, self.s2d_query_labels = _collect_uni1652_split(
            os.path.join(test_root, "query_satellite"))
        self.s2d_gallery_samples, self.s2d_gallery_labels = _collect_uni1652_split(
            os.path.join(test_root, "gallery_drone"))
        print(f"  [test] D2S {len(self.d2s_query_samples)}Q/{len(self.d2s_gallery_samples)}G"
              f" | S2D {len(self.s2d_query_samples)}Q/{len(self.s2d_gallery_samples)}G")
