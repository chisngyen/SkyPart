"""SUES-200 dataset.

Protocol (as in the paper):
- 120 train / 80 test locations, 4 altitudes (150/200/250/300m).
- Test gallery = ALL 200 satellite tiles (120 train + 80 test) -- the full
  confusion gallery. Distractor labels in the gallery are encoded as negative
  ints (-1000 - i) so they never match any drone query label.

Layout:
    SUES-200/
      drone-view/<loc4>/<alt>/*.png
      satellite-view/<loc4>/0.png

Returned per item:
    drone, satellite (transformed tensors)
    label (int, location idx within the split)
    altitude (int, m)
    alt_idx (int, 0..3)
    alt_norm (float, (alt-150)/150)
"""
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset

from ..config import CFG


class SUES200Dataset(Dataset):
    def __init__(self, root, mode: str = "train", altitudes=None,
                 drone_transform=None, sat_transform=None, transform=None):
        self.root = root
        self.mode = mode
        self.altitudes = altitudes or CFG.ALTITUDES
        self.drone_transform = drone_transform or transform
        self.sat_transform = sat_transform or transform
        self.drone_dir = os.path.join(root, CFG.DRONE_DIR)
        self.sat_dir = os.path.join(root, CFG.SAT_DIR)
        loc_ids = CFG.TRAIN_LOCS if mode == "train" else CFG.TEST_LOCS
        self.locations = [f"{l:04d}" for l in loc_ids]
        self.location_to_idx = {l: i for i, l in enumerate(self.locations)}
        self.samples = []
        self.drone_by_label = defaultdict(list)
        for loc in self.locations:
            li = self.location_to_idx[loc]
            sp = os.path.join(self.sat_dir, loc, "0.png")
            if not os.path.exists(sp):
                continue
            for alt in self.altitudes:
                ad = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(ad):
                    continue
                for img in sorted(os.listdir(ad)):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(ad, img), sp, li, alt))
                        self.drone_by_label[li].append(len(self.samples) - 1)
        self.num_classes = len(self.locations)
        print(f"  [{mode}] {len(self.samples)} samples, {self.num_classes} locations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, li, alt = self.samples[idx]
        try:
            drone = Image.open(dp).convert('RGB')
            sat = Image.open(sp).convert('RGB')
        except Exception:
            sz = CFG.IMG_SIZE
            drone = Image.new('RGB', (sz, sz), (128, 128, 128))
            sat = Image.new('RGB', (sz, sz), (128, 128, 128))
        if self.drone_transform:
            drone = self.drone_transform(drone)
        if self.sat_transform:
            sat = self.sat_transform(sat)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0
        return {'drone': drone, 'satellite': sat, 'label': li,
                'altitude': int(alt), 'alt_idx': alt_idx, 'alt_norm': alt_norm}
