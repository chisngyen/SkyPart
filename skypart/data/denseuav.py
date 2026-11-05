"""DenseUAV dataset.

Protocol (as in the paper):
- Train: 2256 locations, 3 heights H80/H90/H100, drone .JPG / satellite .tif.
- Test:  777 query drone folders; the evaluator uses the official full
  gallery (engine/evaluator_denseuav.py).
- Training skips files containing '_old' (legacy variant).
- Train-time DiscreteRotation{0,90,180,270} on BOTH drone and satellite
  (DenseUAV only: basemap orientation is arbitrary; SUES-200 / Uni-1652 do
  NOT use rotation).

Returned per item:
    drone, satellite (transformed tensors)
    label (int, drone-folder index)
    altitude (int, m -- 80/90/100)
    alt_idx (int, 0..2 from H80/H90/H100)
    alt_norm (float, (h-80)/20)
"""
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset

from ..config import CFG

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif')


def _find_image(folder: str, prefix: str, exts=IMG_EXTS):
    """Find first non-_old file under `folder` matching `prefix`+ext (case-insensitive)."""
    for ext in exts:
        for actual_ext in [ext, ext.upper()]:
            path = os.path.join(folder, prefix + actual_ext)
            if os.path.isfile(path):
                return path
    for f in sorted(os.listdir(folder)):
        if '_old' in f.lower():
            continue
        if f.lower().startswith(prefix.lower()) and f.lower().endswith(exts):
            return os.path.join(folder, f)
    return None


class DenseUAVDataset(Dataset):
    def __init__(self, root: str, mode: str = "train", transform=None,
                 drone_transform=None, sat_transform=None):
        self.root = root
        self.mode = mode
        self.drone_transform = drone_transform or transform
        self.sat_transform = sat_transform or transform
        if mode == "train":
            drone_root = os.path.join(root, "train", "drone")
            sat_root = os.path.join(root, "train", "satellite")
        else:
            drone_root = os.path.join(root, "test", "query_drone")
            sat_root = os.path.join(root, "test", "gallery_satellite")
        self.drone_root = drone_root
        self.sat_root = sat_root

        drone_folders = sorted([d for d in os.listdir(drone_root)
                                if os.path.isdir(os.path.join(drone_root, d))])
        sat_folders_set = set(d for d in os.listdir(sat_root)
                              if os.path.isdir(os.path.join(sat_root, d)))

        self.samples = []
        self.drone_by_label = defaultdict(list)
        self.sat_paths = {}

        if mode == "train":
            label = 0
            for folder_name in drone_folders:
                if folder_name not in sat_folders_set:
                    continue
                drone_dir = os.path.join(drone_root, folder_name)
                sat_dir = os.path.join(sat_root, folder_name)
                has_any = False
                for h in CFG.HEIGHTS:
                    dp = _find_image(drone_dir, h)
                    sp = _find_image(sat_dir, h)
                    if dp and sp:
                        self.samples.append((dp, sp, label, h))
                        self.drone_by_label[label].append(len(self.samples) - 1)
                        has_any = True
                if has_any:
                    rep = (_find_image(sat_dir, 'H100')
                           or _find_image(sat_dir, CFG.HEIGHTS[0]))
                    self.sat_paths[label] = rep
                    label += 1
            self.num_classes = label
        else:
            label = 0
            for folder_name in drone_folders:
                drone_dir = os.path.join(drone_root, folder_name)
                has_any = False
                for h in CFG.HEIGHTS:
                    dp = _find_image(drone_dir, h)
                    if dp:
                        sat_dir = os.path.join(sat_root, folder_name)
                        sp = _find_image(sat_dir, h) if os.path.isdir(sat_dir) else None
                        self.samples.append((dp, sp or "", label, h))
                        self.drone_by_label[label].append(len(self.samples) - 1)
                        has_any = True
                if has_any:
                    sat_dir = os.path.join(sat_root, folder_name)
                    if os.path.isdir(sat_dir):
                        rep = _find_image(sat_dir, 'H100')
                        if rep:
                            self.sat_paths[label] = rep
                    label += 1
            self.gallery_sat_folders = sorted([d for d in os.listdir(sat_root)
                                                if os.path.isdir(os.path.join(sat_root, d))])
            self.drone_folder_to_label = {name: i for i, name in enumerate(drone_folders)}
            self.num_classes = len(drone_folders)

        gallery_info = (f', gallery: {len(self.gallery_sat_folders)} sat folders'
                        if mode != 'train' else '')
        print(f"  [{mode}] {len(self.samples)} samples, {self.num_classes} locations{gallery_info}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, label, height_name = self.samples[idx]
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
        alt_idx = CFG.ALT_TO_IDX.get(height_name, 1)
        h = int(height_name.replace('H', '')) if height_name.startswith('H') else 90
        alt_norm = (h - 80) / 20.0
        return {'drone': drone, 'satellite': sat, 'label': label,
                'altitude': h, 'alt_idx': alt_idx, 'alt_norm': alt_norm}
