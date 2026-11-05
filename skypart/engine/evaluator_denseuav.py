"""DenseUAV evaluator (no re-ranking, no TTA).

Single-pass cosine retrieval on the official full gallery (all locations,
H80/H90/H100, `_old` variants included). Reports R@1/R@5/R@10/R@top1/AP/SDM@{1,3,5} for BOTH
D->S and S->D. SDM uses Dense_GPS_ALL.txt (shipped with the dataset). If
the GPS file is missing, SDM@k entries are None.
"""
import os
import time

import numpy as np
import torch
from PIL import Image

from ..config import CFG
from ..data.denseuav import IMG_EXTS
from ..data.transforms import get_drone_transforms


def _folder_id(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2] if len(parts) >= 2 else ""


def load_dense_gps(dataset_root: str):
    """Parse Dense_GPS_ALL.txt -> {folder_id: [lon, lat]}. Returns None if missing."""
    p = os.path.join(dataset_root, CFG.DENSE_GPS_FILE)
    if not os.path.isfile(p):
        return None
    out = {}
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            key = parts[0].replace('\\', '/').split('/')[-2]
            try:
                t1, t2 = parts[1], parts[2]
                lon = float(t1.split('E')[-1]) if 'E' in t1.upper() else float(t1)
                lat = float(t2.split('N')[-1]) if 'N' in t2.upper() else float(t2)
                out[key] = [lon, lat]
            except (ValueError, IndexError):
                continue
    return out or None


def _sdm_score(distance, K: int) -> float:
    d = np.asarray(distance[:K], dtype=np.float64)
    w = np.ones(K) - np.arange(K) / K
    return float(np.sum(w * (1.0 / np.exp(d * 5000.0))) / (w.sum() + 1e-12))


@torch.no_grad()
def _embed_paths(model, paths, alt_idxs, device, cfg, weather_aug=None,
                  use_alt: bool = False):
    """Embed images. Evaluation always uses mean-FiLM (alt_idx=None); `alt_idxs`
    are passed only when `use_alt=True`."""
    from ..data.weather import apply_weather
    tf = get_drone_transforms("test")
    feats = []
    bs = cfg.EVAL_BATCH_SIZE
    for i in range(0, len(paths), bs):
        chunk_paths = paths[i:i + bs]
        chunk_alt = alt_idxs[i:i + bs]
        imgs = []
        for p in chunk_paths:
            im = Image.open(p).convert('RGB')
            if weather_aug is not None:
                im = apply_weather(im, weather_aug)
            imgs.append(tf(im))
        batch = torch.stack(imgs).to(device)
        ai = (torch.tensor(chunk_alt, device=device) if use_alt else None)
        with torch.amp.autocast('cuda', enabled=cfg.USE_AMP):
            feats.append(model.extract_embedding(batch, alt_idx=ai).float().cpu())
    return torch.cat(feats) if feats else torch.empty(0)


def _retrieval(query_feats, query_labels, gal_feats, gal_labels,
               query_paths=None, gal_paths=None, gps=None):
    sim = query_feats @ gal_feats.T
    _, rank = sim.sort(1, descending=True)
    N, G = query_feats.size(0), gal_feats.size(0)
    top1_pct = max(1, int(G * 0.01))
    r1 = r5 = r10 = r_top1pct = 0
    ap_sum = 0.0
    sdm_sum = {1: 0.0, 3: 0.0, 5: 0.0}
    sdm_cnt = {1: 0, 3: 0, 5: 0}
    max_K = 5
    for i in range(N):
        m = torch.where(gal_labels[rank[i]] == query_labels[i])[0]
        if len(m) == 0:
            continue
        f = m[0].item()
        r1 += int(f < 1)
        r5 += int(f < 5)
        r10 += int(f < 10)
        r_top1pct += int(f < top1_pct)
        ap_sum += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
        if gps is not None and query_paths is not None and gal_paths is not None and rank.size(1) >= max_K:
            qkey = _folder_id(query_paths[i])
            qpos = gps.get(qkey)
            if qpos is None:
                continue
            dists, ok = [], True
            for kk in range(max_K):
                gi = int(rank[i, kk].item())
                gpos = gps.get(_folder_id(gal_paths[gi]))
                if gpos is None:
                    ok = False
                    break
                dists.append(float(np.linalg.norm(np.array(gpos) - np.array(qpos))))
            if ok:
                d = np.array(dists)
                for K in (1, 3, 5):
                    sdm_sum[K] += _sdm_score(d, K)
                    sdm_cnt[K] += 1
    out = {'R@1': r1 / N * 100, 'R@5': r5 / N * 100, 'R@10': r10 / N * 100,
           'R@top1': r_top1pct / N * 100, 'AP': ap_sum / N * 100, 'n': N}
    for K in (1, 3, 5):
        out[f'SDM@{K}'] = (sdm_sum[K] / sdm_cnt[K] * 100) if sdm_cnt[K] > 0 else None
    return out


def _list_images(folder: str):
    """Every image in `folder`, legacy `_old` variants included (official gallery)."""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXTS)]


def _collect(root: str, name_to_id: dict):
    """All images under root/<location>/, labelled by location folder name."""
    paths, labels = [], []
    if not os.path.isdir(root):
        return paths, labels
    for name in sorted(os.listdir(root)):
        imgs = _list_images(os.path.join(root, name))
        if not imgs:
            continue
        lid = name_to_id.setdefault(name, len(name_to_id))
        paths += imgs
        labels += [lid] * len(imgs)
    return paths, labels


@torch.no_grad()
def evaluate(model, test_ds, device, cfg=CFG, weather_drone_aug=None):
    """Official DenseUAV protocol. Single pass, no re-ranking, no TTA.

    D->S: test/query_drone queries vs the full test/gallery_satellite
          (all locations, all heights; 18,198 images).
    S->D: test/query_satellite queries vs the full test/gallery_drone.
    Every embedding uses mean-FiLM: no altitude metadata at inference.

    Note on SDM: place `Dense_GPS_ALL.txt` (shipped with DenseUAV) under
    `cfg.DENSEUAV_ROOT`; otherwise SDM@k entries will be None.
    """
    model.eval()
    t0 = time.time()
    root = cfg.DENSEUAV_ROOT or test_ds.root
    gps = load_dense_gps(root)
    if gps is None:
        print(f"  [GPS] {cfg.DENSE_GPS_FILE} not found -- SDM@k will be None.")

    name_to_id = {}
    query_paths = [test_ds.samples[i][0] for i in range(len(test_ds))]
    query_labels = [name_to_id.setdefault(os.path.basename(os.path.dirname(p)), len(name_to_id))
                    for p in query_paths]
    sat_paths, sat_labels = _collect(os.path.join(root, "test", "gallery_satellite"), name_to_id)
    qsat_paths, qsat_labels = _collect(os.path.join(root, "test", "query_satellite"), name_to_id)
    gdr_paths, gdr_labels = _collect(os.path.join(root, "test", "gallery_drone"), name_to_id)

    def emb(paths, aug=None):
        return _embed_paths(model, paths, [0] * len(paths), device, cfg,
                            weather_aug=aug, use_alt=False)

    qf = emb(query_paths, weather_drone_aug)
    sf = emb(sat_paths)
    print(f"  D->S: {len(qf)} drone queries vs {len(sf)} satellite gallery")
    d2s = _retrieval(qf, torch.tensor(query_labels), sf, torch.tensor(sat_labels),
                     query_paths=query_paths, gal_paths=sat_paths, gps=gps)

    if qsat_paths and gdr_paths:
        qsf = emb(qsat_paths)
        gdf = emb(gdr_paths, weather_drone_aug)  # drone side corrupted in both directions
        print(f"  S->D: {len(qsf)} satellite queries vs {len(gdf)} drone gallery")
        s2d = _retrieval(qsf, torch.tensor(qsat_labels), gdf, torch.tensor(gdr_labels),
                         query_paths=qsat_paths, gal_paths=gdr_paths, gps=gps)
    else:
        print("  [S->D] test/query_satellite or test/gallery_drone missing -- skipped.")
        s2d = {k: float('nan') for k in ('R@1', 'R@5', 'R@10', 'R@top1', 'AP')}
        s2d.update({'n': 0, 'SDM@1': None, 'SDM@3': None, 'SDM@5': None})

    t = time.time() - t0
    print(f"\n  DenseUAV (no re-ranking, no TTA)  --  {t:.1f}s")
    for tag, m in [('D2S', d2s), ('S2D', s2d)]:
        sdm = "  ".join([f"SDM@{K} {m[f'SDM@{K}']:.2f}" if m[f'SDM@{K}'] is not None
                          else f"SDM@{K} n/a" for K in (1, 3, 5)])
        print(f"  {tag} R@1 {m['R@1']:6.2f}  R@5 {m['R@5']:6.2f}  R@10 {m['R@10']:6.2f}"
              f"  R@top1 {m['R@top1']:6.2f}  AP {m['AP']:6.2f}  {sdm}")
    return {'D2S': d2s, 'S2D': s2d, 'inference_time': t}
