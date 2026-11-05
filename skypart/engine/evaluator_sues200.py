"""SUES-200 evaluator (full 200-tile confusion gallery).

Path-based embedding to support optional `weather_drone_aug` (imgaug Sequential)
applied to drone images on BOTH sides (D->S queries and S->D gallery).
Satellite stays clean. Single-pass, no re-ranking, no TTA.
"""
import os
import time

import torch
from PIL import Image

from ..config import CFG
from ..data.transforms import get_drone_transforms


@torch.no_grad()
def _embed_paths(model, paths, alt_idxs, device, cfg=CFG, weather_aug=None,
                 use_alt: bool = False):
    """Embed a list of image paths.

    paths: list[str]
    alt_idxs: list[int] same length, used only if `use_alt`
    weather_aug: optional imgaug Sequential applied to PIL before transforms
    use_alt: pass alt_idx to model.extract_embedding. Evaluation always uses
             use_alt=False (mean-FiLM): no altitude metadata at inference.
    """
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


def _retrieval(qf, ql, gf, gl):
    sim = qf @ gf.T
    _, rank = sim.sort(1, descending=True)
    N = qf.size(0)
    r1 = r5 = r10 = 0
    ap_sum = 0.0
    for i in range(N):
        m = torch.where(gl[rank[i]] == ql[i])[0]
        if len(m) == 0:
            continue
        f = m[0].item()
        r1 += int(f < 1)
        r5 += int(f < 5)
        r10 += int(f < 10)
        ap_sum += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
    return {'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
            'R@10': r10 / N * 100, 'AP': ap_sum / N * 100, 'n': N}


@torch.no_grad()
def evaluate(model, test_ds, device, cfg=CFG, weather_drone_aug=None):
    """SUES-200 evaluation: both directions, per-altitude D->S, full gallery.

    weather_drone_aug: imgaug Sequential applied to drone images only
        (sat stays clean). When None, runs the clean protocol (Tab. 1 of paper).
    """
    model.eval()
    t0 = time.time()
    root = test_ds.root

    # ------------------------------------------------------------------
    # 1) Test-split drone queries (for D->S).
    # ------------------------------------------------------------------
    drone_paths = [s[0] for s in test_ds.samples]
    drone_labels = torch.tensor([s[2] for s in test_ds.samples])
    drone_alts = [int(s[3]) for s in test_ds.samples]
    drone_alt_idxs = [cfg.ALT_TO_IDX.get(s[3], 0) for s in test_ds.samples]
    qf = _embed_paths(model, drone_paths, drone_alt_idxs, device, cfg,
                       weather_aug=weather_drone_aug, use_alt=False)

    # ------------------------------------------------------------------
    # 2) Full satellite gallery: ALL 200 locations (full confusion).
    #    Distractor labels encoded as -1000 - i (never match any query label).
    # ------------------------------------------------------------------
    sat_paths, sat_labels = [], []
    distractor_cnt = 0
    for l in range(1, 201):
        loc = f"{l:04d}"
        sp = os.path.join(root, cfg.SAT_DIR, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_paths.append(sp)
        if loc in test_ds.location_to_idx:
            sat_labels.append(test_ds.location_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt)
            distractor_cnt += 1
    gf = _embed_paths(model, sat_paths, [0] * len(sat_paths), device, cfg,
                      weather_aug=None, use_alt=False)
    gl = torch.tensor(sat_labels)

    print(f"  Gallery {len(gf)} sat | D->S queries {len(qf)}")

    # ------------------------------------------------------------------
    # 3) D -> S overall + per-altitude
    # ------------------------------------------------------------------
    d2s_overall = _retrieval(qf, drone_labels, gf, gl)
    drone_alts_t = torch.tensor(drone_alts)
    d2s_per_alt = {}
    for alt in sorted(set(drone_alts)):
        mask = drone_alts_t == alt
        if mask.sum() == 0:
            continue
        d2s_per_alt[int(alt)] = _retrieval(qf[mask], drone_labels[mask], gf, gl)

    # ------------------------------------------------------------------
    # 4) S -> D: test-split sat queries vs FULL drone gallery (all 200 locs).
    #    Drone gallery rebuilt with weather_drone_aug applied where given.
    # ------------------------------------------------------------------
    drone_gal_paths, drone_gal_labels, drone_gal_alts = [], [], []
    distractor_cnt = 0
    for l in range(1, 201):
        loc = f"{l:04d}"
        ad_root = os.path.join(root, cfg.DRONE_DIR, loc)
        if not os.path.isdir(ad_root):
            continue
        is_distractor = loc not in test_ds.location_to_idx
        loc_label = (test_ds.location_to_idx[loc] if not is_distractor
                     else -1000 - distractor_cnt)
        for alt in cfg.ALTITUDES:
            ad = os.path.join(ad_root, alt)
            if not os.path.isdir(ad):
                continue
            for img in sorted(os.listdir(ad)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    drone_gal_paths.append(os.path.join(ad, img))
                    drone_gal_labels.append(loc_label)
                    drone_gal_alts.append(cfg.ALT_TO_IDX.get(alt, 0))
        if is_distractor:
            distractor_cnt += 1
    drone_gal_feats = _embed_paths(model, drone_gal_paths, drone_gal_alts,
                                    device, cfg, weather_aug=weather_drone_aug,
                                    use_alt=False)
    drone_gal_labels_t = torch.tensor(drone_gal_labels)

    test_sat_mask = torch.tensor([lbl >= 0 for lbl in sat_labels])
    s2d_overall = _retrieval(gf[test_sat_mask], gl[test_sat_mask],
                             drone_gal_feats, drone_gal_labels_t)

    t = time.time() - t0
    print(f"  D2S R@1 {d2s_overall['R@1']:6.2f}  R@5 {d2s_overall['R@5']:6.2f}"
          f"  R@10 {d2s_overall['R@10']:6.2f}  AP {d2s_overall['AP']:6.2f}")
    for alt, m in d2s_per_alt.items():
        print(f"     {alt:>4d}m  R@1 {m['R@1']:6.2f}  AP {m['AP']:6.2f}  ({m['n']})")
    print(f"  S2D R@1 {s2d_overall['R@1']:6.2f}  R@5 {s2d_overall['R@5']:6.2f}"
          f"  R@10 {s2d_overall['R@10']:6.2f}  AP {s2d_overall['AP']:6.2f}")
    print(f"  Time {t:.1f}s")
    return {'D2S': dict(d2s_overall, per_alt=d2s_per_alt),
            'S2D': s2d_overall, 'inference_time': t}
