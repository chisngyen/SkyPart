"""University-1652 evaluator (4-folder D2S + S2D protocol).

Single-pass cosine retrieval. No re-ranking, no TTA.
No altitude metadata: alt_idx=None at inference (mean-FiLM, paper Sec.3.3).
"""
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import CFG
from ..data.transforms import get_drone_transforms


@torch.no_grad()
def _embed_paths(model, samples, labels, device, cfg=CFG, weather_aug=None):
    from ..data.weather import apply_weather as _apply  # local import
    tf = get_drone_transforms("test")

    class _DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            img = Image.open(samples[i]).convert('RGB')
            if weather_aug is not None:
                img = _apply(img, weather_aug)
            return {'img': tf(img), 'lbl': labels[i]}

    loader = DataLoader(_DS(), batch_size=cfg.EVAL_BATCH_SIZE,
                        num_workers=cfg.EVAL_NUM_WORKERS, pin_memory=True)
    feats, lbls = [], []
    for b in loader:
        with torch.amp.autocast('cuda', enabled=cfg.USE_AMP):
            feats.append(model.extract_embedding(b['img'].to(device), alt_idx=None).cpu())
        lbls.append(b['lbl'])
    return torch.cat(feats), torch.cat(lbls)


def _retrieval_metrics(qf, ql, gf, gl):
    sim = qf @ gf.T
    _, rank = sim.sort(1, descending=True)
    N = qf.size(0)
    r1 = r5 = r10 = 0
    ap_sum = 0.0
    for i in range(N):
        m = torch.where(gl[rank[i]] == ql[i])[0]
        if len(m) > 0:
            f = m[0].item()
            r1 += int(f < 1)
            r5 += int(f < 5)
            r10 += int(f < 10)
            ap_sum += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
    return {'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
            'R@10': r10 / N * 100, 'AP': ap_sum / N * 100, 'n': N}


@torch.no_grad()
def evaluate(model, test_ds, device, cfg=CFG, weather_drone_aug=None):
    """Evaluate Uni-1652. weather_drone_aug: optional imgaug Sequential applied to
    DRONE-side images only (D2S queries + S2D gallery); satellite stays clean."""
    model.eval()
    qf_d, ql_d = _embed_paths(model, test_ds.d2s_query_samples,
                              test_ds.d2s_query_labels, device, cfg,
                              weather_aug=weather_drone_aug)
    gf_d, gl_d = _embed_paths(model, test_ds.d2s_gallery_samples,
                              test_ds.d2s_gallery_labels, device, cfg)
    qf_s, ql_s = _embed_paths(model, test_ds.s2d_query_samples,
                              test_ds.s2d_query_labels, device, cfg)
    gf_s, gl_s = _embed_paths(model, test_ds.s2d_gallery_samples,
                              test_ds.s2d_gallery_labels, device, cfg,
                              weather_aug=weather_drone_aug)
    d2s = _retrieval_metrics(qf_d, ql_d, gf_d, gl_d)
    s2d = _retrieval_metrics(qf_s, ql_s, gf_s, gl_s)
    print(f"  D2S R@1 {d2s['R@1']:6.2f}  R@5 {d2s['R@5']:6.2f}"
          f"  R@10 {d2s['R@10']:6.2f}  AP {d2s['AP']:6.2f}")
    print(f"  S2D R@1 {s2d['R@1']:6.2f}  R@5 {s2d['R@5']:6.2f}"
          f"  R@10 {s2d['R@10']:6.2f}  AP {s2d['AP']:6.2f}")
    return {'D2S': d2s, 'S2D': s2d}
