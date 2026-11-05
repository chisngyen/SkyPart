"""Model size, GFLOPs and per-query latency of the deployed embedding path."""
import copy
import time

import torch
import torch.nn as nn

from ..config import CFG


def print_model_complexity(model, device, img_size: int = None):
    sz = img_size or CFG.IMG_SIZE
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 3, sz, sz).to(device)
    gflops = None
    try:
        from thop import profile as thop_profile

        class _Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m.extract_embedding(x)

        # Profile a copy: thop registers buffers and changes train/eval mode.
        probe = _Wrap(copy.deepcopy(model)).to(device).eval()
        macs, _ = thop_profile(probe, inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
        del probe
    except Exception:
        pass
    with torch.no_grad():
        for _ in range(10):
            model.extract_embedding(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            model.extract_embedding(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000
    model.train(was_training)
    print(f"  Params {total/1e6:.2f}M | trainable {trainable/1e6:.2f}M"
          f" | GFLOPs {gflops if gflops is not None else 'N/A'} | {ms_query:.1f} ms/query")
    return {'total_params': total, 'trainable_params': trainable,
            'gflops': gflops, 'ms_query': ms_query}
