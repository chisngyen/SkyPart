"""Altitude-conditioned FiLM modulation.

Train: per-altitude affine. Inference: marginalized mean (no altitude metadata).
"""
import torch
import torch.nn as nn


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes: int = 4, feat_dim: int = 256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        gamma = self.gamma[alt_idx]
        beta = self.beta[alt_idx]
        return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)
