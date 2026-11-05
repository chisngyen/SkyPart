"""Altitude-aware semantic part discovery + part-aware pooling.

Single-pass cosine assignment over a learnable prototype bank with a per-image
Gumbel-sigmoid salience gate (typical active K=10 / bank K_max=12).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import DeepAltitudeFiLM


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim: int = 384, n_parts: int = 8, part_dim: int = 256,
                 temperature: float = 0.07, num_altitudes: int = 4):
        super().__init__()
        self.n_parts = n_parts
        self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim),
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.part_gate = nn.Sequential(
            nn.Linear(part_dim, part_dim // 4), nn.GELU(),
            nn.Linear(part_dim // 4, 1),
        )
        nn.init.constant_(self.part_gate[-1].bias, 2.0)

    @staticmethod
    def _gumbel_sigmoid(logits, tau: float = 0.5, hard: bool = False):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y_soft = torch.sigmoid((logits + gumbel_noise) / tau)
        if hard:
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape
        H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)

        gate_logits = self.part_gate(part_feat).squeeze(-1)
        if self.training:
            gate = self._gumbel_sigmoid(gate_logits, tau=0.5, hard=True)
        else:
            gate = (torch.sigmoid(gate_logits) > 0.5).float()
        active_count = gate.sum(dim=1)
        for b_idx in range(B):
            if active_count[b_idx] < 4:
                _, topk = torch.sigmoid(gate_logits[b_idx]).topk(4)
                gate[b_idx] = 0.0
                gate[b_idx, topk] = 1.0

        part_feat = part_feat * gate.unsqueeze(-1)

        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        gate_sparsity = torch.sigmoid(gate_logits).mean()

        return {
            'part_features': part_feat, 'part_positions': part_pos,
            'assignment': assign, 'salience': salience,
            'projected_patches': feat, 'gate_sparsity': gate_sparsity,
            'gate': gate,
        }


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim: int = 256, embed_dim: int = 768):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, part_features, salience=None):
        aw = self.attn(part_features)
        if salience is not None:
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1)
        max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)
