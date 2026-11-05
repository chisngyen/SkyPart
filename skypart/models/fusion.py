"""ThreeWayFusionGate: learned softmax over (part, CLS, graph) branches."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeWayFusionGate(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 3),
        )
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor([0.6, -1.0, -0.3]))

    def forward(self, part_emb, cls_emb, graph_emb):
        combined = torch.cat([part_emb, cls_emb, graph_emb], dim=-1)
        weights = F.softmax(self.gate(combined), dim=-1)
        fused = (weights[:, 0:1] * part_emb
                 + weights[:, 1:2] * cls_emb
                 + weights[:, 2:3] * graph_emb)
        return F.normalize(fused, dim=-1)
