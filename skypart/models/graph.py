"""Graph attention readout over active prototype nodes."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, adj):
        B, K, _ = x.shape
        h = self.W(x).view(B, K, self.n_heads, self.head_dim)
        src_score = (h * self.a_src).sum(-1)
        dst_score = (h * self.a_dst).sum(-1)
        attn = self.leaky_relu(src_score.unsqueeze(2) + dst_score.unsqueeze(1))
        mask = adj.unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
        attn = attn.masked_fill(mask == 0, 0)
        h_perm = h.permute(0, 2, 1, 3)
        attn_perm = attn.permute(0, 3, 1, 2)
        return torch.matmul(attn_perm, h_perm).permute(0, 2, 1, 3).reshape(B, K, -1)


class PartGraphEncoder(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, out_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.gat1 = SimpleGATLayer(in_dim, hidden_dim, n_heads)
        self.gat2 = SimpleGATLayer(hidden_dim, out_dim, n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.readout_attn = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2), nn.Tanh(), nn.Linear(out_dim // 2, 1)
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def build_adjacency(self, part_positions):
        B, K, _ = part_positions.shape
        if K <= 12:
            return torch.ones(B, K, K, device=part_positions.device)
        dist = torch.cdist(part_positions, part_positions)
        _, topk_idx = dist.topk(7, dim=-1, largest=False)
        adj = torch.zeros(B, K, K, device=part_positions.device)
        adj.scatter_(2, topk_idx, 1)
        return (adj + adj.transpose(1, 2)).clamp(max=1)

    def forward(self, part_features, part_positions):
        adj = self.build_adjacency(part_positions)
        h = F.gelu(self.norm1(self.gat1(part_features, adj)))
        h = self.norm2(self.gat2(h, adj))
        h = h + part_features
        attn = F.softmax(self.readout_attn(h), dim=1)
        return F.normalize(self.proj((attn * h).sum(1)), dim=-1)
