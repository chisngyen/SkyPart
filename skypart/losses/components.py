"""Building-block losses combined by GeoPartLoss.

Alignment   : SupInfoNCE, Circle, ProxyAnchor, UAPA, PatchNCE, VICReg,
              CrossViewGraphConsistency
Distillation: CrossDistillation (teacher), EMADistillation
Part        : PrototypeDiversity (+ MaskedPartReconstruction in models.heads)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupInfoNCELoss(nn.Module):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb, r_emb, labels):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.t() / t
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, dim=-1)
        t = F.normalize(teacher_feat, dim=-1)
        return F.mse_loss(s, t) + (1.0 - F.cosine_similarity(s, t).mean())


class UAPALoss(nn.Module):
    def __init__(self, base_temperature: float = 4.0):
        super().__init__()
        self.T0 = base_temperature

    @staticmethod
    def _entropy(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    def forward(self, drone_logits, sat_logits):
        delta_U = self._entropy(drone_logits) - self._entropy(sat_logits)
        T = self.T0 * (1 + torch.sigmoid(delta_U))
        p_sat = F.softmax(sat_logits / T, dim=1).detach()
        return (T ** 2) * F.kl_div(
            F.log_softmax(drone_logits / T, dim=1), p_sat, reduction='batchmean'
        )


class CircleLoss(nn.Module):
    def __init__(self, margin: float = 0.25, gamma: int = 256):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.Op = 1 + margin
        self.On = -margin

    def forward(self, embeddings, labels):
        sim = embeddings @ embeddings.T
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        pos_mask = labels_eq.float() - torch.eye(len(labels), device=sim.device)
        neg_mask = (~labels_eq).float()
        ap = torch.clamp_min(self.Op - sim.detach(), 0) * pos_mask
        an = torch.clamp_min(sim.detach() - self.On, 0) * neg_mask
        logit_p = -self.gamma * ap * (sim - (1 - self.margin))
        logit_n = self.gamma * an * (sim - self.margin)
        loss_p = torch.logsumexp(logit_p.masked_fill(pos_mask == 0, -1e9), dim=1)
        loss_n = torch.logsumexp(logit_n.masked_fill(neg_mask == 0, -1e9), dim=1)
        return F.softplus(loss_p + loss_n).mean()


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, margin: float = 0.1, alpha: int = 32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=-1)
        sim = embeddings @ P.T
        one_hot = F.one_hot(labels, self.num_classes).float()
        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        P_plus = one_hot.sum(0)
        has_pos = P_plus > 0
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = (pos_term[has_pos].mean() if has_pos.sum() > 0
                    else torch.tensor(0.0, device=embeddings.device))
        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        neg_loss = torch.log(1 + neg_exp.sum(0)).mean()
        return pos_loss + neg_loss


class EMADistillationLoss(nn.Module):
    def forward(self, student_emb, ema_emb):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_a, z_b):
        sim_loss = F.mse_loss(z_a, z_b)
        std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = (torch.relu(1 - std_a).mean() + torch.relu(1 - std_b).mean()) / 2
        B, D = z_a.shape
        z_a_c = z_a - z_a.mean(dim=0)
        z_b_c = z_b - z_b.mean(dim=0)
        cov_a = (z_a_c.T @ z_a_c) / (B - 1)
        cov_b = (z_b_c.T @ z_b_c) / (B - 1)
        off_a = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
        off_b = cov_b.pow(2).sum() - cov_b.diagonal().pow(2).sum()
        cov_loss = (off_a + off_b) / (2 * D)
        return self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss


class PatchNCELoss(nn.Module):
    def __init__(self, n_samples: int = 64, temp: float = 0.07):
        super().__init__()
        self.n_samples = n_samples
        self.temp = temp

    def forward(self, drone_patches, sat_patches):
        B, N, _ = drone_patches.shape
        n = min(self.n_samples, N)
        idx = torch.randperm(N, device=drone_patches.device)[:n]
        d_sel = F.normalize(drone_patches[:, idx], dim=-1)
        s_sel = F.normalize(sat_patches[:, idx], dim=-1)
        loss = 0.0
        for i in range(B):
            sim = d_sel[i] @ s_sel[i].T / self.temp
            labels = torch.arange(n, device=sim.device)
            loss = loss + F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss / (2 * B)


class CrossViewGraphConsistencyLoss(nn.Module):
    def forward(self, drone_graph_emb, sat_graph_emb, labels):
        sim = drone_graph_emb @ sat_graph_emb.T
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
        return (1 - (sim * labels_eq).sum(1) / labels_eq.sum(1).clamp(min=1)).mean()


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1)
        sim = P @ P.T
        K = sim.size(0)
        mask = 1 - torch.eye(K, device=sim.device)
        return (sim * mask).abs().sum() / (K * (K - 1))
