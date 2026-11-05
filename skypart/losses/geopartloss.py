"""GeoPartLoss: 4-group Kendall uncertainty-weighted multi-task loss.

Groups: alignment / distillation / part / altitude.
Total = sum_g [exp(-s_g) * L_g + s_g], where s_g = log sigma_g^2 is learned.
"""
import torch
import torch.nn as nn

from ..config import CFG
from .components import (
    SupInfoNCELoss, CircleLoss, ProxyAnchorLoss, UAPALoss,
    CrossDistillationLoss, EMADistillationLoss, VICRegLoss,
    PatchNCELoss, CrossViewGraphConsistencyLoss, PrototypeDiversityLoss,
)


class GeoPartLoss(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, num_groups: int = 4, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.log_vars = nn.Parameter(torch.zeros(num_groups))
        self.infonce = SupInfoNCELoss(temp=0.05)
        self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
        self.proxy_anchor = ProxyAnchorLoss(num_classes, embed_dim,
                                             margin=cfg.PROXY_MARGIN, alpha=cfg.PROXY_ALPHA)
        self.circle_loss = CircleLoss(margin=cfg.CIRCLE_MARGIN, gamma=cfg.CIRCLE_GAMMA)
        self.uapa = UAPALoss(base_temperature=cfg.DISTILL_TEMP)
        self.cross_dist = CrossDistillationLoss()
        self.ema_dist = EMADistillationLoss()
        self.diversity = PrototypeDiversityLoss()
        self.graph_consistency = CrossViewGraphConsistencyLoss()
        self.patch_nce = PatchNCELoss(n_samples=cfg.PATCH_NCE_SAMPLES)
        self.vicreg = VICRegLoss(cfg.VICREG_SIM_COEFF, cfg.VICREG_STD_COEFF, cfg.VICREG_COV_COEFF)
        self.recon_warmup = cfg.RECON_WARMUP

    def forward(self, d_out, s_out, labels, model, teacher, ema,
                drone, sat, alt_idx, alt_norm, epoch, device):
        cfg = self.cfg
        loss_dict = {}

        l_ce = (self.ce(d_out['logits'], labels) + self.ce(s_out['logits'], labels))
        l_ce += 0.3 * (self.ce(d_out['cls_logits'], labels) + self.ce(s_out['cls_logits'], labels))
        l_nce = self.infonce(d_out['embedding'], s_out['embedding'], labels)
        l_proxy = 0.5 * (self.proxy_anchor(d_out['embedding'], labels)
                         + self.proxy_anchor(s_out['embedding'], labels))
        all_emb = torch.cat([d_out['embedding'], s_out['embedding']], dim=0)
        all_lbl = torch.cat([labels, labels], dim=0)
        l_circle = self.circle_loss(all_emb, all_lbl)
        l_uapa = self.uapa(d_out['logits'], s_out['logits'])
        l_graph = cfg.GRAPH_LOSS_WEIGHT * self.graph_consistency(
            d_out['graph_emb'], s_out['graph_emb'], labels)
        l_pnce = cfg.PATCH_NCE_WEIGHT * self.patch_nce(
            d_out['parts']['projected_patches'], s_out['parts']['projected_patches'])
        l_vicreg = cfg.VICREG_WEIGHT * self.vicreg(d_out['embedding'], s_out['embedding'])
        L_align = l_ce + l_nce + l_proxy + l_uapa + l_graph + l_circle + l_pnce + l_vicreg

        if teacher is not None:
            with torch.no_grad():
                t_drone = teacher(drone)
                t_sat = teacher(sat)
            l_cross = (self.cross_dist(d_out['projected_feat'], t_drone)
                       + self.cross_dist(s_out['projected_feat'], t_sat))
        else:
            l_cross = torch.tensor(0.0, device=device)
        with torch.no_grad():
            ema_drone_emb = ema.forward(drone, alt_idx=alt_idx)
            ema_sat_emb = ema.forward(sat, alt_idx=None)
        l_ema = 0.5 * (self.ema_dist(d_out['embedding'], ema_drone_emb)
                       + self.ema_dist(s_out['embedding'], ema_sat_emb))
        L_distill = l_cross + l_ema

        if epoch >= self.recon_warmup:
            l_recon_d = model.mask_recon(
                d_out['parts']['projected_patches'],
                d_out['parts']['part_features'],
                d_out['parts']['assignment'])
            l_recon_s = model.mask_recon(
                s_out['parts']['projected_patches'],
                s_out['parts']['part_features'],
                s_out['parts']['assignment'])
            l_recon = 0.5 * (l_recon_d + l_recon_s)
        else:
            l_recon = torch.tensor(0.0, device=device)
        l_div = self.diversity(model.part_disc.prototypes)
        l_sparse = cfg.GATE_SPARSITY * (
            d_out['parts'].get('gate_sparsity', 0) + s_out['parts'].get('gate_sparsity', 0))
        L_part = l_recon + l_div + l_sparse

        # The altitude-supervision term is omitted on datasets without altitude
        # labels (University-1652): its group then contributes nothing.
        has_alt = cfg.NUM_ALTITUDES > 1
        l_alt = (model.alt_pred(d_out['embedding'].detach(), alt_norm) if has_alt
                 else torch.tensor(0.0, device=device))

        groups = [L_align, L_distill, L_part, l_alt]
        names = ['align', 'distill', 'part', 'alt']
        total = torch.tensor(0.0, device=device)
        for i, (L, name) in enumerate(zip(groups, names)):
            if name == 'alt' and not has_alt:
                loss_dict['G_alt'] = 0.0
                continue
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * L + self.log_vars[i]
            loss_dict[f'G_{name}'] = float(L.item())
        loss_dict['total'] = float(total.item())
        return total, loss_dict
