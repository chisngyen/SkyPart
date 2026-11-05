"""SkyPart student model: DINOv2 ViT-S/14 + part discovery + 3-way fusion.

At inference: cosine retrieval over the fused 768-D embedding. No teacher,
no EMA, no altitude metadata required.
"""
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG
from .backbone import DINOv2Backbone
from .part_discovery import AltitudeAwarePartDiscovery, PartAwarePooling
from .fusion import ThreeWayFusionGate
from .graph import PartGraphEncoder
from .heads import MaskedPartReconstruction, AltitudePredictionHead


class SkyPartModel(nn.Module):
    def __init__(self, num_classes: int, cfg=CFG):
        super().__init__()
        self.backbone = DINOv2Backbone(
            repo=cfg.DINOV2_REPO,
            model=cfg.DINOV2_STUDENT_MODEL,
            unfreeze_blocks=cfg.UNFREEZE_BLOCKS,
        )
        self.part_disc = AltitudeAwarePartDiscovery(
            self.backbone.feature_dim, cfg.N_PARTS, cfg.PART_DIM,
            cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES,
        )
        self.pool = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = ThreeWayFusionGate(cfg.EMBED_DIM)
        self.graph_encoder = PartGraphEncoder(
            in_dim=cfg.PART_DIM, hidden_dim=cfg.GAT_HIDDEN,
            out_dim=cfg.GRAPH_EMBED_DIM, n_heads=cfg.GAT_HEADS,
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(cfg.GRAPH_EMBED_DIM, cfg.EMBED_DIM), nn.LayerNorm(cfg.EMBED_DIM)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True),
        )
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj = nn.Sequential(
            nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM), nn.LayerNorm(cfg.TEACHER_DIM)
        )
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred = AltitudePredictionHead(cfg.EMBED_DIM)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SkyPart student: {total/1e6:.1f}M trainable parameters")

    def _embed_single(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        graph_emb = self.graph_encoder(parts['part_features'], parts['part_positions'])
        graph_emb = F.normalize(self.graph_proj(graph_emb), dim=-1)
        fused = self.fusion_gate(F.normalize(emb, dim=-1), cls_emb, graph_emb)
        return fused, patches

    def extract_embedding(self, x, alt_idx=None, return_patches=False):
        # Single forward pass, no TTA, no re-ranking.
        emb, patches = self._embed_single(x, alt_idx=alt_idx)
        if return_patches:
            return emb, F.normalize(patches, dim=-1)
        return emb

    def forward(self, x, alt_idx=None, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb)
        logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        graph_emb = self.graph_encoder(parts['part_features'], parts['part_positions'])
        graph_emb_proj = F.normalize(self.graph_proj(graph_emb), dim=-1)
        fused = self.fusion_gate(F.normalize(emb, dim=-1), cls_emb_norm, graph_emb_proj)
        projected_feat = self.teacher_proj(emb)
        out = {
            'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
            'projected_feat': projected_feat, 'part_emb': emb,
            'cls_emb': cls_emb_norm, 'graph_emb': graph_emb,
        }
        if return_parts:
            out['parts'] = parts
        return out
