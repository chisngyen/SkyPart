"""DINOv2 ViT-S/14 student + ViT-B/14 teacher."""
import torch
import torch.nn as nn

from ..config import (
    DEFAULT_DINOV2_REPO,
    DEFAULT_DINOV2_STUDENT_MODEL,
    DEFAULT_DINOV2_TEACHER_MODEL,
)


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        unfreeze_blocks: int = 4,
        *,
        repo: str = DEFAULT_DINOV2_REPO,
        model: str = DEFAULT_DINOV2_STUDENT_MODEL,
    ):
        super().__init__()
        self.model = torch.hub.load(repo, model, pretrained=True)
        block_count = len(self.model.blocks)
        if (
            isinstance(unfreeze_blocks, bool)
            or not isinstance(unfreeze_blocks, int)
            or not 0 <= unfreeze_blocks <= block_count
        ):
            raise ValueError(
                "unfreeze_blocks must be an integer between "
                f"0 and {block_count}; got {unfreeze_blocks!r}"
            )
        self.feature_dim = 384
        self.patch_size = 14
        for p in self.model.parameters():
            p.requires_grad = False
        if unfreeze_blocks:
            for blk in self.model.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size
        W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DINOv2Teacher(nn.Module):
    def __init__(
        self,
        *,
        repo: str = DEFAULT_DINOV2_REPO,
        model: str = DEFAULT_DINOV2_TEACHER_MODEL,
    ):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher ...")
        self.model = torch.hub.load(repo, model, pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters():
            p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']
