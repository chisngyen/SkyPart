"""Shared image transforms.

DiscreteRotation: SUES-200 / Uni-1652 use canonical orientations -> identity at
inference. DenseUAV uses {0,90,180,270} on BOTH views during training (its
satellite tiles are cropped at arbitrary basemap orientations).
"""
import random

from torchvision import transforms

from ..config import CFG

_IM_MEAN = [0.485, 0.456, 0.406]
_IM_STD = [0.229, 0.224, 0.225]


class DiscreteRotation:
    """Sample uniformly from a fixed set of multiples of 90 degrees."""

    def __init__(self, choices=(0, 90, 180, 270)):
        self.choices = choices

    def __call__(self, img):
        angle = random.choice(self.choices)
        if angle == 0:
            return img
        return transforms.functional.rotate(img, angle)


def get_drone_transforms(mode: str = "train", img_size: int = None, rotate: bool = False):
    """`rotate=True` only for DenseUAV (arbitrary basemap orientation)."""
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            *([DiscreteRotation((0, 90, 180, 270))] if rotate else []),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(_IM_MEAN, _IM_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(_IM_MEAN, _IM_STD),
    ])


def get_sat_transforms(mode: str = "train", img_size: int = None, rotate: bool = False):
    """`rotate=True` only for DenseUAV (arbitrary basemap orientation)."""
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            *([DiscreteRotation((0, 90, 180, 270))] if rotate else []),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(_IM_MEAN, _IM_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(_IM_MEAN, _IM_STD),
    ])
