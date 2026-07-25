import random
from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@dataclass
class ShiftScaleConfig:
    max_shift: int = 5
    min_scale: float = 0.65
    max_scale: float = 1.0
    p: float = 0.9
    modes: tuple[str, ...] = ("shift", "scale", "shift_scale")


class RandomShiftScale:
    """Augment MNIST tensors for shifted, small, and shifted-small digits."""

    def __init__(self, config: ShiftScaleConfig | None = None):
        self.config = config or ShiftScaleConfig()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() > self.config.p:
            return image

        mode = random.choice(self.config.modes)
        translate = [0, 0]
        scale = 1.0

        if "shift" in mode:
            translate = [
                random.randint(-self.config.max_shift, self.config.max_shift),
                random.randint(-self.config.max_shift, self.config.max_shift),
            ]

        if "scale" in mode:
            scale = random.uniform(self.config.min_scale, self.config.max_scale)

        augmented = F.affine(
            image,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        return augmented.clamp(0.0, 1.0)


def get_shift_scale_augmentation(
    max_shift: int = 5,
    min_scale: float = 0.65,
    max_scale: float = 1.0,
    p: float = 0.9,
    modes: tuple[str, ...] = ("shift", "scale", "shift_scale"),
) -> RandomShiftScale:
    config = ShiftScaleConfig(
        max_shift=max_shift,
        min_scale=min_scale,
        max_scale=max_scale,
        p=p,
        modes=modes,
    )
    return RandomShiftScale(config)
