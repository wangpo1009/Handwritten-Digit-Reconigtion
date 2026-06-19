import torch

from src.data.augmentation import get_shift_scale_augmentation


def test_shift_scale_augmentation_keeps_mnist_shape_and_range():
    image = torch.zeros((1, 28, 28), dtype=torch.float32)
    image[:, 8:20, 8:20] = 1.0
    transform = get_shift_scale_augmentation(
        max_shift=4,
        min_scale=0.6,
        max_scale=0.8,
        p=1.0,
        modes=("shift_scale",),
    )

    augmented = transform(image)

    assert augmented.shape == image.shape
    assert augmented.min().item() >= 0.0
    assert augmented.max().item() <= 1.0
    assert not torch.equal(augmented, image)
