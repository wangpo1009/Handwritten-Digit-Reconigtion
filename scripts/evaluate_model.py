import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import data_loader
from src.data.augmentation import get_shift_scale_augmentation
from src.models.neural_network import PoModel
from src.training.trainer import measure_inference_time, test_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "saved" / "latest" / "model.pt")
    return parser.parse_args()


def build_test_loaders(test_images, test_labels, batch_size):
    configs = {
        "original": None,
        "shifted": get_shift_scale_augmentation(
            max_shift=6,
            min_scale=1.0,
            max_scale=1.0,
            p=1.0,
            modes=("shift",),
        ),
        "small": get_shift_scale_augmentation(
            max_shift=0,
            min_scale=0.55,
            max_scale=0.75,
            p=1.0,
            modes=("scale",),
        ),
        "shifted_small": get_shift_scale_augmentation(
            max_shift=6,
            min_scale=0.55,
            max_scale=0.75,
            p=1.0,
            modes=("shift_scale",),
        ),
    }

    return {
        name: DataLoader(
            data_loader.MNISTDataset(test_images, test_labels, transform=transform),
            batch_size=batch_size,
            shuffle=False,
        )
        for name, transform in configs.items()
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_images, test_labels = data_loader.load_mnist()
    test_loaders = build_test_loaders(test_images, test_labels, args.batch_size)

    model = PoModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    loss_fn = nn.CrossEntropyLoss()

    results = {
        name: test_model(model, loader, loss_fn, device)
        for name, loader in test_loaders.items()
    }
    results["inference_timing"] = measure_inference_time(
        model,
        test_loaders["original"],
        device=device,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
