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
from src.training.trainer import measure_inference_time, train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "saved" / "latest")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_images, train_labels, test_images, test_labels = data_loader.load_mnist()
    train_transform = get_shift_scale_augmentation(
        max_shift=5,
        min_scale=0.60,
        max_scale=1.0,
        p=0.9,
        modes=("shift", "scale", "shift_scale"),
    )

    train_dataset = data_loader.MNISTDataset(train_images, train_labels, transform=train_transform)
    test_dataset = data_loader.MNISTDataset(test_images, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PoModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
    )
    timing = measure_inference_time(model, test_dataloader, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")

    metrics = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "history": history,
        "inference_timing": timing,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {args.output_dir / 'model.pt'}")
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
