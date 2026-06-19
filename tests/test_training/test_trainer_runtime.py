import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.neural_network import PoModel
from src.training.trainer import measure_inference_time, test_model as evaluate_model


def test_model_returns_loss_accuracy_and_sample_count():
    model = PoModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.rand(8, 1, 28, 28),
            torch.randint(0, 10, (8,)),
        ),
        batch_size=4,
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device="cpu",
    )

    assert set(metrics) == {"loss", "accuracy", "samples"}
    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["samples"] == 8.0


def test_measure_inference_time_returns_runtime_metrics():
    model = PoModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.rand(8, 1, 28, 28),
            torch.randint(0, 10, (8,)),
        ),
        batch_size=4,
    )

    timing = measure_inference_time(
        model=model,
        dataloader=dataloader,
        device="cpu",
        warmup_batches=1,
    )

    assert timing["total_seconds"] > 0.0
    assert timing["avg_ms_per_batch"] > 0.0
    assert timing["avg_ms_per_sample"] > 0.0
    assert timing["samples_per_second"] > 0.0
    assert timing["batches"] == 2.0
    assert timing["samples"] == 8.0
