import time
from typing import Any

import torch


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_logits = model(X)
        loss = loss_fn(y_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred = torch.argmax(y_logits, dim=1)
        train_acc += (y_pred == y).sum().item() / len(y)

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_model(model, dataloader, loss_fn, device) -> dict[str, float]:
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    total_samples = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            y_pred = torch.argmax(y_logits, dim=1)

            batch_size = len(y)
            test_loss += loss.item()
            test_acc += (y_pred == y).sum().item() / batch_size
            total_samples += batch_size

    return {
        "loss": test_loss / len(dataloader),
        "accuracy": test_acc / len(dataloader),
        "samples": float(total_samples),
    }


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    device,
    epochs: int = 5,
) -> list[dict[str, Any]]:
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_metrics = test_model(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(epoch_result)

        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f}"
        )

    return history


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def measure_inference_time(
    model,
    dataloader,
    device,
    warmup_batches: int = 3,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    total_batches = 0

    with torch.inference_mode():
        for batch_idx, (X, _) in enumerate(dataloader):
            if batch_idx >= warmup_batches:
                break
            X = X.to(device)
            _ = model(X)
        _sync_if_cuda(device)

        start = time.perf_counter()
        for batch_idx, (X, _) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            X = X.to(device)
            _ = model(X)
            total_samples += X.shape[0]
            total_batches += 1

        _sync_if_cuda(device)
        elapsed_seconds = time.perf_counter() - start

    return {
        "total_seconds": elapsed_seconds,
        "avg_ms_per_batch": (elapsed_seconds / total_batches) * 1000,
        "avg_ms_per_sample": (elapsed_seconds / total_samples) * 1000,
        "samples_per_second": total_samples / elapsed_seconds,
        "batches": float(total_batches),
        "samples": float(total_samples),
    }
