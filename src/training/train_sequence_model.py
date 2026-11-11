"""Training loop shared by the LSTM and GRU sequence classifiers."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.sequence_dataset import PoseSequenceDataset
from src.data.uiprmd_dataset import Episode
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_sequence_model(
    model: torch.nn.Module,
    train_episodes: list[Episode],
    val_episodes: list[Episode],
    target_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    patience: int = 10,
) -> dict:
    train_ds = PoseSequenceDataset(train_episodes, target_length)
    val_ds = PoseSequenceDataset(val_episodes, target_length, normalize_stats=train_ds.normalize_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * x.size(0)
        train_loss = train_loss_total / len(train_ds)

        model.eval()
        val_loss_total = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_total += loss.item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
        val_loss = val_loss_total / len(val_ds)
        val_accuracy = correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        logger.info(f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "normalize_stats": train_ds.normalize_stats, "best_val_loss": best_val_loss}


def predict_sequence_model(
    model: torch.nn.Module, episodes: list[Episode], target_length: int, normalize_stats: tuple, device: str
) -> tuple[np.ndarray, np.ndarray]:
    ds = PoseSequenceDataset(episodes, target_length, normalize_stats=normalize_stats)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)
