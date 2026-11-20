"""Convolutional neural network for Connect-4 position evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from board import COLS, ROWS


class Connect4CNN(nn.Module):
    """CNN over 6x7 board with 3 input planes (X, O, empty)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flat = 64 * ROWS * COLS
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNTrainer:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Connect4CNN().to(self.device)

    def predict_proba(self, planes: np.ndarray) -> np.ndarray:
        """planes: (3, 6, 7) or (N, 3, 6, 7) -> probabilities (N, 3)."""
        single = planes.ndim == 3
        if single:
            planes = planes[np.newaxis, ...]
        tensor = torch.from_numpy(planes).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs[0] if single else probs

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 25,
        batch_size: int = 256,
        learning_rate: float = 0.001,
    ) -> Tuple[float, float]:
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train),
                torch.from_numpy(y_train),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        x_val_t = torch.from_numpy(x_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                batches += 1

            train_loss = running_loss / max(batches, 1)
            val_loss, val_acc = self._evaluate(x_val_t, y_val_t, criterion)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            print(
                f"Epoch {epoch:2d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best validation accuracy: {best_val_acc:.4f}")

        x_train_t = torch.from_numpy(x_train).to(self.device)
        y_train_t = torch.from_numpy(y_train).to(self.device)
        _, train_acc = self._evaluate(x_train_t, y_train_t, criterion)
        _, val_acc = self._evaluate(x_val_t, y_val_t, criterion)
        return train_acc, val_acc

    def _evaluate(
        self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module
    ) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            loss = float(criterion(logits, y).item())
            preds = torch.argmax(logits, dim=1)
            accuracy = float((preds == y).float().mean().item())
        return loss, accuracy

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> CNNTrainer:
        trainer = cls(device=device)
        try:
            payload = torch.load(path, map_location=trainer.device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=trainer.device)
        trainer.model.load_state_dict(payload["state_dict"])
        trainer.model.eval()
        return trainer
