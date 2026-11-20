"""Train the CNN model on the UCI Connect-4 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cnn_model import CNNTrainer
from data_loader import load_uci_data, train_val_split
from download_data import download

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "connect-4.data"
MODEL_PATH = ROOT / "models" / "connect4_cnn.pt"


def train(
    epochs: int = 25,
    learning_rate: float = 0.001,
    batch_size: int = 256,
) -> CNNTrainer:
    if not DATA_FILE.exists():
        download()

    print("Loading UCI Connect-4 data (CNN planes) ...")
    x, y = load_uci_data(DATA_FILE, cnn=True)
    print(f"  samples: {len(x)}, shape: {x.shape[1:]}, classes: {len(np.unique(y))}")

    x_train, y_train, x_val, y_val = train_val_split(x, y)
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    print("  architecture: Conv(3->32->64->64) + FC(128) -> 3")

    trainer = CNNTrainer()
    train_acc, val_acc = trainer.train(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    print(f"Final  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    trainer.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Connect-4 CNN")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(42)
    train(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
