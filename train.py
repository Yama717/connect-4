"""Train linear or logistic regression on the Connect-4 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_loader import load_regression_data, train_val_split
from download_data import download
from linear_model import Connect4Model, LinearConnect4Model, LogisticConnect4Model

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "connect-4.data"
MODEL_PATHS = {
    "linear": ROOT / "models" / "connect4_linear.joblib",
    "logistic": ROOT / "models" / "connect4_logistic.joblib",
}


def train(model_type: str = "linear", epochs: int = 100) -> Connect4Model:
    if not DATA_FILE.exists():
        download()

    print("Loading Connect-4 data ...")
    x, y = load_regression_data(DATA_FILE)
    print(f"  samples: {len(x)}, features: {x.shape[1]}")
    print(f"  X wins: {int(y.sum())}, X loss/draw: {int(len(y) - y.sum())}")

    x_train, y_train, x_val, y_val = train_val_split(x, y)
    if model_type == "logistic":
        print(
            f"  architecture: NumPy SGD logistic regression "
            f"(42 features -> win probability, {epochs} epochs)"
        )
        model: Connect4Model = LogisticConnect4Model()
    else:
        print(
            f"  architecture: NumPy SGD linear regression "
            f"(42 features -> win probability, {epochs} epochs)"
        )
        model = LinearConnect4Model()

    model.train(x_train, y_train, x_val, y_val, epochs=epochs)
    model_path = MODEL_PATHS[model_type]
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Connect-4 position evaluation model")
    parser.add_argument(
        "--model",
        choices=("linear", "logistic"),
        default="linear",
        help="Model type (default: linear)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    args = parser.parse_args()
    np.random.seed(42)
    train(model_type=args.model, epochs=args.epochs)


if __name__ == "__main__":
    main()
