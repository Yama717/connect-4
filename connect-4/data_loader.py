"""Load and preprocess the UCI / Kaggle Connect-4 dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np

from board import Connect4Board

CLASS_LABELS = ("win", "loss", "draw")
CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}

POSITIVE = 1.0
NEGATIVE = 0.0
VALID_OUTCOMES = {"win", "loss", "draw"}


def _outcome_label(raw: str) -> float | None:
    """Map UCI outcome to P(X wins): win=1, loss/draw=0."""
    outcome = raw.strip().lower()
    if outcome not in VALID_OUTCOMES:
        return None
    return POSITIVE if outcome == "win" else NEGATIVE


def load_uci_data(path: Path, cnn: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Parse connect-4.data into feature matrix X and label vector y."""
    features: List = []
    labels: List[int] = []

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 43:
                continue
            board = Connect4Board.from_uci_row(row[:42])
            if cnn:
                features.append(board.encode_planes())
            else:
                features.append(board.encode())
            labels.append(CLASS_TO_INDEX[row[42].strip()])

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def load_regression_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse connect-4.data into board features X and P(X wins) labels y."""
    features: List[List[float]] = []
    labels: List[float] = []

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 43:
                continue
            label = _outcome_label(row[42])
            if label is None:
                continue
            board = Connect4Board.from_uci_row(row[:42])
            features.append(board.encode())
            labels.append(label)

    if not features:
        raise ValueError(f"No samples parsed from {path}")

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return x, y


def train_val_split(
    x: np.ndarray, y: np.ndarray, val_fraction: float = 0.1, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split = int(len(x) * (1.0 - val_fraction))
    train_idx, val_idx = indices[:split], indices[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
