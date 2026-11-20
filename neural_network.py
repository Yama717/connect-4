"""Simple feedforward neural network (NumPy) for position evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class NeuralNetwork:
    """MLP: input -> hidden (ReLU) -> ... -> softmax (win/loss/draw)."""

    def __init__(self, layer_sizes: List[int], seed: int = 42) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output")
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0, scale, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        pre_activations: List[np.ndarray] = []
        current = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                current = _relu(z)
            else:
                current = _softmax(z)
            activations.append(current)
        return current, activations, pre_activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs, _, _ = self.forward(x)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> float:
        n = len(x)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb, yb = x[batch_idx], y[batch_idx]
            loss = self._train_batch(xb, yb, learning_rate)
            total_loss += loss
            batches += 1
        return total_loss / max(batches, 1)

    def _train_batch(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        probs, activations, pre_activations = self.forward(x)
        m = x.shape[0]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(m), y] = 1.0
        delta = (probs - one_hot) / m
        loss = -np.mean(np.log(probs[np.arange(m), y] + 1e-9))

        grad_w: List[np.ndarray] = []
        grad_b: List[np.ndarray] = []
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w.insert(0, activations[i].T @ delta)
            grad_b.insert(0, np.sum(delta, axis=0))
            if i > 0:
                delta = (delta @ self.weights[i].T) * _relu_grad(pre_activations[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= lr * grad_w[i]
            self.biases[i] -= lr * grad_b[i]
        return float(loss)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        probs, _, _ = self.forward(x)
        preds = np.argmax(probs, axis=1)
        accuracy = float(np.mean(preds == y))
        loss = -float(np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9)))
        return loss, accuracy

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> NeuralNetwork:
        payload = json.loads(path.read_text())
        net = cls(payload["layer_sizes"])
        net.weights = [np.asarray(w, dtype=np.float32) for w in payload["weights"]]
        net.biases = [np.asarray(b, dtype=np.float32) for b in payload["biases"]]
        return net