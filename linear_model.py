"""Linear and logistic models for Connect-4 position evaluation (NumPy only)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np

Connect4Model = Union["LinearConnect4Model", "LogisticConnect4Model"]


def _log_every(epochs: int) -> int:
    return 10 if epochs > 20 else max(1, epochs // 5)


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred >= 0.5) == (y_true >= 0.5)))


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    clipped = np.clip(y_pred, eps, 1.0 - eps)
    losses = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    mse = _mean_squared_error(y_true, y_pred)
    acc = _accuracy_score(y_true, y_pred)
    return mse, acc, _r2_score(y_true, y_pred)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class _SGDLinearRegressor:
    """One-pass SGD linear regression: y = w·x + b."""

    def __init__(self, learning_rate: float = 0.01, random_state: int = 42) -> None:
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _ensure_initialized(self, n_features: int) -> None:
        if self.weights is None:
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0.0, 0.01, size=n_features)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._ensure_initialized(x.shape[1])

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(x))
        for idx in indices:
            sample = x[idx]
            target = y[idx]
            prediction = float(np.dot(self.weights, sample) + self.bias)
            error = target - prediction
            self.weights += self.learning_rate * error * sample
            self.bias += self.learning_rate * error

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet.")
        x = np.asarray(x, dtype=np.float64)
        return x @ self.weights + self.bias


class _SGDLogisticClassifier:
    """One-pass SGD logistic regression with log loss."""

    def __init__(self, learning_rate: float = 0.01, random_state: int = 42) -> None:
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _ensure_initialized(self, n_features: int) -> None:
        if self.weights is None:
            rng = np.random.default_rng(self.random_state)
            self.weights = rng.normal(0.0, 0.01, size=n_features)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._ensure_initialized(x.shape[1])

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(len(x))
        for idx in indices:
            sample = x[idx]
            target = y[idx]
            linear = float(np.dot(self.weights, sample) + self.bias)
            probability = float(_sigmoid(linear))
            error = probability - target
            self.weights -= self.learning_rate * error * sample
            self.bias -= self.learning_rate * error

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet.")
        x = np.asarray(x, dtype=np.float64)
        positive = _sigmoid(x @ self.weights + self.bias)
        return np.column_stack((1.0 - positive, positive))


class LinearConnect4Model:
    """Predicts P(X wins) from a 42-dimensional board encoding."""

    MODEL_TYPE = "linear"

    def __init__(self) -> None:
        self.model = _SGDLinearRegressor(learning_rate=0.01, random_state=42)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (42,) or (N, 42) -> win probability in [0, 1]."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, ...]
        raw = self.model.predict(x)
        probs = np.clip(raw, 0.0, 1.0)
        return probs[0] if single else probs

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
    ) -> Tuple[float, float, float, float]:
        log_every = _log_every(epochs)

        for epoch in range(1, epochs + 1):
            self.model.fit(x_train, y_train)
            if epoch == 1 or epoch == epochs or epoch % log_every == 0:
                train_pred = self.predict(x_train)
                val_pred = self.predict(x_val)
                train_mse, train_acc, _ = _classification_metrics(y_train, train_pred)
                val_mse, val_acc, _ = _classification_metrics(y_val, val_pred)
                print(
                    f"  epoch {epoch:3d}/{epochs}  "
                    f"train MSE={train_mse:.4f} acc={train_acc:.4f}  "
                    f"val MSE={val_mse:.4f} acc={val_acc:.4f}"
                )

        train_pred = self.predict(x_train)
        val_pred = self.predict(x_val)
        train_mse, train_acc, train_r2 = _classification_metrics(y_train, train_pred)
        val_mse, val_acc, val_r2 = _classification_metrics(y_val, val_pred)
        print(
            f"  final  train MSE={train_mse:.4f} R2={train_r2:.4f} acc={train_acc:.4f}"
        )
        print(
            f"         val   MSE={val_mse:.4f} R2={val_r2:.4f} acc={val_acc:.4f}"
        )
        return train_acc, val_acc, train_mse, val_mse

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.MODEL_TYPE,
            "weights": self.model.weights,
            "bias": self.model.bias,
            "learning_rate": self.model.learning_rate,
            "random_state": self.model.random_state,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> LinearConnect4Model:
        payload = joblib.load(path)
        instance = cls()
        instance.model.weights = np.asarray(payload["weights"], dtype=np.float64)
        instance.model.bias = float(payload["bias"])
        return instance


class LogisticConnect4Model:
    """Predicts P(X wins) from a 42-dimensional board encoding via logistic regression."""

    MODEL_TYPE = "logistic"

    def __init__(self) -> None:
        self.model = _SGDLogisticClassifier(learning_rate=0.01, random_state=42)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (42,) or (N, 42) -> win probability in [0, 1]."""
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, ...]
        probs = self.model.predict_proba(x)[:, 1]
        return float(probs[0]) if single else probs

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
    ) -> Tuple[float, float, float, float]:
        log_every = _log_every(epochs)
        y_train_cls = y_train.astype(int)
        y_val_cls = y_val.astype(int)

        for epoch in range(1, epochs + 1):
            self.model.fit(x_train, y_train_cls)
            if epoch == 1 or epoch == epochs or epoch % log_every == 0:
                train_pred = self.predict(x_train)
                val_pred = self.predict(x_val)
                train_mse, train_acc, _ = _classification_metrics(y_train, train_pred)
                val_mse, val_acc, _ = _classification_metrics(y_val, val_pred)
                train_logloss = _log_loss(y_train_cls, train_pred)
                val_logloss = _log_loss(y_val_cls, val_pred)
                print(
                    f"  epoch {epoch:3d}/{epochs}  "
                    f"train logloss={train_logloss:.4f} acc={train_acc:.4f}  "
                    f"val logloss={val_logloss:.4f} acc={val_acc:.4f}"
                )

        train_pred = self.predict(x_train)
        val_pred = self.predict(x_val)
        train_mse, train_acc, train_r2 = _classification_metrics(y_train, train_pred)
        val_mse, val_acc, val_r2 = _classification_metrics(y_val, val_pred)
        train_logloss = _log_loss(y_train_cls, train_pred)
        val_logloss = _log_loss(y_val_cls, val_pred)
        print(
            f"  final  train logloss={train_logloss:.4f} R2={train_r2:.4f} acc={train_acc:.4f}"
        )
        print(
            f"         val   logloss={val_logloss:.4f} R2={val_r2:.4f} acc={val_acc:.4f}"
        )
        return train_acc, val_acc, train_mse, val_mse

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.MODEL_TYPE,
            "weights": self.model.weights,
            "bias": self.model.bias,
            "learning_rate": self.model.learning_rate,
            "random_state": self.model.random_state,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> LogisticConnect4Model:
        payload = joblib.load(path)
        instance = cls()
        instance.model.weights = np.asarray(payload["weights"], dtype=np.float64)
        instance.model.bias = float(payload["bias"])
        return instance


def load_model(path: Path) -> Connect4Model:
    """Load a saved linear or logistic model from disk."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model_type" in payload:
        if payload["model_type"] == "logistic":
            return LogisticConnect4Model.load(path)
        return LinearConnect4Model.load(path)

    raise ValueError(
        f"Unsupported model file at {path}. Retrain with: python train.py"
    )
