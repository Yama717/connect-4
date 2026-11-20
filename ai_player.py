"""AI player that picks the best next move from model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from board import Connect4Board
from linear_model import load_model
from neural_network import NeuralNetwork

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "connect4_linear.joblib"


@dataclass(frozen=True)
class MoveEvaluation:
    """One legal next move and how the model rates the resulting position."""

    col: int
    p_x_wins: float
    score: float
    reason: str = "model"


class AIPlayer:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self._encode_board = Connect4Board.encode
        if model_path.suffix == ".pt":
            from cnn_model import CNNTrainer

            self.model = CNNTrainer.load(model_path)
            self._encode_board = Connect4Board.encode_planes
        elif model_path.suffix == ".json":
            self.model = NeuralNetwork.load(model_path)
        else:
            self.model = load_model(model_path)

    @staticmethod
    def _opponent(player: str) -> str:
        return "O" if player == "X" else "X"

    def _predict_scores(
        self, encodings: Sequence[list[float] | np.ndarray], player: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return X win probabilities and scores for the player making the move."""
        if hasattr(self.model, "predict_proba"):
            p_x, p_o, p_draw = self.model.predict_proba(
                np.asarray(encodings, dtype=np.float32)
            ).T
            player_wins = p_x if player == "X" else p_o
            return p_x, player_wins + 0.5 * p_draw

        p_x = self.model.predict(np.asarray(encodings, dtype=np.float64))
        return p_x, p_x if player == "X" else 1.0 - p_x

    def _forced_columns(
        self, board: Connect4Board, player: str
    ) -> tuple[List[int], List[int]]:
        """Return immediate winning columns and columns that block opponent wins."""
        opponent = self._opponent(player)
        winning_cols: List[int] = []
        blocking_cols: List[int] = []

        for col in board.legal_columns():
            after_our_move = board.copy()
            after_our_move.drop(col, player)
            if after_our_move.winner() == player:
                winning_cols.append(col)

            after_block = board.copy()
            after_block.drop(col, opponent)
            if after_block.winner() == opponent:
                blocking_cols.append(col)

        return winning_cols, blocking_cols

    def _evaluate_positions(
        self,
        board: Connect4Board,
        columns: Sequence[int],
        player: str,
        reason: str = "model",
    ) -> List[MoveEvaluation]:
        """Run the trained model on the board state after each candidate move."""
        if not columns:
            return []

        encodings: list[list[float] | np.ndarray] = []
        terminal: dict[int, MoveEvaluation] = {}

        for col in columns:
            trial = board.copy()
            trial.drop(col, player)
            winner = trial.winner()
            if winner == player:
                p_x_wins = 1.0 if player == "X" else 0.0
                terminal[col] = MoveEvaluation(col, p_x_wins, 1.0, reason="win")
            elif winner == self._opponent(player):
                p_x_wins = 1.0 if player == "O" else 0.0
                terminal[col] = MoveEvaluation(col, p_x_wins, 0.0, reason="loss")
            elif trial.is_full():
                terminal[col] = MoveEvaluation(col, 0.5, 0.5, reason="draw")
            else:
                encodings.append(self._encode_board(trial))

        evaluations: List[MoveEvaluation] = []
        if encodings:
            preds, scores = self._predict_scores(encodings, player)
            encode_index = 0
            for col in columns:
                if col in terminal:
                    evaluations.append(terminal[col])
                else:
                    p_x_wins = float(preds[encode_index])
                    evaluations.append(
                        MoveEvaluation(
                            col,
                            p_x_wins,
                            float(scores[encode_index]),
                            reason=reason,
                        )
                    )
                    encode_index += 1
        else:
            evaluations.extend(terminal[col] for col in columns if col in terminal)

        return evaluations

    def evaluate_moves(self, board: Connect4Board) -> List[MoveEvaluation]:
        """Score every legal next move; best move is first after sorting by score."""
        player = board.current_player()
        legal = board.legal_columns()
        if not legal:
            raise ValueError("No legal moves")

        winning_cols, blocking_cols = self._forced_columns(board, player)
        if winning_cols:
            return self._evaluate_positions(board, winning_cols, player, reason="win")

        candidate_cols = blocking_cols if blocking_cols else legal
        reason = "block+model" if blocking_cols else "model"
        evaluations = self._evaluate_positions(board, candidate_cols, player, reason=reason)
        evaluations.sort(key=lambda item: item.score, reverse=True)
        return evaluations

    def choose_move(self, board: Connect4Board) -> int:
        """Pick the column with the highest model score among legal next positions."""
        evaluations = self.evaluate_moves(board)
        return evaluations[0].col

    def choose_move_with_analysis(
        self, board: Connect4Board
    ) -> tuple[int, List[MoveEvaluation]]:
        """Return the chosen column and all scored candidate moves (best first)."""
        evaluations = self.evaluate_moves(board)
        return evaluations[0].col, evaluations
