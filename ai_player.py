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

    def search_value(self) -> float:
        """Rank proven outcomes beyond even a saturated model prediction."""
        if self.reason == "win":
            return 2.0
        if self.reason == "loss":
            return -1.0
        return self.score


class AIPlayer:
    def __init__(self, model_path: Path = DEFAULT_MODEL, depth: int = 1) -> None:
        if depth < 1:
            raise ValueError("Search depth must be at least 1")
        self.depth = depth
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
        root_player: str,
        depth: int,
    ) -> List[MoveEvaluation]:
        """Score moves from the root player's perspective, assuming strongest replies."""
        results: dict[int, MoveEvaluation] = {}
        leaf_columns = []
        encodings = []

        for col in columns:
            trial = board.copy()
            trial.drop(col, player)
            if trial.winner() == player:
                root_won = player == root_player
                results[col] = MoveEvaluation(
                    col, float(player == "X"), float(root_won), "win" if root_won else "loss"
                )
            elif trial.is_full():
                results[col] = MoveEvaluation(col, 0.0, 0.5, "draw")
            elif depth == 1:
                leaf_columns.append(col)
                encodings.append(self._encode_board(trial))
            else:
                opponent = self._opponent(player)
                replies = self._evaluate_positions(
                    trial, trial.legal_columns(), opponent, root_player, depth - 1
                )
                select = max if opponent == root_player else min
                best = select(replies, key=MoveEvaluation.search_value)
                reason = "search" if best.reason == "model" else best.reason
                results[col] = MoveEvaluation(col, best.p_x_wins, best.score, reason)

        if leaf_columns:
            preds, scores = self._predict_scores(encodings, root_player)
            for col, pred, score in zip(leaf_columns, preds, scores):
                results[col] = MoveEvaluation(col, float(pred), float(score))

        return [results[col] for col in columns]

    def evaluate_moves(self, board: Connect4Board) -> List[MoveEvaluation]:
        """Score every legal next move; best move is first after sorting by score."""
        player = board.current_player()
        legal = board.legal_columns()
        if not legal:
            raise ValueError("No legal moves")

        winning_cols, blocking_cols = self._forced_columns(board, player)
        candidate_cols = winning_cols or blocking_cols or legal
        evaluations = self._evaluate_positions(
            board, candidate_cols, player, player, self.depth
        )
        if blocking_cols and not winning_cols:
            evaluations = [
                MoveEvaluation(ev.col, ev.p_x_wins, ev.score, "block+" + ev.reason)
                if ev.reason in ("model", "search") else ev
                for ev in evaluations
            ]
        evaluations.sort(key=MoveEvaluation.search_value, reverse=True)
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
