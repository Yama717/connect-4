"""Connect-4 board with UCI dataset encoding (x=first player, o=second)."""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np

ROWS = 6
COLS = 7
# Internal grid: row 0 = top, row ROWS-1 = bottom (gravity fills upward).
# UCI feature order: a1..a6 per column, bottom (1) to top (6).
UCI_COLUMNS = "abcdefg"


def _uci_row_indices() -> range:
    """Row indices in UCI order: bottom row first, then upward."""
    return range(ROWS - 1, -1, -1)


class Connect4Board:
    """6x7 board. Cells are '.', 'X', or 'O'. X is always the first player (UCI x)."""

    def __init__(self, grid: Optional[List[List[str]]] = None) -> None:
        if grid is None:
            self.grid = [["."] * COLS for _ in range(ROWS)]
        else:
            self.grid = deepcopy(grid)

    def copy(self) -> Connect4Board:
        return Connect4Board(self.grid)

    def piece_count(self) -> int:
        return sum(1 for r in range(ROWS) for c in range(COLS) if self.grid[r][c] != ".")

    def current_player(self) -> str:
        """Player to move: X on even ply count, O on odd."""
        return "X" if self.piece_count() % 2 == 0 else "O"

    def legal_columns(self) -> List[int]:
        return [c for c in range(COLS) if self.grid[0][c] == "."]

    def drop(self, col: int, player: str) -> bool:
        if col < 0 or col >= COLS or player not in ("X", "O"):
            return False
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] == ".":
                self.grid[row][col] = player
                return True
        return False

    def undo(self, col: int) -> None:
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] != ".":
                self.grid[row][col] = "."
                return

    def winner(self) -> Optional[str]:
        """Return 'X', 'O', or None."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.grid[r][c]
                if piece == ".":
                    continue
                for dr, dc in directions:
                    if all(
                        0 <= r + dr * k < ROWS
                        and 0 <= c + dc * k < COLS
                        and self.grid[r + dr * k][c + dc * k] == piece
                        for k in range(4)
                    ):
                        return piece
        return None

    def is_full(self) -> bool:
        return all(self.grid[0][c] != "." for c in range(COLS))

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_full()

    def to_uci_features(self) -> List[str]:
        """42 cell labels in UCI order: {x, o, b}, bottom row to top per column."""
        features: List[str] = []
        for col in range(COLS):
            for row in _uci_row_indices():
                cell = self.grid[row][col]
                if cell == "X":
                    features.append("x")
                elif cell == "O":
                    features.append("o")
                else:
                    features.append("b")
        return features

    def encode(self) -> List[float]:
        """Numeric encoding for the MLP: x=1, o=-1, b=0."""
        mapping = {"x": 1.0, "o": -1.0, "b": 0.0}
        return [mapping[c] for c in self.to_uci_features()]

    def encode_planes(self) -> np.ndarray:
        """3-channel board tensor (3, 6, 7): X plane, O plane, empty plane."""
        planes = np.zeros((3, ROWS, COLS), dtype=np.float32)
        for row in range(ROWS):
            for col in range(COLS):
                cell = self.grid[row][col]
                if cell == "X":
                    planes[0, row, col] = 1.0
                elif cell == "O":
                    planes[1, row, col] = 1.0
                else:
                    planes[2, row, col] = 1.0
        return planes

    @staticmethod
    def from_uci_row(cells: List[str]) -> Connect4Board:
        board = Connect4Board()
        idx = 0
        for col in range(COLS):
            for row in _uci_row_indices():
                ch = cells[idx]
                idx += 1
                if ch == "x":
                    board.grid[row][col] = "X"
                elif ch == "o":
                    board.grid[row][col] = "O"
        return board

    def render(self) -> str:
        lines = []
        for row in range(ROWS):
            lines.append("|" + "|".join(f" {self.grid[row][c]} " for c in range(COLS)) + "|")
        lines.append("+" + "---+" * COLS)
        lines.append("  " + "   ".join(str(c + 1) for c in range(COLS)))
        return "\n".join(lines)
