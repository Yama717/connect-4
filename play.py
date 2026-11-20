"""Play Connect-4 against a trained position-evaluation AI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_player import AIPlayer
from board import Connect4Board

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "cnn": ROOT / "models" / "connect4_cnn.pt",
    "nn": ROOT / "models" / "connect4_nn.json",
    "linear": ROOT / "models" / "connect4_linear.joblib",
    "logistic": ROOT / "models" / "connect4_logistic.joblib",
}
MODEL_LABELS = {
    "cnn": "convolutional neural network (CNN)",
    "nn": "NumPy neural network",
    "linear": "linear regression",
    "logistic": "logistic regression",
}


def play(
    human_first: bool = True,
    show_ai_analysis: bool = False,
    model_type: str = "linear",
    depth: int = 1,
) -> None:
    board = Connect4Board()
    ai = AIPlayer(model_path=MODEL_PATHS[model_type], depth=depth)
    human = "X" if human_first else "O"
    ai_player = "O" if human_first else "X"

    print(f"Connect-4 — {MODEL_LABELS[model_type]} opponent (depth {depth})")
    print(f"You are {human}. AI is {ai_player}. Columns are numbered 1-7.\n")

    while not board.is_terminal():
        print(board.render())
        print()

        current = board.current_player()
        if current == human:
            while True:
                try:
                    raw = input("Your move (1-7), or 'q' to quit: ").strip()
                    if raw.lower() == "q":
                        print("Goodbye.")
                        return
                    col = int(raw) - 1
                    if col in board.legal_columns():
                        board.drop(col, human)
                        break
                    print("Column full or invalid. Try again.")
                except (ValueError, EOFError):
                    print("Enter a number 1-7.")
        else:
            col, evaluations = ai.choose_move_with_analysis(board)
            board.drop(col, ai_player)
            best = evaluations[0]
            if show_ai_analysis:
                print("AI move analysis (score from the AI player's perspective):")
                for rank, ev in enumerate(evaluations, start=1):
                    marker = " <--" if rank == 1 else ""
                    print(
                        f"  {rank}. col {ev.col + 1}  "
                        f"P(X wins)={ev.p_x_wins:.3f}  score={ev.score:.9f}  [{ev.reason}]{marker}"
                    )
            print(
                f"AI plays column {col + 1}  "
                f"P(X wins)={best.p_x_wins:.3f}  score={best.score:.9f}  [{best.reason}]\n"
            )

        winner = board.winner()
        if winner:
            print(board.render())
            if winner == human:
                print("You win!")
            else:
                print("AI wins!")
            return

    print(board.render())
    print("Draw.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect-4 vs a trained AI")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PATHS),
        default="linear",
        help="Model type (default: linear)",
    )
    parser.add_argument(
        "--ai-first",
        action="store_true",
        help="Let the AI play as X (first player)",
    )
    parser.add_argument(
        "--show-ai",
        action="store_true",
        help="Print all candidate columns and model scores when the AI moves",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of moves to look ahead, counting both players (default: 1)",
    )
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    try:
        play(
            human_first=not args.ai_first,
            show_ai_analysis=args.show_ai,
            model_type=args.model,
            depth=args.depth,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        if args.model == "cnn":
            print("Train the model first:  python train_cnn.py", file=sys.stderr)
        elif args.model != "nn":
            print(f"Train the model first:  python train.py --model {args.model}", file=sys.stderr)
        sys.exit(1)
    except ModuleNotFoundError as exc:
        if args.model != "cnn" or exc.name != "torch":
            raise
        sys.exit(f"The CNN requires PyTorch. Run: {sys.executable!r} -m pip install torch")


if __name__ == "__main__":
    main()
