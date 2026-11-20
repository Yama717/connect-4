# Connect 4 AI

Play Connect 4 in the terminal against one of four trained models. Each model evaluates board positions. You choose how many moves the AI looks ahead before it drops a piece. All four sets of weights are included in `models/`, so you can play as soon as the dependencies are installed.

## Play your first game

Open a terminal in the project folder (the folder containing `play.py`). In a fresh GitHub Codespace or on macOS or Linux, set up the Python environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the CNN, also install PyTorch. In a Linux Codespace, the CPU build is sufficient:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On a Mac, use `python -m pip install torch` instead. Start a game with:

```bash
python play.py --model cnn --depth 2 --show-ai
```

You play **X** and move first. Type a column number from **1** (left) to **7** (right) and press Enter. Your piece drops into the lowest open space in that column. Connect four pieces horizontally, vertically, or diagonally to win. Type `q` at the move prompt to quit.

To play with only the base dependencies, run `python play.py`. This starts the linear model at depth 1. Add `--ai-first` to any game command to play second as O.

## Choose an opponent

| `--model` | How it reads the board | What it predicts |
| --- | --- | --- |
| `linear` (default) | Assigns a weight to each of the 42 cells | A clipped estimate of X's win probability |
| `logistic` | Applies a sigmoid to a weighted sum of the 42 cells | X's win probability |
| `nn` | Passes the 42 cells through two NumPy hidden layers with 128 and 64 units | X win, O win, or draw |
| `cnn` | Applies PyTorch convolutional filters to three 6 × 7 grids showing X, O, and empty cells | X win, O win, or draw |

## How search changes a move

`--depth` counts individual moves by either player. At depth 1, the AI tries its candidate moves and scores the resulting boards. At depth 2, it also tries your legal replies and assumes you choose the worst one for the AI. For example, dropping a piece might give you the support you need to complete a diagonal on your next turn. Depth 2 can see that loss before the AI plays the piece.

At greater depths the same minimax search continues, with the players taking turns. Each extra move can introduce up to seven replies, so deeper searches take longer. You can change depth with the same saved weights.

With `--show-ai`, the arrow marks the chosen column. The AI checks for an immediate win or a required block first, so some turns list fewer than seven candidates.

- `P(X wins)` always refers to X. `score` is from the AI's perspective. For the two neural networks, the score at an evaluated position is the AI's predicted win probability plus half the draw probability.
- `[model]` marks a direct model estimate. `[search]` marks an estimate selected after looking ahead. At depth 2, this is the evaluation after the opponent's strongest predicted reply.
- `[win]` or `[loss]` means the search found a forced result within its depth. A model estimate displayed as `1.000` can still be wrong.

## Train a model

Training starts from fresh weights and replaces the corresponding file in `models/` when it finishes. To keep an existing CNN for comparison, copy `models/connect4_cnn.pt` to a different filename first.

| Model | Training command |
| --- | --- |
| CNN | `python train_cnn.py --epochs 25` |
| Linear regression | `python train.py --model linear --epochs 100` |
| Logistic regression | `python train.py --model logistic --epochs 100` |

The CNN script saves the weights from the epoch with the highest validation accuracy. Its `val_acc` measures how often it predicts the labeled outcome correctly on held-out positions. To compare playing strength, play the models at the same search depth and alternate which one goes first.

The included dataset contains **67,557 positions**, each with four X pieces and four O pieces. During a full game the AI also encounters earlier and later positions, where its estimates can be less reliable. [UCI Connect-4 dataset](https://archive.ics.uci.edu/dataset/26/connect+4)
