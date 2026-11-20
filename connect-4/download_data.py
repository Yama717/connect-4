"""Download the Connect-4 dataset.

Sources:
  - https://www.kaggle.com/datasets/tbrewer/connect-4/data
  - https://archive.ics.uci.edu/dataset/26/connect+4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "connect-4.data"
COMPRESSED = DATA_DIR / "connect-4.data.Z"
KAGGLE_DATASET = "tbrewer/connect-4"
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z"
)


def _download_uci() -> Path:
    print(f"Downloading from UCI: {UCI_DATA_URL} ...")
    response = requests.get(UCI_DATA_URL, timeout=120)
    response.raise_for_status()
    COMPRESSED.write_bytes(response.content)
    print(f"Saved {COMPRESSED} ({len(response.content)} bytes)")

    print("Decompressing ...")
    result = subprocess.run(
        ["uncompress", "-f", str(COMPRESSED)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uncompress failed: {result.stderr}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Expected {DATA_FILE} after decompression")
    return DATA_FILE


def _download_kaggle() -> Path:
    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET} ...")
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(DATA_DIR),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Kaggle download failed. Configure credentials at ~/.kaggle/kaggle.json\n"
            f"Details: {message}"
        )

    if not DATA_FILE.exists():
        candidates = sorted(DATA_DIR.glob("*.data"))
        if len(candidates) == 1:
            candidates[0].rename(DATA_FILE)
        else:
            raise FileNotFoundError(
                f"Expected {DATA_FILE} after download. Found: {[p.name for p in candidates]}"
            )
    return DATA_FILE


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not force:
        print(f"Dataset already present: {DATA_FILE}")
        return DATA_FILE

    try:
        path = _download_kaggle()
    except Exception as kaggle_error:
        print(f"Kaggle download unavailable ({kaggle_error}). Trying UCI ...")
        path = _download_uci()

    print(f"Ready: {path} ({path.stat().st_size} bytes)")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Connect-4 dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    try:
        download(force=args.force)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
