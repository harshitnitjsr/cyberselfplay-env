"""
Path bootstrap shared by every training script.

Allows scripts to be invoked as either:
  python train/colab_trl_selfplay.py
  python -m train.colab_trl_selfplay

without relative-import fragility.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    """Insert the repo root (parent of the `train` folder) onto sys.path."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
