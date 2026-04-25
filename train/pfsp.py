"""Prioritized Fictitious Self-Play (PFSP) utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class OpponentStats:
    name: str
    win_rate_vs_learner: float
    games: int = 0


def pfsp_weight(win_rate: float) -> float:
    """
    PFSP weighting function f(w)=w(1-w).
    Peaks at 0.5 and deprioritizes trivial/hopeless opponents.
    """
    w = min(1.0, max(0.0, win_rate))
    return w * (1.0 - w)


def sample_opponent(opponents: Iterable[OpponentStats]) -> OpponentStats:
    pool: List[OpponentStats] = list(opponents)
    if not pool:
        raise ValueError("opponent pool is empty")

    weights = [pfsp_weight(o.win_rate_vs_learner) for o in pool]
    total = sum(weights)
    if total <= 0:
        return random.choice(pool)
    return random.choices(pool, weights=weights, k=1)[0]
