"""Minimal PSRO-style restricted-game meta solver utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MetaStrategy:
    probs: List[float]


def normalize(xs: List[float]) -> List[float]:
    s = sum(max(0.0, x) for x in xs)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [max(0.0, x) / s for x in xs]


def replicator_update(payoff_row: List[float], current: List[float], eta: float = 0.2) -> List[float]:
    """
    One-step discrete replicator-style update:
      p_i <- p_i * (1 + eta*(u_i - u_bar))
    """
    if len(payoff_row) != len(current):
        raise ValueError("payoff and current lengths differ")
    u_bar = sum(p * u for p, u in zip(current, payoff_row))
    nxt = [p * (1.0 + eta * (u - u_bar)) for p, u in zip(current, payoff_row)]
    return normalize(nxt)
