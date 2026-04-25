"""Minimal red training skeleton against blue league pool."""

from dataclasses import dataclass
from typing import List

try:
    from .pfsp import OpponentStats, sample_opponent
except ImportError:
    from pfsp import OpponentStats, sample_opponent

@dataclass
class PolicyCheckpoint(OpponentStats):
    pass


def main():
    blue_pool: List[PolicyCheckpoint] = [
        PolicyCheckpoint(name="B0", win_rate_vs_learner=0.3, games=50),
        PolicyCheckpoint(name="B1", win_rate_vs_learner=0.5, games=50),
        PolicyCheckpoint(name="B2", win_rate_vs_learner=0.7, games=50),
    ]
    red_candidate = PolicyCheckpoint(name="R_new", win_rate_vs_learner=0.0, games=0)
    for _ in range(10):
        opponent = sample_opponent(blue_pool)
        # Placeholder: run self-play episodes and update red policy via PPO/GRPO.
        red_candidate.win_rate_vs_learner += 0.01
    print(
        f"Trained {red_candidate.name} vs PFSP pool. "
        f"proxy_score={red_candidate.win_rate_vs_learner:.2f}"
    )


if __name__ == "__main__":
    main()
