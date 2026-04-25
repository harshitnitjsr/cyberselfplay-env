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
    red_pool: List[PolicyCheckpoint] = [
        PolicyCheckpoint(name="R0", win_rate_vs_learner=0.2, games=50),
        PolicyCheckpoint(name="R1", win_rate_vs_learner=0.5, games=50),
        PolicyCheckpoint(name="R2", win_rate_vs_learner=0.8, games=50),
    ]
    blue_candidate = PolicyCheckpoint(name="B_new", win_rate_vs_learner=0.0, games=0)
    for _ in range(10):
        opponent = sample_opponent(red_pool)
        # Placeholder: run episodes and collect trajectories against opponent.
        # Plug trajectory batches into TRL PPO/GRPO trainer here.
        blue_candidate.win_rate_vs_learner += 0.01
    print(
        f"Trained {blue_candidate.name} vs PFSP pool. "
        f"proxy_win_rate={blue_candidate.win_rate_vs_learner:.2f}"
    )


if __name__ == "__main__":
    main()
