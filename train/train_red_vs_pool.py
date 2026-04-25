"""
Collect Red-side rollouts against a PFSP-sampled pool of Blue defenders.

A learnable Red policy is out of scope for the minimal scaffold (the simulator
provides a stable scripted Red), but this entry point produces Red-aligned
training data and updates Blue-pool win-rate estimates so the league loop
can use it for opponent sampling.
"""

from __future__ import annotations

from train._bootstrap import ensure_repo_root_on_path  # noqa: E402

ensure_repo_root_on_path()

import argparse
import json
import os
from statistics import mean
from typing import List

from train.colab_trl_selfplay import (
    collect_rollouts,
    save_artifacts,
)
from train.pfsp import OpponentStats, sample_opponent


def _seed_pool() -> List[OpponentStats]:
    return [
        OpponentStats(name="B-novice", win_rate_vs_learner=0.30, games=10),
        OpponentStats(name="B-mid", win_rate_vs_learner=0.50, games=10),
        OpponentStats(name="B-strong", win_rate_vs_learner=0.70, games=10),
    ]


def _update_winrate(opp: OpponentStats, summaries: list[dict]) -> None:
    if not summaries:
        return
    avg_r = mean(s.get("avg_learner_reward", 0.0) for s in summaries)
    red_won = 1.0 if avg_r > 0 else 0.0
    new_games = opp.games + len(summaries)
    blended = (
        (opp.win_rate_vs_learner * opp.games) + ((1.0 - red_won) * len(summaries))
    ) / max(1, new_games)
    opp.win_rate_vs_learner = float(min(1.0, max(0.0, blended)))
    opp.games = new_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Red-side data collection vs Blue pool.")
    parser.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "2")))
    parser.add_argument("--episodes", type=int, default=int(os.getenv("EPISODES", "8")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "50")))
    args = parser.parse_args()

    pool = _seed_pool()
    last_out_dir = "outputs/cyber-red-data"
    os.makedirs(last_out_dir, exist_ok=True)

    all_summaries: list[dict] = []
    for r in range(args.rounds):
        opp = sample_opponent(pool)
        print(
            f"[red-data round {r}] PFSP picked Blue={opp.name} "
            f"(w_vs_learner={opp.win_rate_vs_learner:.2f})"
        )
        rollouts, summaries = collect_rollouts(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            side="red",
        )
        _update_winrate(opp, summaries)
        save_artifacts(rollouts, summaries, last_out_dir)
        all_summaries.extend(summaries)

    print("[red-data] final pool state:")
    for o in pool:
        print(
            f"  {o.name}: win_rate_vs_learner={o.win_rate_vs_learner:.2f} games={o.games}"
        )

    with open(os.path.join("artifacts", "red_pool_state.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pool": [
                    {
                        "name": o.name,
                        "win_rate_vs_learner": o.win_rate_vs_learner,
                        "games": o.games,
                    }
                    for o in pool
                ],
                "rounds": args.rounds,
                "summaries": all_summaries,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )


if __name__ == "__main__":
    main()
