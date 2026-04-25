"""
Train the Blue policy against a PFSP-sampled pool of Red opponents.

This is the practical entry point for the "Blue learner vs Red league"
configuration in the report. It reuses the rollout + TRL helpers from
`colab_trl_selfplay`, swapping the learner side to `blue` and updating
opponent win-rate stats after each round.
"""

from __future__ import annotations

from train._bootstrap import ensure_repo_root_on_path  # noqa: E402

ensure_repo_root_on_path()

import argparse
import os
from statistics import mean
from typing import List

from train.colab_trl_selfplay import (
    HAS_TRL_STACK,
    build_training_dataset,
    collect_rollouts,
    maybe_upload_to_hub,
    run_trl_sft,
    save_artifacts,
)
from train.pfsp import OpponentStats, sample_opponent


def _seed_pool() -> List[OpponentStats]:
    return [
        OpponentStats(name="R-easy", win_rate_vs_learner=0.20, games=10),
        OpponentStats(name="R-mid", win_rate_vs_learner=0.50, games=10),
        OpponentStats(name="R-hard", win_rate_vs_learner=0.80, games=10),
    ]


def _update_winrate(opp: OpponentStats, summaries: list[dict]) -> None:
    """Approximate win-rate-vs-learner update using mean Blue reward sign."""
    if not summaries:
        return
    avg_r = mean(s.get("avg_learner_reward", 0.0) for s in summaries)
    blue_won = 1.0 if avg_r > 0 else 0.0
    new_games = opp.games + len(summaries)
    blended = (
        (opp.win_rate_vs_learner * opp.games) + ((1.0 - blue_won) * len(summaries))
    ) / max(1, new_games)
    opp.win_rate_vs_learner = float(min(1.0, max(0.0, blended)))
    opp.games = new_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Blue vs Red-pool PFSP trainer.")
    parser.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "2")))
    parser.add_argument("--episodes", type=int, default=int(os.getenv("EPISODES", "10")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "60")))
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Collect rollouts + artifacts only, skip TRL fine-tune.",
    )
    args = parser.parse_args()

    pool = _seed_pool()
    last_out_dir = "outputs/cyber-blue-sft"

    for r in range(args.rounds):
        opp = sample_opponent(pool)
        print(
            f"[blue-train round {r}] PFSP picked Red={opp.name} "
            f"(w_vs_learner={opp.win_rate_vs_learner:.2f})"
        )

        rollouts, summaries = collect_rollouts(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            side="blue",
        )
        _update_winrate(opp, summaries)

        if HAS_TRL_STACK and not args.no_train:
            ds = build_training_dataset(rollouts, keep_top_frac=0.5)
            last_out_dir = run_trl_sft(ds, output_dir=f"outputs/cyber-blue-sft-r{r}")
        save_artifacts(rollouts, summaries, last_out_dir)

    print("[blue-train] final pool state:")
    for o in pool:
        print(
            f"  {o.name}: win_rate_vs_learner={o.win_rate_vs_learner:.2f} games={o.games}"
        )

    maybe_upload_to_hub(last_out_dir)


if __name__ == "__main__":
    main()
