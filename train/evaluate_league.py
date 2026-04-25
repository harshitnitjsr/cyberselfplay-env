"""
League evaluation: produce POSG metrics + an exploitability-gap proxy
across a (Blue, Red) pool grid.
"""

from __future__ import annotations

from train._bootstrap import ensure_repo_root_on_path  # noqa: E402

ensure_repo_root_on_path()

import argparse
import csv
import json
import os
from itertools import product
from statistics import mean

from train.colab_trl_selfplay import collect_rollouts
from train.psro_meta import replicator_update


def evaluate_pair(blue_name: str, red_name: str, episodes: int = 5) -> dict:
    """Run real rollouts and compute aggregate POSG metrics."""
    rollouts, summaries = collect_rollouts(num_episodes=episodes, max_steps=50, side="blue")
    if not summaries:
        return {
            "blue_reward": 0.0,
            "exfil_rate": 0.0,
            "mttd": -1.0,
            "mttr": -1.0,
            "critical_asset_compromise_rate": 0.0,
        }
    return {
        "blue_reward": float(mean(s["avg_learner_reward"] for s in summaries)),
        "exfil_rate": float(mean(s["final_exfil_rate"] for s in summaries)),
        "mttd": float(mean(s["final_mttd"] for s in summaries)),
        "mttr": float(mean(s["final_mttr"] for s in summaries)),
        "critical_asset_compromise_rate": float(
            mean(1.0 if s["final_exfil_rate"] > 0 else 0.0 for s in summaries)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate league grid + exploitability proxy.")
    parser.add_argument("--episodes", type=int, default=int(os.getenv("EPISODES", "3")))
    parser.add_argument(
        "--out", type=str, default="artifacts/league_eval.csv", help="Output CSV path."
    )
    args = parser.parse_args()

    blue_pool = ["B0", "B1", "B2"]
    red_pool = ["R0", "R1", "R2"]

    rows = []
    payoff_row: list[float] = []
    for b, r in product(blue_pool, red_pool):
        m = evaluate_pair(b, r, episodes=args.episodes)
        m.update({"blue": b, "red": r})
        rows.append(m)
        payoff_row.append(m["blue_reward"])
        print(
            f"{b} vs {r}: reward={m['blue_reward']:.3f} exfil={m['exfil_rate']:.3f} "
            f"mttd={m['mttd']:.3f} mttr={m['mttr']:.3f}"
        )

    os.makedirs("artifacts", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "blue",
                "red",
                "blue_reward",
                "exfil_rate",
                "mttd",
                "mttr",
                "critical_asset_compromise_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    p0 = [1.0 / len(payoff_row)] * len(payoff_row)
    p1 = replicator_update(payoff_row, p0, eta=0.2)
    best_response = max(payoff_row)
    current_value = sum(pi * ui for pi, ui in zip(p1, payoff_row))
    exploitability_gap = best_response - current_value

    summary = {
        "exploitability_gap_proxy": exploitability_gap,
        "best_response_value": best_response,
        "meta_strategy_value": current_value,
        "meta_strategy": p1,
    }
    with open("artifacts/league_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"[exploitability_gap_proxy] {exploitability_gap:.4f}")


if __name__ == "__main__":
    main()
