"""League evaluation scaffold with POSG metrics output."""

from itertools import product
from random import random

try:
    from .psro_meta import replicator_update
except ImportError:
    from psro_meta import replicator_update


def evaluate_pair(blue_name: str, red_name: str) -> dict:
    # Placeholder for actual rollout-based metrics from env episodes.
    return {
        "blue_reward": round(0.3 + random() * 0.5, 3),
        "exfil_rate": round(random() * 0.6, 3),
        "mttd": round(1.0 + random() * 8.0, 3),
        "mttr": round(2.0 + random() * 10.0, 3),
        "critical_asset_compromise_rate": round(random() * 0.7, 3),
    }


def main():
    blue_pool = ["B0", "B1", "B2"]
    red_pool = ["R0", "R1", "R2"]
    print("blue,red,blue_reward,exfil_rate,mttd,mttr,critical_asset_compromise_rate")
    payoff_row = []
    for b, r in product(blue_pool, red_pool):
        m = evaluate_pair(b, r)
        payoff_row.append(m["blue_reward"])
        print(
            f"{b},{r},{m['blue_reward']:.3f},{m['exfil_rate']:.3f},"
            f"{m['mttd']:.3f},{m['mttr']:.3f},{m['critical_asset_compromise_rate']:.3f}"
        )
    p0 = [1.0 / len(payoff_row)] * len(payoff_row)
    p1 = replicator_update(payoff_row, p0, eta=0.2)
    best_response = max(payoff_row)
    current_value = sum(pi * ui for pi, ui in zip(p1, payoff_row))
    exploitability_gap = best_response - current_value
    print(f"exploitability_gap_proxy,{exploitability_gap:.4f}")


if __name__ == "__main__":
    main()
