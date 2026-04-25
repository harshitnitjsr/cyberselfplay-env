"""PFSP + PSRO meta-solver tests."""

from __future__ import annotations

import math

from train.pfsp import OpponentStats, pfsp_weight, sample_opponent
from train.psro_meta import normalize, replicator_update


def test_pfsp_weight_peaks_at_half():
    assert math.isclose(pfsp_weight(0.5), 0.25)
    assert pfsp_weight(0.0) == 0.0
    assert pfsp_weight(1.0) == 0.0
    assert pfsp_weight(0.5) > pfsp_weight(0.1)


def test_sample_opponent_handles_degenerate_pool():
    pool = [
        OpponentStats(name="zero", win_rate_vs_learner=0.0),
        OpponentStats(name="one", win_rate_vs_learner=1.0),
    ]
    chosen = sample_opponent(pool)
    assert chosen.name in {"zero", "one"}


def test_replicator_update_increases_high_payoff():
    payoffs = [1.0, 0.0]
    p = [0.5, 0.5]
    p_next = replicator_update(payoffs, p, eta=0.5)
    assert p_next[0] > p[0]
    assert math.isclose(sum(p_next), 1.0, abs_tol=1e-6)


def test_normalize_handles_zero_vector():
    out = normalize([0.0, 0.0, 0.0])
    assert math.isclose(sum(out), 1.0, abs_tol=1e-6)
    assert all(v > 0 for v in out)
