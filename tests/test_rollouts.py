"""Tests for the rollout collection helper used by training scripts."""

from __future__ import annotations

import importlib

import pytest


def test_collect_rollouts_short_episode_runs():
    mod = importlib.import_module("train.colab_trl_selfplay")
    rollouts, summaries = mod.collect_rollouts(num_episodes=2, max_steps=8, side="blue")
    assert isinstance(rollouts, list)
    assert isinstance(summaries, list)
    assert len(summaries) == 2
    if rollouts:
        first = rollouts[0]
        assert hasattr(first, "reward")
        assert hasattr(first, "prompt")
        assert hasattr(first, "response")


@pytest.mark.parametrize("side", ["blue", "red"])
def test_collect_rollouts_side_param(side: str):
    mod = importlib.import_module("train.colab_trl_selfplay")
    _, summaries = mod.collect_rollouts(num_episodes=1, max_steps=6, side=side)
    assert summaries
    assert "avg_learner_reward" in summaries[0]
