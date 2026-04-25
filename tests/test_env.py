"""Smoke tests for the CyberSelfPlay environment and reward layer."""

from __future__ import annotations

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction


def test_reset_returns_blue_observation():
    env = CyberSelfPlayEnvironment()
    obs = env.reset()
    assert obs.actor == "blue"
    assert obs.done is False
    assert "scenario" in obs.metadata


def test_step_red_then_blue_runs_metrics():
    env = CyberSelfPlayEnvironment()
    env.reset()
    obs = env.step(CyberAction(actor="red", tool_name="recon_network", target="host-00"))
    assert obs.actor == "red"
    obs = env.step(CyberAction(actor="blue", tool_name="query_siem", target="host-00"))
    assert obs.actor == "blue"
    assert "posg_metrics" in obs.metadata


def test_invalid_actor_returns_error_observation():
    env = CyberSelfPlayEnvironment()
    env.reset()
    obs = env.step(CyberAction(actor="green", tool_name="query_siem"))
    assert obs.metadata.get("error", "").startswith("invalid actor")
    assert obs.done is False


def test_invalid_blue_tool_does_not_raise():
    env = CyberSelfPlayEnvironment()
    env.reset()
    obs = env.step(CyberAction(actor="blue", tool_name="nuke_everything"))
    assert "invalid blue tool" in obs.metadata.get("error", "")


def test_long_episode_terminates_within_max_turns():
    env = CyberSelfPlayEnvironment()
    env.reset()
    for i in range(500):
        actor = "red" if i % 2 == 0 else "blue"
        tool = "recon_network" if actor == "red" else "query_siem"
        obs = env.step(CyberAction(actor=actor, tool_name=tool, target="host-00"))
        if obs.done:
            break
    assert obs.done is True
    assert obs.incident_summary["winner"] in {"red", "blue"}


def test_default_action_payload_does_not_crash():
    """Empty payloads from the OpenEnv web UI must not crash the env."""
    env = CyberSelfPlayEnvironment()
    env.reset()
    obs = env.step(CyberAction())
    assert obs is not None
