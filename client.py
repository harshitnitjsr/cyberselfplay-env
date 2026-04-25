# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CyberSelfPlay Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from cyber_selfplay_env.models import CyberAction, CyberObservation


class CyberSelfPlayEnv(EnvClient[CyberAction, CyberObservation, State]):
    """
    Client for the CyberSelfPlay environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: CyberAction) -> Dict:
        """
        Convert CyberAction to JSON payload for step message.

        Args:
            action: CyberAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "actor": action.actor,
            "tool_name": action.tool_name,
            "target": action.target,
            "params": action.params,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CyberObservation]:
        """
        Parse server response into StepResult[CyberObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CyberObservation
        """
        obs_data = payload.get("observation", {})
        observation = CyberObservation(
            actor=obs_data.get("actor", ""),
            public_state=obs_data.get("public_state", {}),
            telemetry=obs_data.get("telemetry", []),
            incident_summary=obs_data.get("incident_summary", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
