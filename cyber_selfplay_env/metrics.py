"""Episode metrics tracker for POSG-style cyber evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EpisodeMetrics:
    mttd: Optional[float]
    mttr: Optional[float]
    exfil_success_rate: float
    critical_asset_compromise_rate: float
    false_positive_disruption_cost: float
    steps: int
    winner: str


class MetricsTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.first_compromise_step: Optional[int] = None
        self.first_detection_step: Optional[int] = None
        self.recovery_step: Optional[int] = None
        self.exfil_done = False
        self.false_positive_disruption_cost = 0.0
        self.critical_asset_compromised = False
        self.winner = ""
        self.total_instructions = 0
        self.completed_instructions = 0
        self.violated_instructions = 0

    def update(
        self,
        *,
        actor: str,
        state: Dict[str, Any],
        events: Dict[str, bool],
    ) -> None:
        self.steps = int(state["time_step"])
        incidents = state["incidents"]

        compromised_hosts = set(incidents["foothold_hosts"]) | set(incidents["persistent_hosts"])
        if compromised_hosts and self.first_compromise_step is None:
            self.first_compromise_step = self.steps

        if actor == "blue" and state["detections"] and self.first_detection_step is None:
            self.first_detection_step = self.steps

        if actor == "blue" and self.first_compromise_step is not None:
            recovered = not compromised_hosts and not incidents["exfil_prepared"] and not incidents["exfil_done"]
            if recovered and self.recovery_step is None:
                self.recovery_step = self.steps

        if incidents["exfil_done"]:
            self.exfil_done = True

        if events.get("critical_downtime"):
            self.false_positive_disruption_cost += 1.0
        if events.get("false_positive"):
            self.false_positive_disruption_cost += 0.5

        if any(h in state["high_value_hosts"] for h in compromised_hosts):
            self.critical_asset_compromised = True

        mission = state.get("mission", {})
        self.total_instructions = len(mission.get("instructions", []))
        self.completed_instructions = len(mission.get("completed_ids", []))
        self.violated_instructions = len(mission.get("violated_ids", []))

        self.winner = state.get("winner", self.winner)

    def snapshot(self) -> Dict[str, Any]:
        mttd = None
        if self.first_compromise_step is not None and self.first_detection_step is not None:
            mttd = float(max(0, self.first_detection_step - self.first_compromise_step))

        mttr = None
        if self.first_detection_step is not None and self.recovery_step is not None:
            mttr = float(max(0, self.recovery_step - self.first_detection_step))

        return {
            "mttd": mttd,
            "mttr": mttr,
            "exfil_success_rate": 1.0 if self.exfil_done else 0.0,
            "critical_asset_compromise_rate": 1.0 if self.critical_asset_compromised else 0.0,
            "false_positive_disruption_cost": self.false_positive_disruption_cost,
            "instruction_completion_rate": (
                float(self.completed_instructions) / float(max(1, self.total_instructions))
            ),
            "instruction_violation_rate": (
                float(self.violated_instructions) / float(max(1, self.total_instructions))
            ),
            "steps": self.steps,
            "winner": self.winner,
        }
