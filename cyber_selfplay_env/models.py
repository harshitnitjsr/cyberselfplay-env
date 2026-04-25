from typing import Any, Dict, List

from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class CyberAction(Action):
    # Defaults prevent OpenEnv web UI / empty payload crashes on first step.
    actor: str = Field(default="blue", description="'red' or 'blue'")
    tool_name: str = Field(default="query_siem", description="Action/tool identifier")
    target: str = Field(default="", description="Host/account/asset target")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    rationale: str = Field(default="", description="Optional rationale text")


class CyberObservation(Observation):
    actor: str = Field(default="", description="Actor receiving observation")
    public_state: Dict[str, Any] = Field(default_factory=dict, description="Visible world state")
    telemetry: List[Dict[str, Any]] = Field(default_factory=list, description="Noisy logs/events")
    incident_summary: Dict[str, Any] = Field(default_factory=dict, description="Incident stats")
    reward: float = Field(default=0.0, description="Reward from previous action")
    done: bool = Field(default=False, description="Episode completion")
