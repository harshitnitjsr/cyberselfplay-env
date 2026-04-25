from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .curriculum import CurriculumManager
from .metrics import MetricsTracker
from .models import CyberAction, CyberObservation
from .rubrics import blue_reward, red_reward
from .scenarios import build_scenario
from .simulator import CyberSimulator
from .tools_blue import validate_blue_tool
from .tools_red import validate_red_tool


class CyberSelfPlayEnvironment(Environment):
    """OpenEnv-compatible cyber self-play environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.curriculum = CurriculumManager()
        self.sim = CyberSimulator(build_scenario(self.curriculum.state.scenario_name))
        self.metrics = MetricsTracker()
        self.last_actor = "blue"
        self.last_reward = 0.0

    def reset(self) -> CyberObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.sim = CyberSimulator(build_scenario(self.curriculum.state.scenario_name))
        self.sim.reset()
        self.metrics.reset()
        self.last_actor = "blue"
        self.last_reward = 0.0
        return CyberObservation(
            actor="blue",
            public_state=self.sim.visible_state("blue"),
            telemetry=self.sim.recent_telemetry("blue"),
            incident_summary={"winner": "", "terminated": False},
            reward=0.0,
            done=False,
            metadata={
                "note": "Blue starts with partial telemetry. Alternate actors each step.",
                "scenario": self.curriculum.state.scenario_name,
            },
        )

    def step(self, action: CyberAction) -> CyberObservation:  # type: ignore[override]
        self._state.step_count += 1
        actor = action.actor.lower()
        tool_name = action.tool_name
        target = action.target or "host-00"

        if actor not in {"red", "blue"}:
            raise ValueError("actor must be 'red' or 'blue'")
        if actor == "red" and not validate_red_tool(tool_name):
            raise ValueError(f"invalid red tool: {tool_name}")
        if actor == "blue" and not validate_blue_tool(tool_name):
            raise ValueError(f"invalid blue tool: {tool_name}")

        state, events = self.sim.step(
            actor=actor,
            tool_name=tool_name,
            target=target,
            params=action.params,
        )
        self.metrics.update(actor=actor, state=state, events=events)

        if actor == "red":
            rb = red_reward(events)
        else:
            rb = blue_reward(
                events,
                instruction_completion_rate=self.sim.instruction_completion_rate(),
            )

        self.last_actor = actor
        self.last_reward = rb.total
        if state["terminated"]:
            self.curriculum.record_episode(blue_win=state["winner"] == "blue")

        return CyberObservation(
            actor=actor,
            public_state=self.sim.visible_state(actor),
            telemetry=self.sim.recent_telemetry(actor),
            incident_summary={
                "winner": state["winner"],
                "terminated": state["terminated"],
                "exfil_done": state["incidents"]["exfil_done"],
                "time_step": state["time_step"],
            },
            reward=rb.total,
            done=state["terminated"],
            metadata={
                "reward_components": rb.components,
                "events": events,
                "posg_metrics": self.metrics.snapshot(),
                "curriculum": {
                    "scenario": self.curriculum.state.scenario_name,
                    "rolling_blue_win_rate": self.curriculum.state.rolling_blue_win_rate,
                    "episodes": self.curriculum.state.episodes,
                },
            },
        )

    @property
    def state(self) -> State:
        return self._state
