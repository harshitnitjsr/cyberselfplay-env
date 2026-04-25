from dataclasses import dataclass


@dataclass
class CurriculumState:
    scenario_name: str = "small"
    rolling_blue_win_rate: float = 0.0
    episodes: int = 0


class CurriculumManager:
    def __init__(self) -> None:
        self.state = CurriculumState()
        self._alpha = 0.2

    def record_episode(self, blue_win: bool) -> None:
        self.state.episodes += 1
        x = 1.0 if blue_win else 0.0
        self.state.rolling_blue_win_rate = (
            (1.0 - self._alpha) * self.state.rolling_blue_win_rate + self._alpha * x
        )
        self._escalate_if_needed()

    def _escalate_if_needed(self) -> None:
        w = self.state.rolling_blue_win_rate
        if self.state.scenario_name == "small" and w >= 0.55 and self.state.episodes >= 10:
            self.state.scenario_name = "medium"
        elif self.state.scenario_name == "medium" and w >= 0.60 and self.state.episodes >= 20:
            self.state.scenario_name = "large"
