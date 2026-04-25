from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardBreakdown:
    total: float
    components: Dict[str, float]


def red_reward(events: Dict[str, bool], turn_penalty: float = 0.1) -> RewardBreakdown:
    c = {
        "foothold": 2.0 if events.get("foothold") else 0.0,
        "priv_esc": 3.0 if events.get("priv_esc") else 0.0,
        "lateral_high_value": 2.0 if events.get("lateral_high_value") else 0.0,
        "exfil_success": 8.0 if events.get("exfil_success") else 0.0,
        "detected": -2.0 if events.get("detected") else 0.0,
        "foothold_removed": -4.0 if events.get("foothold_removed") else 0.0,
        "campaign_disrupted": -6.0 if events.get("campaign_disrupted") else 0.0,
        "plan_sabotage": 1.5 if events.get("plan_sabotaged") else 0.0,
        "time_cost": -turn_penalty,
    }
    return RewardBreakdown(total=sum(c.values()), components=c)


def blue_reward(
    events: Dict[str, bool],
    unresolved_penalty: float = 0.1,
    instruction_completion_rate: float = 0.0,
) -> RewardBreakdown:
    c = {
        "true_positive_triage": 2.0 if events.get("true_positive_triage") else 0.0,
        "containment_success": 3.0 if events.get("containment_success") else 0.0,
        "persistence_removed": 4.0 if events.get("persistence_removed") else 0.0,
        "full_recovery": 6.0 if events.get("full_recovery") else 0.0,
        "exfil_loss": -8.0 if events.get("exfil_success") else 0.0,
        "critical_downtime": -3.0 if events.get("critical_downtime") else 0.0,
        "false_positive": -2.0 if events.get("false_positive") else 0.0,
        "unresolved_cost": -unresolved_penalty if events.get("incident_active") else 0.0,
        "instruction_progress": 1.2 if events.get("instruction_progress") else 0.0,
        "checkpoint_bonus": 2.0 if events.get("checkpoint_bonus") else 0.0,
        "instruction_violation": -1.0 if events.get("instruction_violation") else 0.0,
        "completion_shaping": 2.0 * instruction_completion_rate,
    }
    return RewardBreakdown(total=sum(c.values()), components=c)
