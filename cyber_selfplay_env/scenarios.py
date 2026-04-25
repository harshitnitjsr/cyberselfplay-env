from typing import Any, Dict


def build_scenario(name: str = "small") -> Dict[str, Any]:
    presets = {
        "small": {"hosts": 6, "high_value_hosts": ["db-01"], "max_turns": 60, "instruction_count": 40, "checkpoint_every": 8},
        "medium": {"hosts": 12, "high_value_hosts": ["db-01", "dc-01"], "max_turns": 100, "instruction_count": 120, "checkpoint_every": 12},
        "large": {"hosts": 24, "high_value_hosts": ["db-01", "dc-01", "payments-01"], "max_turns": 180, "instruction_count": 300, "checkpoint_every": 20},
    }
    cfg = presets.get(name, presets["small"]).copy()
    cfg["name"] = name if name in presets else "small"
    return cfg
