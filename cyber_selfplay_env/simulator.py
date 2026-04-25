import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple


class CyberSimulator:
    """Lightweight stochastic cyber range with hidden compromise state."""

    def __init__(self, scenario: Dict[str, Any]):
        self.scenario = scenario
        self.max_turns = scenario["max_turns"]
        self.reset()

    def reset(self) -> Dict[str, Any]:
        host_count = self.scenario["hosts"]
        instruction_count = int(self.scenario.get("instruction_count", 40))
        instructions = self._build_instructions(instruction_count)
        self.state: Dict[str, Any] = {
            "time_step": 0,
            "network_topology": {"zones": ["corp", "prod", "secure"]},
            "assets": {
                host: {"criticality": 1}
                for host in [f"host-{i:02d}" for i in range(host_count)]
            },
            "services": {f"host-{i:02d}": {"patch_level": random.uniform(0.2, 0.9)} for i in range(host_count)},
            "vulns": {f"host-{i:02d}": {"exploitability": random.uniform(0.2, 0.95)} for i in range(host_count)},
            "identities": {"disabled_accounts": [], "rotated_secrets": False},
            "detections": [],
            "incidents": {"foothold_hosts": [], "persistent_hosts": [], "exfil_prepared": False, "exfil_done": False},
            "business_impact": {"downtime": 0, "data_exfil_estimate_gb": 0.0},
            "high_value_hosts": list(self.scenario["high_value_hosts"]),
            "mission": {
                "instructions": instructions,
                "completed_ids": [],
                "violated_ids": [],
                "checkpoint_every": int(self.scenario.get("checkpoint_every", 10)),
            },
            "terminated": False,
            "winner": "",
        }
        return deepcopy(self.state)

    def _build_instructions(self, n: int) -> List[Dict[str, Any]]:
        catalog = [
            "triage_alerts",
            "isolate_host",
            "deploy_patch",
            "rotate_secrets",
            "run_forensics",
            "restore_backup",
            "harden_policy",
            "publish_ioc_blocklist",
        ]
        rows: List[Dict[str, Any]] = []
        for idx in range(n):
            required_tool = catalog[idx % len(catalog)]
            rows.append(
                {
                    "id": f"inst-{idx:03d}",
                    "required_tool": required_tool,
                    "required_target_prefix": "host-",
                    "priority": 1 + (idx % 5),
                    "done": False,
                }
            )
        return rows

    def _emit_detection(self, confidence: float, message: str) -> None:
        if random.random() < confidence:
            self.state["detections"].append(
                {
                    "type": "alert",
                    "confidence": round(confidence, 2),
                    "message": message,
                    "time_step": self.state["time_step"],
                }
            )

    def _resolve_red(self, tool_name: str, target: str) -> Dict[str, bool]:
        e = {k: False for k in [
            "foothold", "priv_esc", "lateral_high_value", "exfil_success", "detected",
            "foothold_removed", "campaign_disrupted", "incident_active", "plan_sabotaged"
        ]}
        incidents = self.state["incidents"]
        vulns = self.state["vulns"].get(target, {"exploitability": 0.4})

        if tool_name == "attempt_exploit":
            p = 0.2 + 0.8 * vulns["exploitability"]
            if random.random() < p and target not in incidents["foothold_hosts"]:
                incidents["foothold_hosts"].append(target)
                e["foothold"] = True
            self._emit_detection(0.4, f"Exploit attempt on {target}")
        elif tool_name == "pivot_host" and incidents["foothold_hosts"]:
            if target in self.state["high_value_hosts"] and random.random() < 0.55:
                incidents["foothold_hosts"].append(target)
                e["lateral_high_value"] = True
            self._emit_detection(0.35, f"Suspicious lateral movement to {target}")
        elif tool_name == "establish_persistence" and target in incidents["foothold_hosts"]:
            if random.random() < 0.7:
                incidents["persistent_hosts"].append(target)
                e["priv_esc"] = True
            self._emit_detection(0.25, f"Persistence artifact on {target}")
        elif tool_name == "prepare_exfiltration" and incidents["foothold_hosts"]:
            incidents["exfil_prepared"] = True
            self._emit_detection(0.2, "Bulk staging behavior")
        elif tool_name == "execute_exfiltration" and incidents["exfil_prepared"]:
            success = any(h in self.state["high_value_hosts"] for h in incidents["foothold_hosts"])
            if success and random.random() < 0.75:
                incidents["exfil_done"] = True
                self.state["business_impact"]["data_exfil_estimate_gb"] += random.uniform(1.0, 9.0)
                e["exfil_success"] = True
            self._emit_detection(0.6, "Large outbound transfer")
        elif tool_name == "cover_tracks":
            self._emit_detection(0.1, "Potential log tampering")
        elif tool_name == "sabotage_recovery_plan":
            mission = self.state["mission"]
            pending = [x for x in mission["instructions"] if not x["done"]]
            if pending:
                victim = random.choice(pending)
                mission["violated_ids"].append(victim["id"])
                e["plan_sabotaged"] = True
            self._emit_detection(0.2, "Recovery playbook mismatch observed")

        e["detected"] = len(self.state["detections"]) > 0
        e["incident_active"] = bool(incidents["foothold_hosts"] or incidents["persistent_hosts"] or incidents["exfil_prepared"])
        return e

    def _resolve_blue(self, tool_name: str, target: str, params: Dict[str, Any] | None = None) -> Dict[str, bool]:
        e = {k: False for k in [
            "true_positive_triage", "containment_success", "persistence_removed", "full_recovery",
            "exfil_success", "critical_downtime", "false_positive", "incident_active",
            "instruction_progress", "checkpoint_bonus", "instruction_violation"
        ]}
        incidents = self.state["incidents"]
        mission = self.state["mission"]

        if tool_name in {"query_siem", "triage_alerts"}:
            e["true_positive_triage"] = bool(self.state["detections"])
        elif tool_name == "isolate_host":
            if target in incidents["foothold_hosts"] or target in incidents["persistent_hosts"]:
                incidents["foothold_hosts"] = [h for h in incidents["foothold_hosts"] if h != target]
                incidents["persistent_hosts"] = [h for h in incidents["persistent_hosts"] if h != target]
                e["containment_success"] = True
            else:
                e["false_positive"] = True
                if target in self.state["high_value_hosts"]:
                    self.state["business_impact"]["downtime"] += 1
                    e["critical_downtime"] = True
        elif tool_name == "deploy_patch":
            if target in self.state["services"]:
                self.state["services"][target]["patch_level"] = min(1.0, self.state["services"][target]["patch_level"] + 0.4)
        elif tool_name == "rotate_secrets":
            self.state["identities"]["rotated_secrets"] = True
            incidents["exfil_prepared"] = False
            e["persistence_removed"] = True
        elif tool_name == "restore_backup":
            self.state["business_impact"]["downtime"] = max(0, self.state["business_impact"]["downtime"] - 1)
        elif tool_name == "execute_instruction":
            pending = [x for x in mission["instructions"] if not x["done"]]
            if pending:
                current = pending[0]
                # Requires matching tool in params to model long-horizon instruction following.
                requested_tool = ""
                if params and isinstance(params.get("required_tool"), str):
                    requested_tool = params["required_tool"]
                elif target:
                    requested_tool = target
                if requested_tool == current["required_tool"]:
                    current["done"] = True
                    mission["completed_ids"].append(current["id"])
                    e["instruction_progress"] = True
                else:
                    mission["violated_ids"].append(current["id"])
                    e["instruction_violation"] = True
        elif tool_name == "checkpoint_plan":
            if self.state["time_step"] % mission["checkpoint_every"] == 0:
                e["checkpoint_bonus"] = True
        elif tool_name == "reconcile_state":
            # Small helper for recovery in long episodes.
            if incidents["foothold_hosts"] and random.random() < 0.4:
                incidents["foothold_hosts"] = incidents["foothold_hosts"][:-1]
                e["containment_success"] = True

        e["exfil_success"] = incidents["exfil_done"]
        e["incident_active"] = bool(incidents["foothold_hosts"] or incidents["persistent_hosts"] or incidents["exfil_prepared"])
        if not e["incident_active"] and not incidents["exfil_done"]:
            e["full_recovery"] = True
        return e

    def step(
        self,
        actor: str,
        tool_name: str,
        target: str,
        params: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        self.state["time_step"] += 1

        if actor == "red":
            events = self._resolve_red(tool_name, target)
        else:
            events = self._resolve_blue(tool_name, target, params=params)

        if self.state["incidents"]["exfil_done"]:
            self.state["terminated"] = True
            self.state["winner"] = "red"
        elif self.state["time_step"] >= self.max_turns:
            self.state["terminated"] = True
            completion = self.instruction_completion_rate()
            # Blue only wins by timeout if it also followed enough instructions.
            self.state["winner"] = "blue" if completion >= 0.6 else "red"

        return deepcopy(self.state), events

    def visible_state(self, actor: str) -> Dict[str, Any]:
        # Partial observability: each side sees different details.
        if actor == "blue":
            return {
                "time_step": self.state["time_step"],
                "detections": self.state["detections"][-5:],
                "business_impact": self.state["business_impact"],
                "known_incident_count": len(self.state["incidents"]["foothold_hosts"]) + len(self.state["incidents"]["persistent_hosts"]),
                "instruction_progress": {
                    "completed": len(self.state["mission"]["completed_ids"]),
                    "violated": len(self.state["mission"]["violated_ids"]),
                    "total": len(self.state["mission"]["instructions"]),
                },
            }
        return {
            "time_step": self.state["time_step"],
            "known_targets": list(self.state["vulns"].keys())[: min(8, len(self.state["vulns"]))],
            "high_value_guess_count": len(self.state["high_value_hosts"]),
            "detection_pressure": len(self.state["detections"]),
        }

    def recent_telemetry(self, actor: str) -> List[Dict[str, Any]]:
        if actor == "blue":
            return self.state["detections"][-5:]
        # Red has only weak indirect signal.
        return [{"type": "risk", "message": f"Detection pressure={len(self.state['detections'])}"}]

    def instruction_completion_rate(self) -> float:
        total = max(1, len(self.state["mission"]["instructions"]))
        return len(self.state["mission"]["completed_ids"]) / total
