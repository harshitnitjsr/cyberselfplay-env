"""
Colab-ready HF TRL training loop for CyberSelfPlay.

This script:
1) collects online rollouts from CyberSelfPlayEnvironment,
2) keeps higher-reward Blue actions as training targets,
3) runs TRL SFT fine-tuning with LoRA.

Usage in Colab:
  !pip install -q "openenv-core[core]" "trl>=0.10.0" "transformers>=4.40.0" "datasets>=2.19.0" "accelerate>=0.30.0" "peft>=0.11.0"
  !python train/colab_trl_selfplay.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

try:
    from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
    from cyber_selfplay_env.models import CyberAction
except Exception:
    # fallback for package-style execution
    from ..cyber_selfplay_env.environment import CyberSelfPlayEnvironment
    from ..cyber_selfplay_env.models import CyberAction


BLUE_TOOLS = [
    "query_siem",
    "triage_alerts",
    "isolate_host",
    "rotate_secrets",
    "deploy_patch",
    "execute_instruction",
    "checkpoint_plan",
    "reconcile_state",
]
RED_TOOLS = [
    "recon_network",
    "attempt_exploit",
    "pivot_host",
    "establish_persistence",
    "prepare_exfiltration",
    "execute_exfiltration",
    "cover_tracks",
]


@dataclass
class RolloutRecord:
    prompt: str
    response: str
    reward: float
    metadata: Dict[str, object]


def obs_to_prompt(obs: Dict[str, object]) -> str:
    return (
        "You are the Blue defender in a cyber self-play environment.\n"
        "Return ONE JSON object with keys: actor, tool_name, target, params, rationale.\n"
        "actor must be 'blue'.\n\n"
        f"Observation:\n{json.dumps(obs, ensure_ascii=True)}\n\n"
        "Respond with action JSON only."
    )


def choose_red_action(step_idx: int) -> CyberAction:
    tool = RED_TOOLS[step_idx % len(RED_TOOLS)]
    return CyberAction(actor="red", tool_name=tool, target=f"host-{step_idx % 6:02d}", params={})


def heuristic_blue_action(obs_public_state: Dict[str, object]) -> CyberAction:
    # Simple baseline policy for data collection.
    known_incidents = int(obs_public_state.get("known_incident_count", 0))
    if known_incidents > 0:
        return CyberAction(actor="blue", tool_name="isolate_host", target="host-00", params={})

    instr = obs_public_state.get("instruction_progress", {})
    if isinstance(instr, dict) and instr.get("completed", 0) < instr.get("total", 1):
        # Required tool can be supplied via params when executing instructions.
        required_tool = random.choice(["triage_alerts", "deploy_patch", "rotate_secrets"])
        return CyberAction(
            actor="blue",
            tool_name="execute_instruction",
            target="",
            params={"required_tool": required_tool},
        )

    return CyberAction(actor="blue", tool_name=random.choice(BLUE_TOOLS), target="host-00", params={})


def action_to_json(action: CyberAction) -> str:
    return json.dumps(
        {
            "actor": action.actor,
            "tool_name": action.tool_name,
            "target": action.target,
            "params": action.params,
            "rationale": "policy_action",
        },
        ensure_ascii=True,
    )


def collect_rollouts(num_episodes: int = 16, max_steps: int = 60) -> List[RolloutRecord]:
    rows: List[RolloutRecord] = []
    for _ in range(num_episodes):
        env = CyberSelfPlayEnvironment()
        obs = env.reset()
        for t in range(max_steps):
            if obs.done:
                break

            # Red turn first
            red_action = choose_red_action(t)
            obs = env.step(red_action)
            if obs.done:
                break

            # Blue turn (teacher policy for bootstrapping)
            blue_action = heuristic_blue_action(obs.public_state)
            blue_obs = env.step(blue_action)

            rows.append(
                RolloutRecord(
                    prompt=obs_to_prompt(
                        {
                            "public_state": blue_obs.public_state,
                            "telemetry": blue_obs.telemetry,
                            "incident_summary": blue_obs.incident_summary,
                            "metadata": blue_obs.metadata,
                        }
                    ),
                    response=action_to_json(blue_action),
                    reward=float(blue_obs.reward),
                    metadata=blue_obs.metadata or {},
                )
            )
            obs = blue_obs
    return rows


def build_training_dataset(rollouts: List[RolloutRecord], keep_top_frac: float = 0.5) -> Dataset:
    if not rollouts:
        raise ValueError("No rollouts collected.")

    rewards = [r.reward for r in rollouts]
    threshold = sorted(rewards)[max(0, int((1.0 - keep_top_frac) * len(rewards)) - 1)]
    filtered = [r for r in rollouts if r.reward >= threshold]
    texts = [f"{r.prompt}\n{r.response}" for r in filtered]
    print(f"Collected {len(rollouts)} samples, keeping {len(filtered)} with reward >= {threshold:.3f}")
    return Dataset.from_dict({"text": texts})


def run_trl_sft(dataset: Dataset, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir="outputs/cyber-blue-sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        max_seq_length=768,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return args.output_dir


def main() -> None:
    random.seed(42)
    rollouts = collect_rollouts(num_episodes=20, max_steps=80)
    print(f"Average rollout reward: {mean([r.reward for r in rollouts]) if rollouts else 0:.3f}")
    train_ds = build_training_dataset(rollouts, keep_top_frac=0.5)
    out_dir = run_trl_sft(train_ds)
    print(f"Training done. Model saved to: {out_dir}")


if __name__ == "__main__":
    main()
