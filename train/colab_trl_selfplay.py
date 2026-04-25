"""
Colab-ready HF TRL training loop for CyberSelfPlay.

This script:
1) collects online rollouts from CyberSelfPlayEnvironment,
2) keeps higher-reward Blue actions as training targets,
3) runs TRL SFT fine-tuning with LoRA,
4) writes logs/metrics/plots for README evidence,
5) optionally uploads model + artifacts to Hugging Face Hub.

Usage in Colab:
  !pip install -q "openenv-core[core]" "trl>=0.10.0" "transformers>=4.40.0" "datasets>=2.19.0" "accelerate>=0.30.0" "peft>=0.11.0" "matplotlib>=3.8.0" "huggingface_hub>=0.23.0"
  !python train/colab_trl_selfplay.py
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from datasets import Dataset
from huggingface_hub import HfApi, login
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
    episode_id: int
    step: int
    prompt: str
    response: str
    reward: float
    exfil_rate: float
    mttd: float
    mttr: float
    instruction_completion_rate: float
    metadata: Dict[str, Any]


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


def _safe_float(val: Any, fallback: float = -1.0) -> float:
    if val is None:
        return fallback
    try:
        return float(val)
    except Exception:
        return fallback


def collect_rollouts(num_episodes: int = 16, max_steps: int = 60) -> Tuple[List[RolloutRecord], List[Dict[str, float]]]:
    rows: List[RolloutRecord] = []
    episode_summaries: List[Dict[str, float]] = []
    for ep in range(num_episodes):
        env = CyberSelfPlayEnvironment()
        obs = env.reset()
        blue_rewards: List[float] = []
        exfil_rate = 0.0
        mttd = -1.0
        mttr = -1.0
        inst_completion = 0.0
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
            blue_rewards.append(float(blue_obs.reward))
            metrics = (blue_obs.metadata or {}).get("posg_metrics", {})
            exfil_rate = _safe_float(metrics.get("exfil_success_rate"), exfil_rate)
            mttd = _safe_float(metrics.get("mttd"), mttd)
            mttr = _safe_float(metrics.get("mttr"), mttr)
            inst_completion = _safe_float(metrics.get("instruction_completion_rate"), inst_completion)

            rows.append(
                RolloutRecord(
                    episode_id=ep,
                    step=t,
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
                    exfil_rate=exfil_rate,
                    mttd=mttd,
                    mttr=mttr,
                    instruction_completion_rate=inst_completion,
                    metadata=blue_obs.metadata or {},
                )
            )
            obs = blue_obs

        episode_summaries.append(
            {
                "episode_id": float(ep),
                "avg_blue_reward": mean(blue_rewards) if blue_rewards else 0.0,
                "final_exfil_rate": exfil_rate,
                "final_mttd": mttd,
                "final_mttr": mttr,
                "final_instruction_completion": inst_completion,
                "steps": float(len(blue_rewards)),
            }
        )
    return rows, episode_summaries


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

    # Persist raw train logs from trainer.
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/trainer_log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=True, indent=2)

    return args.output_dir


def save_artifacts(
    rollouts: List[RolloutRecord],
    episode_summaries: List[Dict[str, float]],
    out_dir: str,
) -> None:
    os.makedirs("artifacts", exist_ok=True)

    # Step-level logs.
    with open("artifacts/rollout_logs.jsonl", "w", encoding="utf-8") as f:
        for r in rollouts:
            row = {
                "episode_id": r.episode_id,
                "step": r.step,
                "reward": r.reward,
                "exfil_rate": r.exfil_rate,
                "mttd": r.mttd,
                "mttr": r.mttr,
                "instruction_completion_rate": r.instruction_completion_rate,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    # Episode summaries CSV.
    with open("artifacts/episode_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode_id",
            "avg_blue_reward",
            "final_exfil_rate",
            "final_mttd",
            "final_mttr",
            "final_instruction_completion",
            "steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_summaries:
            writer.writerow(row)

    with open("artifacts/train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_rollout_samples": len(rollouts),
                "num_episodes": len(episode_summaries),
                "avg_rollout_reward": mean([r.reward for r in rollouts]) if rollouts else 0.0,
                "model_output_dir": out_dir,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    # Graph 1: average reward per episode.
    xs = [int(r["episode_id"]) for r in episode_summaries]
    ys_reward = [float(r["avg_blue_reward"]) for r in episode_summaries]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys_reward, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average Blue Reward")
    plt.title("Blue Reward Over Episodes")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/reward_curve.png", dpi=150)
    plt.close()

    # Graph 2: exfiltration downtrend.
    ys_exfil = [float(r["final_exfil_rate"]) for r in episode_summaries]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys_exfil, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Exfiltration Success Rate")
    plt.title("Exfiltration Rate Across Episodes")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/exfiltration_rate_curve.png", dpi=150)
    plt.close()

    # Graph 3: MTTD / MTTR.
    ys_mttd = [float(r["final_mttd"]) for r in episode_summaries]
    ys_mttr = [float(r["final_mttr"]) for r in episode_summaries]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys_mttd, marker="o", label="MTTD")
    plt.plot(xs, ys_mttr, marker="o", label="MTTR")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("MTTD / MTTR Across Episodes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/mttd_mttr_curve.png", dpi=150)
    plt.close()


def maybe_upload_to_hub(model_output_dir: str) -> None:
    """
    Optional upload. Requires:
      HF_TOKEN=<token>
      HF_MODEL_REPO=namespace/model-name
    """
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = os.getenv("HF_MODEL_REPO", "").strip()
    if not token or not repo_id:
        print("[upload] skipped (set HF_TOKEN and HF_MODEL_REPO to enable)")
        return

    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(repo_id=repo_id, folder_path=model_output_dir, repo_type="model")
    api.upload_folder(repo_id=repo_id, folder_path="artifacts", path_in_repo="artifacts", repo_type="model")
    print(f"[upload] model + artifacts uploaded to https://huggingface.co/{repo_id}")


def main() -> None:
    random.seed(42)
    rollouts, episode_summaries = collect_rollouts(num_episodes=20, max_steps=80)
    print(f"Average rollout reward: {mean([r.reward for r in rollouts]) if rollouts else 0:.3f}")
    train_ds = build_training_dataset(rollouts, keep_top_frac=0.5)
    out_dir = run_trl_sft(train_ds)
    save_artifacts(rollouts, episode_summaries, out_dir)
    maybe_upload_to_hub(out_dir)
    print(f"Training done. Model saved to: {out_dir}")


if __name__ == "__main__":
    main()
