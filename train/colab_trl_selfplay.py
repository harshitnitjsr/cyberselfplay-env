"""
Colab + HF Spaces ready TRL training loop for CyberSelfPlay.

Pipeline:
  1) Collect online rollouts from CyberSelfPlayEnvironment.
  2) Keep higher-reward Blue actions as supervised targets.
  3) Run TRL SFT fine-tuning with LoRA (PEFT).
  4) Save logs / metrics / plots under ./artifacts.
  5) Optionally upload model + artifacts to Hugging Face Hub.

Usage:
  # one-shot single-policy run (default)
  python train/colab_trl_selfplay.py

  # league-style alternating self-play (PFSP opponent sampling)
  TRAIN_LEAGUE_ROUNDS=2 python train/colab_trl_selfplay.py

  # optional HF Hub upload
  HF_TOKEN=hf_xxx HF_MODEL_REPO=myname/cyber-blue python train/colab_trl_selfplay.py
"""

from __future__ import annotations

# Path bootstrap MUST run before any cyber_selfplay/train imports.
from train._bootstrap import ensure_repo_root_on_path  # noqa: E402

ensure_repo_root_on_path()

import csv
import inspect
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Tuple

# Heavy / optional ML deps. Wrapped so the module can still be imported for
# unit tests in environments that lack torch/transformers.
try:
    from datasets import Dataset
    from huggingface_hub import HfApi, login
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    HAS_TRL_STACK = True
except Exception as exc:  # pragma: no cover - depends on env
    print(f"[colab_trl_selfplay] ML stack unavailable: {exc}")
    Dataset = None  # type: ignore[assignment]
    HfApi = None  # type: ignore[assignment]
    login = None  # type: ignore[assignment]
    LoraConfig = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    SFTConfig = None  # type: ignore[assignment]
    SFTTrainer = None  # type: ignore[assignment]
    HAS_TRL_STACK = False

try:
    import matplotlib

    matplotlib.use("Agg")  # headless server safe
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction
from train.pfsp import OpponentStats, sample_opponent


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
    """Simple but informative baseline policy used to bootstrap data."""
    known_incidents = int(obs_public_state.get("known_incident_count", 0))
    if known_incidents > 0:
        return CyberAction(actor="blue", tool_name="isolate_host", target="host-00", params={})

    instr = obs_public_state.get("instruction_progress", {})
    if isinstance(instr, dict) and instr.get("completed", 0) < instr.get("total", 1):
        required_tool = random.choice(["triage_alerts", "deploy_patch", "rotate_secrets"])
        return CyberAction(
            actor="blue",
            tool_name="execute_instruction",
            target="",
            params={"required_tool": required_tool},
        )

    return CyberAction(
        actor="blue",
        tool_name=random.choice(BLUE_TOOLS),
        target="host-00",
        params={},
    )


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


def collect_rollouts(
    num_episodes: int = 16,
    max_steps: int = 60,
    side: str = "blue",
) -> Tuple[List[RolloutRecord], List[Dict[str, float]]]:
    """
    Collect rollouts using the heuristic Blue policy and a scripted Red policy.
    `side` controls which agent's reward stream we record into RolloutRecord.
    """
    rows: List[RolloutRecord] = []
    episode_summaries: List[Dict[str, float]] = []
    for ep in range(num_episodes):
        env = CyberSelfPlayEnvironment()
        obs = env.reset()
        learner_rewards: List[float] = []
        exfil_rate = 0.0
        mttd = -1.0
        mttr = -1.0
        inst_completion = 0.0
        for t in range(max_steps):
            if obs.done:
                break

            red_action = choose_red_action(t)
            red_obs = env.step(red_action)
            if side == "red":
                learner_rewards.append(float(red_obs.reward))
            if red_obs.done:
                obs = red_obs
                break

            blue_action = heuristic_blue_action(red_obs.public_state)
            blue_obs = env.step(blue_action)
            if side == "blue":
                learner_rewards.append(float(blue_obs.reward))

            metrics = (blue_obs.metadata or {}).get("posg_metrics", {})
            exfil_rate = _safe_float(metrics.get("exfil_success_rate"), exfil_rate)
            mttd = _safe_float(metrics.get("mttd"), mttd)
            mttr = _safe_float(metrics.get("mttr"), mttr)
            inst_completion = _safe_float(
                metrics.get("instruction_completion_rate"), inst_completion
            )

            target_action = blue_action if side == "blue" else red_action
            target_reward = (
                float(blue_obs.reward) if side == "blue" else float(red_obs.reward)
            )
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
                    response=action_to_json(target_action),
                    reward=target_reward,
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
                "avg_learner_reward": mean(learner_rewards) if learner_rewards else 0.0,
                "final_exfil_rate": exfil_rate,
                "final_mttd": mttd,
                "final_mttr": mttr,
                "final_instruction_completion": inst_completion,
                "steps": float(len(learner_rewards)),
            }
        )
    return rows, episode_summaries


def build_training_dataset(
    rollouts: List[RolloutRecord],
    keep_top_frac: float = 0.5,
):
    if Dataset is None:
        raise RuntimeError("`datasets` is not installed. pip install datasets")
    if not rollouts:
        raise ValueError("No rollouts collected.")

    rewards = [r.reward for r in rollouts]
    threshold = sorted(rewards)[max(0, int((1.0 - keep_top_frac) * len(rewards)) - 1)]
    filtered = [r for r in rollouts if r.reward >= threshold]
    texts = [f"{r.prompt}\n{r.response}" for r in filtered]
    print(
        f"[dataset] collected {len(rollouts)} samples, "
        f"keeping {len(filtered)} with reward >= {threshold:.3f}"
    )
    return Dataset.from_dict({"text": texts})


def run_trl_sft(
    dataset,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs/cyber-blue-sft",
) -> str:
    if not HAS_TRL_STACK:
        raise RuntimeError(
            "TRL/transformers/peft are required for run_trl_sft. "
            "pip install -e .[train] or use the Colab notebook."
        )

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

    cfg_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "save_steps": 100,
        "report_to": "none",
    }
    sft_cfg_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in sft_cfg_params:
        cfg_kwargs["max_seq_length"] = 768
    elif "max_length" in sft_cfg_params:
        cfg_kwargs["max_length"] = 768

    args = SFTConfig(**cfg_kwargs)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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

    with open("artifacts/episode_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode_id",
            "avg_learner_reward",
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

    if not HAS_MATPLOTLIB:
        print("[artifacts] matplotlib not installed; skipping PNG plots.")
        return

    xs = [int(r["episode_id"]) for r in episode_summaries]
    ys_reward = [float(r["avg_learner_reward"]) for r in episode_summaries]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys_reward, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average Learner Reward")
    plt.title("Learner Reward Over Episodes")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/reward_curve.png", dpi=150)
    plt.close()

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
    """Upload model + artifacts when HF_TOKEN and HF_MODEL_REPO are set."""
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = os.getenv("HF_MODEL_REPO", "").strip()
    if not token or not repo_id:
        print("[upload] skipped (set HF_TOKEN and HF_MODEL_REPO to enable)")
        return
    if HfApi is None or login is None:
        print("[upload] huggingface_hub missing; cannot upload.")
        return

    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    if os.path.isdir(model_output_dir):
        api.upload_folder(
            repo_id=repo_id, folder_path=model_output_dir, repo_type="model"
        )
    if os.path.isdir("artifacts"):
        api.upload_folder(
            repo_id=repo_id,
            folder_path="artifacts",
            path_in_repo="artifacts",
            repo_type="model",
        )
    print(f"[upload] model + artifacts uploaded to https://huggingface.co/{repo_id}")


def run_league_selfplay(rounds: int = 2) -> None:
    """Alternating PFSP league loop. Blue is trained each round; Red side
    contributes data via scripted attacks (placeholder for a learnable Red).
    """
    red_pool = [
        OpponentStats(name="R0", win_rate_vs_learner=0.25, games=20),
        OpponentStats(name="R1", win_rate_vs_learner=0.50, games=20),
        OpponentStats(name="R2", win_rate_vs_learner=0.75, games=20),
    ]
    blue_pool = [
        OpponentStats(name="B0", win_rate_vs_learner=0.25, games=20),
        OpponentStats(name="B1", win_rate_vs_learner=0.50, games=20),
        OpponentStats(name="B2", win_rate_vs_learner=0.75, games=20),
    ]

    last_out_dir = "outputs/cyber-blue-sft"
    for r in range(rounds):
        sampled_red = sample_opponent(red_pool)
        print(f"[league round {r}] train BLUE vs {sampled_red.name}")
        rollouts_b, summaries_b = collect_rollouts(num_episodes=10, max_steps=60, side="blue")
        if HAS_TRL_STACK:
            train_ds_b = build_training_dataset(rollouts_b, keep_top_frac=0.5)
            last_out_dir = run_trl_sft(train_ds_b, output_dir=f"outputs/cyber-blue-sft-r{r}")
        save_artifacts(rollouts_b, summaries_b, last_out_dir)

        sampled_blue = sample_opponent(blue_pool)
        print(f"[league round {r}] collect RED-side data vs {sampled_blue.name}")
        rollouts_r, _ = collect_rollouts(num_episodes=6, max_steps=40, side="red")
        print(f"[league round {r}] red-phase samples={len(rollouts_r)}")

    maybe_upload_to_hub(last_out_dir)


def _print_env_self_check() -> None:
    """Helpful one-line dependency banner so HF Space logs are readable."""
    versions = {}
    for mod in ("trl", "transformers", "datasets", "peft", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "missing"
    print(f"[env-check] python={sys.version.split()[0]} versions={versions}")


def main() -> None:
    random.seed(42)
    _print_env_self_check()
    league_rounds = int(os.getenv("TRAIN_LEAGUE_ROUNDS", "0"))
    try:
        if league_rounds > 0:
            run_league_selfplay(rounds=league_rounds)
            print("[done] league self-play")
            return

        rollouts, episode_summaries = collect_rollouts(num_episodes=20, max_steps=80, side="blue")
        if rollouts:
            print(f"[stats] avg rollout reward: {mean(r.reward for r in rollouts):.3f}")
        out_dir = "outputs/cyber-blue-sft"
        if HAS_TRL_STACK:
            train_ds = build_training_dataset(rollouts, keep_top_frac=0.5)
            out_dir = run_trl_sft(train_ds)
        else:
            print("[skip] TRL stack missing — only collecting rollouts/artifacts.")
        save_artifacts(rollouts, episode_summaries, out_dir)
        maybe_upload_to_hub(out_dir)
        print(f"[done] training. Model dir: {out_dir}")
    except Exception:
        print("[fatal] training run crashed:")
        traceback.print_exc()
        # Re-raise so HF Space logs surface the failure clearly.
        raise


if __name__ == "__main__":
    main()
