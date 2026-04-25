"""SFT + GRPO trainer for CyberSelfPlay — runs inside an HF Space (Docker, GPU).

Two-phase pipeline:
    Phase 1 (SFT):  imitate the heuristic Blue policy on real env rollouts.
                    No reward function — the model just learns the JSON schema.
    Phase 2 (GRPO): refine using ONLY the environment's reward signal.
                    No string-matching, length tricks, or hand-crafted shaping.

Triggered by `server/app.py::maybe_start_training()` when these are set:
    RUN_TRAIN_ON_STARTUP=1
    TRAIN_SCRIPT_PATH=train/grpo_space.py

Reads these env vars (all optional, sensible defaults):
    BASE_MODEL          (default: unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit)
    MAX_STEPS           (default: 150)         GRPO max steps
    NUM_GENERATIONS     (default: 4)
    LEARNING_RATE       (default: 5e-6)        GRPO learning rate
    LORA_RANK           (default: 16)
    DO_SFT              (default: 1)           "0" skips SFT warm-start
    SFT_EPISODES        (default: 30)          rollouts collected with heuristic
    SFT_EPOCHS          (default: 2)
    SFT_LEARNING_RATE   (default: 2e-4)
    INVALID_PENALTY     (default: -1.0)        reward for unparseable output
    HF_TOKEN            (write scope; pushes adapter+plots on completion)
    TARGET_REPO_ID      (default: <user>/cyber-blue-grpo)
    OUTPUT_DIR          (default: /data/outputs_cyber)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

import torch

# Repo root on sys.path so cyber_selfplay_env imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import unsloth  # noqa: E402  — must come before transformers
from datasets import Dataset
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction
from cyber_selfplay_env.tools_blue import BLUE_TOOLS

VALID_BLUE = list(BLUE_TOOLS)
VALID_BLUE_SET = set(BLUE_TOOLS)
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
RED_TOOLS = ["recon_network", "attempt_exploit", "lateral_move", "exfiltrate_data"]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _envf(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _envi(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def parse_action(text: str):
    m = JSON_RE.search(text)
    if not m:
        return None, False
    try:
        a = json.loads(m.group(0))
    except Exception:
        return None, False
    if a.get("tool_name") not in VALID_BLUE_SET:
        return None, False
    a["actor"] = "blue"
    a.setdefault("target", "host-00")
    a.setdefault("params", {})
    a.setdefault("rationale", "grpo")
    return a, True


def assistant_suffix(is_qwen: bool, is_llama: bool) -> str:
    if is_llama:
        return "<|eot_id|>"
    if is_qwen:
        return "<|im_end|>"
    return ""


def heuristic_blue_action(public_state: dict) -> dict:
    """Lightweight imitation target — same logic as train/colab_trl_selfplay.py."""
    known = int(public_state.get("known_incident_count", 0) or 0)
    if known > 0:
        return {
            "actor": "blue", "tool_name": "isolate_host", "target": "host-00",
            "params": {}, "rationale": "contain known breach",
        }
    instr = public_state.get("instruction_progress", {}) or {}
    if isinstance(instr, dict) and instr.get("completed", 0) < instr.get("total", 1):
        required_tool = random.choice(["triage_alerts", "deploy_patch", "rotate_secrets"])
        return {
            "actor": "blue", "tool_name": "execute_instruction", "target": "",
            "params": {"required_tool": required_tool}, "rationale": "follow instruction",
        }
    return {
        "actor": "blue", "tool_name": random.choice(VALID_BLUE),
        "target": "host-00", "params": {}, "rationale": "baseline action",
    }


def collect_sft_dataset(num_episodes: int, system_msg: str,
                        is_qwen: bool, is_llama: bool) -> Dataset:
    """Roll out the env with the heuristic Blue policy and record (prompt, action) pairs."""
    pairs = []
    for _ep in range(num_episodes):
        env = CyberSelfPlayEnvironment()
        obs = env.reset()
        for t in range(40):
            if obs.done:
                break
            red_act = CyberAction(
                actor="red", tool_name=RED_TOOLS[t % len(RED_TOOLS)],
                target=f"host-{t % 6:02d}", params={},
            )
            red_obs = env.step(red_act)
            if red_obs.done:
                break
            blue_dict = heuristic_blue_action(red_obs.public_state)
            prompt = make_prompt(
                {
                    "public_state": red_obs.public_state,
                    "telemetry": red_obs.telemetry,
                    "incident_summary": red_obs.incident_summary,
                },
                system_msg, is_qwen, is_llama,
            )
            completion = json.dumps(blue_dict, ensure_ascii=True) + assistant_suffix(is_qwen, is_llama)
            pairs.append({"text": prompt + completion})
            obs = env.step(CyberAction(**blue_dict))
    return Dataset.from_list(pairs)


def make_prompt(obs_dict: dict, system_msg: str, is_qwen: bool, is_llama: bool) -> str:
    user_msg = (
        f"Observation:\n{json.dumps(obs_dict, ensure_ascii=True)}\n\n"
        "Reply with ONE JSON line ending with '}'. Nothing else."
    )
    if is_llama:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if is_qwen:
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"{system_msg}\n\n{user_msg}\n\nAction JSON:"


INVALID_PENALTY = float(os.environ.get("INVALID_PENALTY", -1.0))


def compute_rewards(prompts, completions, **_kwargs):
    """Pure environmental reward — no string-matching, no length tricks.

    The model is warm-started with SFT against the heuristic Blue policy, so
    by the time GRPO starts, valid JSON is the norm. The reward signal here
    therefore comes entirely from `env.step(action).reward`.

    - parseable action -> env reward
    - unparseable      -> INVALID_PENALTY (single fixed value, default -1.0)
    """
    rewards = []
    for _p, comp in zip(prompts, completions):
        text = comp if isinstance(comp, str) else comp.get("content", "")
        action, ok = parse_action(text)
        if not ok:
            rewards.append(INVALID_PENALTY)
            continue
        env = CyberSelfPlayEnvironment()
        env.reset()
        env.step(CyberAction(actor="red", tool_name="attempt_exploit", target="host-01"))
        try:
            obs = env.step(CyberAction(**action))
            rewards.append(float(obs.reward or 0.0))
        except Exception:
            rewards.append(INVALID_PENALTY)
    return rewards


class MetricsLogger(TrainerCallback):
    """Tee per-step metrics to stdout AND a TSV file for later inspection."""

    def __init__(self, log_path: str) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("step\tloss\treward\treward_std\tkl\tmean_len\n")

    def on_log(self, args, state, control, logs=None, **_kwargs):
        if not logs or "reward" not in logs:
            return
        loss = logs.get("loss", logs.get("train_loss", 0.0))
        line = (
            f"[grpo_space] step={state.global_step:>4d}  "
            f"loss={loss:+.4f}  "
            f"reward={logs.get('reward', 0):+.3f}  "
            f"reward_std={logs.get('reward_std', 0):.3f}  "
            f"kl={logs.get('kl', 0):.4f}  "
            f"mean_len={logs.get('completions/mean_length', 0):.1f}"
        )
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(
                f"{state.global_step}\t{loss:.6f}\t"
                f"{logs.get('reward', 0):.6f}\t{logs.get('reward_std', 0):.6f}\t"
                f"{logs.get('kl', 0):.6f}\t{logs.get('completions/mean_length', 0):.3f}\n"
            )


def save_training_plots(trainer, out_dir: str) -> None:
    """Render reward / loss / kl / length plots from trainer.state.log_history.

    Writes ``training_curves.png`` (4-panel summary) and ``log_history.json``
    (raw metrics, for re-plotting later) into ``out_dir``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[grpo_space] matplotlib unavailable, skipping plots: {exc!s}")
        return

    history = list(trainer.state.log_history or [])
    if not history:
        print("[grpo_space] log_history empty, skipping plots.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(out_dir) / "log_history.json"
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[grpo_space] wrote raw metrics -> {json_path}")

    def series(key: str):
        xs, ys = [], []
        for row in history:
            if key in row and "step" in row:
                xs.append(row["step"])
                ys.append(row[key])
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("CyberSelfPlay GRPO — training curves", fontsize=14)

    ax = axes[0, 0]
    rx, ry = series("reward")
    sx, sy = series("reward_std")
    if rx:
        ax.plot(rx, ry, color="#2563eb", linewidth=2, label="reward (mean)")
    if sx:
        ax.fill_between(
            sx,
            [m - s for m, s in zip(ry, sy)] if len(ry) == len(sy) else sy,
            [m + s for m, s in zip(ry, sy)] if len(ry) == len(sy) else sy,
            color="#2563eb", alpha=0.15, label="±1 std",
        )
        ax2 = ax.twinx()
        ax2.plot(sx, sy, color="#f97316", linewidth=1, linestyle="--", label="reward_std")
        ax2.set_ylabel("reward_std", color="#f97316")
        ax2.tick_params(axis="y", labelcolor="#f97316")
    ax.axhline(0.5, color="#dc2626", linestyle=":", linewidth=1, label="format ceiling (0.5)")
    ax.axhline(1.5, color="#16a34a", linestyle=":", linewidth=1, label="valid-JSON floor (1.5)")
    ax.set_title("Reward progression")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    kx, ky = series("kl")
    if kx:
        ax.plot(kx, ky, color="#9333ea", linewidth=2)
    ax.set_title("KL divergence (policy vs reference)")
    ax.set_xlabel("step")
    ax.set_ylabel("kl")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    mx, my = series("completions/mean_length")
    nx, ny = series("completions/min_length")
    xx, xy = series("completions/max_length")
    if mx:
        ax.plot(mx, my, color="#0891b2", linewidth=2, label="mean")
    if nx:
        ax.plot(nx, ny, color="#64748b", linewidth=1, linestyle="--", label="min")
    if xx:
        ax.plot(xx, xy, color="#64748b", linewidth=1, linestyle=":", label="max")
    ax.set_title("Completion length (tokens)")
    ax.set_xlabel("step")
    ax.set_ylabel("length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lx, ly = series("loss")
    if not lx:
        lx, ly = series("train_loss")
    if lx:
        ax.plot(lx, ly, color="#be123c", linewidth=2)
    ax.set_title("Training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = Path(out_dir) / "training_curves.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grpo_space] wrote plots -> {png_path}")


def main() -> None:
    base_model = _env("BASE_MODEL", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    max_steps = _envi("MAX_STEPS", 150)
    num_generations = _envi("NUM_GENERATIONS", 4)
    learning_rate = _envf("LEARNING_RATE", 5e-6)
    lora_rank = _envi("LORA_RANK", 16)
    output_dir = _env("OUTPUT_DIR", "/data/outputs_cyber")
    target_repo = _env("TARGET_REPO_ID", "")
    hf_token = _env("HF_TOKEN", "")
    # GRPO trainer knobs (all overridable via env vars)
    max_seq_len = _envi("MAX_SEQ_LEN", 1024)
    max_completion_length = _envi("MAX_COMPLETION_LENGTH", 128)
    per_device_batch_size = _envi("PER_DEVICE_BATCH_SIZE", 4)
    grad_accum_steps = _envi("GRAD_ACCUM_STEPS", 2)
    warmup_steps = _envi("WARMUP_STEPS", 10)
    save_steps = _envi("SAVE_STEPS", 50)
    logging_steps = _envi("LOGGING_STEPS", 5)
    temperature = _envf("TEMPERATURE", 0.7)
    top_p = _envf("TOP_P", 0.9)
    beta = _envf("BETA", 0.02)
    max_grad_norm = _envf("MAX_GRAD_NORM", 1.0)
    num_prompts = _envi("NUM_PROMPTS", 64)
    # SFT warm-start knobs
    do_sft = os.environ.get("DO_SFT", "1") == "1"
    sft_episodes = _envi("SFT_EPISODES", 30)
    sft_epochs = _envi("SFT_EPOCHS", 2)
    sft_lr = _envf("SFT_LEARNING_RATE", 2e-4)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    print(f"[grpo_space] cuda={has_cuda} bf16={use_bf16} fp16={use_fp16}")
    print(f"[grpo_space] base_model={base_model} max_steps={max_steps}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.model_max_length = max_seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
    )

    is_llama = "llama" in base_model.lower()
    is_qwen = "qwen" in base_model.lower()
    system_msg = (
        "You are the Blue defender in a cyber self-play game. "
        "Output EXACTLY one JSON object on a single line, then stop. No explanation.\n"
        f"Schema: {{\"actor\":\"blue\",\"tool_name\":<one of {sorted(VALID_BLUE)}>,"
        "\"target\":\"host-XX\",\"params\":{},\"rationale\":\"short\"}}\n"
        "Example: {\"actor\":\"blue\",\"tool_name\":\"isolate_host\","
        "\"target\":\"host-02\",\"params\":{},\"rationale\":\"contain breach\"}"
    )

    random.seed(42)

    if do_sft:
        print(f"[grpo_space] === Phase 1: SFT warm-start "
              f"({sft_episodes} episodes, {sft_epochs} epochs) ===")
        sft_dataset = collect_sft_dataset(sft_episodes, system_msg, is_qwen, is_llama)
        print(f"[grpo_space] SFT dataset size = {len(sft_dataset)}")
        sft_args = SFTConfig(
            output_dir=os.path.join(output_dir, "sft"),
            learning_rate=sft_lr,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=sft_epochs,
            logging_steps=logging_steps,
            warmup_steps=5,
            optim="adamw_8bit",
            bf16=use_bf16,
            fp16=use_fp16,
            save_strategy="no",
            report_to="none",
            max_length=max_seq_len,
            dataset_text_field="text",
            packing=True,
            packing_strategy="bfd",
        )
        sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=sft_args,
            train_dataset=sft_dataset,
        )
        sft_trainer.train()
        print("[grpo_space] SFT done.")
    else:
        print("[grpo_space] DO_SFT=0 — skipping SFT warm-start.")

    print("[grpo_space] === Phase 2: GRPO with environment reward only ===")
    prompts = []
    for _ in range(num_prompts):
        env = CyberSelfPlayEnvironment()
        env.reset()
        for _ in range(random.randint(0, 4)):
            env.step(CyberAction(
                actor="red",
                tool_name="attempt_exploit",
                target=f"host-{random.randint(0, 5):02d}",
            ))
        obs = env.step(CyberAction(actor="red", tool_name="recon_network", target="host-00"))
        prompts.append({"prompt": make_prompt(
            {
                "public_state": obs.public_state,
                "telemetry": obs.telemetry,
                "incident_summary": obs.incident_summary,
            },
            system_msg, is_qwen, is_llama,
        )})
    train_dataset = Dataset.from_list(prompts)
    print(f"[grpo_space] dataset size = {len(train_dataset)}")

    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        optim="adamw_8bit",
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=not has_cuda,
        save_steps=save_steps,
        report_to="none",
        temperature=temperature,
        top_p=top_p,
        beta=beta,
        max_grad_norm=max_grad_norm,
    )

    metrics_log = os.path.join(output_dir, "train_metrics.log")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_rewards],
        args=args,
        train_dataset=train_dataset,
        callbacks=[MetricsLogger(metrics_log)],
    )
    trainer.train()

    save_training_plots(trainer, output_dir)

    save_dir = os.path.join(output_dir, "cyber-blue-grpo-lora")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[grpo_space] saved adapter to {save_dir}")

    if target_repo and hf_token:
        from huggingface_hub import HfApi, login
        login(token=hf_token)
        api = HfApi()
        api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=target_repo, folder_path=save_dir, repo_type="model")
        for fname in ("training_curves.png", "log_history.json", "train_metrics.log"):
            fpath = Path(output_dir) / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=target_repo,
                    repo_type="model",
                )
        print(f"[grpo_space] uploaded -> https://huggingface.co/{target_repo}")
    else:
        print("[grpo_space] HF_TOKEN or TARGET_REPO_ID missing — skipping push.")


if __name__ == "__main__":
    main()
