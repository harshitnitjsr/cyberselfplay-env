"""GRPO trainer for CyberSelfPlay — runs inside an HF Space (Docker, GPU).

Triggered by `server/app.py::maybe_start_training()` when these are set:
    RUN_TRAIN_ON_STARTUP=1
    TRAIN_SCRIPT_PATH=train/grpo_space.py

Reads these env vars (all optional, sensible defaults):
    BASE_MODEL          (default: unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit)
    MAX_STEPS           (default: 150)
    NUM_GENERATIONS     (default: 4)
    LEARNING_RATE       (default: 5e-6)
    LORA_RANK           (default: 16)
    HF_TOKEN            (write scope; if set, pushes adapter on completion)
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
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction
from cyber_selfplay_env.tools_blue import BLUE_TOOLS

VALID_BLUE = set(BLUE_TOOLS)
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


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
    if a.get("tool_name") not in VALID_BLUE:
        return None, False
    a["actor"] = "blue"
    a.setdefault("target", "host-00")
    a.setdefault("params", {})
    a.setdefault("rationale", "grpo")
    return a, True


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


def _format_reward(text: str) -> float:
    """Smooth, partial-credit reward for JSON formatting progress.

    Gives gradient signal even when the model can't yet emit perfect JSON,
    so GRPO has something to climb toward instead of a flat -1.5 plateau.
    Range: [-1.0, +1.5].
    """
    s = text.strip()
    if len(s) < 5:
        return -1.0
    score = -0.5
    if "{" in s:
        score += 0.2
    if "}" in s:
        score += 0.2
    if "actor" in s and "blue" in s:
        score += 0.3
    if "tool_name" in s:
        score += 0.3
    for tool in VALID_BLUE:
        if tool in s:
            score += 0.5
            break
    if "target" in s and "host-" in s:
        score += 0.2
    return score


def compute_rewards(prompts, completions, **_kwargs):
    rewards = []
    for _p, comp in zip(prompts, completions):
        text = comp if isinstance(comp, str) else comp.get("content", "")

        fmt = _format_reward(text)
        action, ok = parse_action(text)

        if not ok:
            rewards.append(fmt)
            continue

        env = CyberSelfPlayEnvironment()
        env.reset()
        env.step(CyberAction(actor="red", tool_name="attempt_exploit", target="host-01"))
        try:
            obs = env.step(CyberAction(**action))
            env_r = float(obs.reward or 0.0)
        except Exception:
            env_r = -1.0

        r = fmt + env_r + 1.0
        if 30 <= len(text) <= 150:
            r += 0.2
        rewards.append(r)
    return rewards


def main() -> None:
    base_model = _env("BASE_MODEL", "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
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

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_rewards],
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

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
        print(f"[grpo_space] uploaded -> https://huggingface.co/{target_repo}")
    else:
        print("[grpo_space] HF_TOKEN or TARGET_REPO_ID missing — skipping push.")


if __name__ == "__main__":
    main()
