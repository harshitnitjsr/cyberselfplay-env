# Multi-stage build for the CyberSelfPlay OpenEnv environment.
# Builds a standalone runtime image suitable for Hugging Face Spaces (Docker SDK)
# and any container host.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=cyber_selfplay

COPY . /app/env
WORKDIR /app/env

# Ensure uv is present (the base image usually ships it).
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Resolve runtime dependencies. The optional `train` extra installs the TRL stack
# so that auto-training (RUN_TRAIN_ON_STARTUP=1) works without extra setup.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable --extra train; \
    else \
        uv sync --no-install-project --no-editable --extra train; \
    fi

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# ---------- Runtime defaults (override in Space "Variables and secrets") ----------
ENV PORT=7870
ENV HOST=0.0.0.0
ENV ENABLE_WEB_INTERFACE=true

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Hugging Face caches under /data so they survive Space restarts on persistent disks.
ENV HF_HOME=/data/.huggingface
ENV HUGGINGFACE_HUB_CACHE=/data/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers

# =============================================================================
# Training configuration (all OPT-IN, override any of these in Space variables)
# =============================================================================
# Master switch — flip to "1" in Space "Variables and secrets" to enable.
ENV RUN_TRAIN_ON_STARTUP=0

# Which trainer to launch. Options:
#   train/grpo_space.py          (Unsloth + GRPO, recommended)
#   train/colab_trl_selfplay.py  (TRL SFT — filtered behavior cloning)
#   train/train_blue_vs_pool.py  (Blue PFSP league trainer)
#   train/train_red_vs_pool.py   (Red-side data collector)
ENV TRAIN_SCRIPT_PATH=train/grpo_space.py

# Idempotency tag — bump (e.g. "grpo-v1" -> "grpo-v2") to force re-train.
# Empty string = no idempotency (training runs every container start).
ENV TRAIN_ONCE_TAG=

# ---------- GRPO trainer knobs (read by train/grpo_space.py) ----------
ENV BASE_MODEL=unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
ENV MAX_STEPS=150
ENV NUM_GENERATIONS=4
ENV LEARNING_RATE=5e-6
ENV LORA_RANK=16
ENV OUTPUT_DIR=/data/outputs_cyber

# ---------- TRL SFT trainer knobs (used by colab_trl_selfplay.py) ----------
ENV TRAIN_LEAGUE_ROUNDS=0
ENV TRAIN_NUM_EPISODES=20
ENV TRAIN_TOPK_FRACTION=0.5
ENV TRAIN_NUM_EPOCHS=1
ENV TRAIN_BATCH_SIZE=2
ENV TRAIN_GRAD_ACCUM=4
ENV TRAIN_LR=2e-4
ENV TRAIN_LORA_R=16
ENV TRAIN_LORA_ALPHA=32

# ---------- Hub push (artifacts go here when training finishes) ----------
# Leave empty to skip the upload. HF_TOKEN MUST be set as a SECRET, not a Variable.
ENV TARGET_REPO_ID=
ENV HF_MODEL_REPO=
# HF_TOKEN is intentionally NOT defaulted here — set it as a Space SECRET.
# ENV HF_TOKEN=

# ---------- Curriculum / scenario defaults ----------
ENV CYBER_SCENARIO=small
ENV CYBER_MAX_TURNS=40
ENV CYBER_INSTRUCTION_COUNT=6

# ---------- Logging & misc ----------
ENV TRANSFORMERS_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false
ENV WANDB_DISABLED=true
ENV BITSANDBYTES_NOWELCOME=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f "http://localhost:${PORT:-7870}/health" || exit 1'

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-7870}"]
