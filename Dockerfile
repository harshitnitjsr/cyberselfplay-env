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

# Training is OPT-IN. Set RUN_TRAIN_ON_STARTUP=1 in Space variables to enable.
# TRAIN_ONCE_TAG ensures we only train once per tag value (e.g. "v1") even if
# the Space container restarts.
ENV RUN_TRAIN_ON_STARTUP=0
ENV TRAIN_SCRIPT_PATH=train/colab_trl_selfplay.py
ENV TRAIN_ONCE_TAG=
# Optional Hub upload — leave empty unless you want every training run pushed.
ENV HF_MODEL_REPO=
ENV TRAIN_LEAGUE_ROUNDS=0

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD sh -c 'curl -f "http://localhost:${PORT:-7870}/health" || exit 1'

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-7870}"]
