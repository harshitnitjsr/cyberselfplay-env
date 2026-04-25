"""FastAPI app exposing the CyberSelfPlay environment over OpenEnv."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction, CyberObservation

REPO_ROOT = Path(__file__).resolve().parents[1]

app = create_app(
    CyberSelfPlayEnvironment,
    CyberAction,
    CyberObservation,
    env_name="CyberSelfPlay",
    max_concurrent_envs=4,
)


def maybe_start_training() -> None:
    """
    Optionally launch the training script in a background subprocess at startup.

    Env vars:
      RUN_TRAIN_ON_STARTUP=1            - opt-in
      TRAIN_SCRIPT_PATH=train/grpo_space.py
      TRAIN_ONCE_TAG=v1                 - run once per tag (uses .runtime/ marker)
    """
    flag = os.getenv("RUN_TRAIN_ON_STARTUP", "1")
    script_rel = os.getenv("TRAIN_SCRIPT_PATH", "train/grpo_space.py")
    once_tag = os.getenv("TRAIN_ONCE_TAG", "").strip()
    train_always = os.getenv("TRAIN_ALWAYS", "1") == "1"
    print(
        f"[train-startup] RUN_TRAIN_ON_STARTUP={flag!r} "
        f"TRAIN_SCRIPT_PATH={script_rel!r} TRAIN_ONCE_TAG={once_tag!r} "
        f"TRAIN_ALWAYS={train_always}"
    )

    if flag != "1":
        print("[train-startup] disabled (RUN_TRAIN_ON_STARTUP != '1') — skipping.")
        return

    script_path = REPO_ROOT / script_rel
    if not script_path.exists():
        print(f"[train-startup] script not found: {script_path}; skipping.")
        return

    # TRAIN_ALWAYS=1 bypasses the once-per-tag marker so training fires every startup.
    if once_tag and not train_always:
        marker_dir = REPO_ROOT / ".runtime"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_file = marker_dir / f"train_done_{once_tag}.marker"
        if marker_file.exists():
            print(f"[train-startup] skipped (already ran for tag '{once_tag}')")
            return
        marker_file.write_text("scheduled", encoding="utf-8")
    elif train_always:
        print("[train-startup] TRAIN_ALWAYS=1 — running every startup.")

    log_path = REPO_ROOT / ".runtime" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(script_path)]
    print(f"[train-startup] launching: {' '.join(cmd)} (log: {log_path})")
    try:
        log_fh = open(log_path, "ab", buffering=0)
        subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[train-startup] failed to launch: {exc!s}")


maybe_start_training()


def _rel_files(base: Path) -> List[str]:
    if not base.exists():
        return []
    return sorted(
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        for p in base.rglob("*")
        if p.is_file()
    )


@app.get("/artifacts")
def list_artifacts() -> dict:
    """List training artifacts and model outputs produced inside the container."""
    artifacts_dir = REPO_ROOT / "artifacts"
    outputs_dir = REPO_ROOT / "outputs"

    if not artifacts_dir.exists() and not outputs_dir.exists():
        raise HTTPException(status_code=404, detail="No artifacts or outputs found yet.")

    return {
        "artifacts": _rel_files(artifacts_dir),
        "outputs": _rel_files(outputs_dir),
    }


@app.get("/info")
def env_info() -> dict:
    """Surface tool sets / scenarios so external clients can discover capabilities."""
    from cyber_selfplay_env.tools_blue import BLUE_TOOLS
    from cyber_selfplay_env.tools_red import RED_TOOLS

    return {
        "name": "CyberSelfPlay",
        "blue_tools": sorted(BLUE_TOOLS),
        "red_tools": sorted(RED_TOOLS),
        "scenarios": ["small", "medium", "large"],
        "valid_actors": ["red", "blue"],
    }


def main(host: str = "0.0.0.0", port: int = 7870) -> None:
    """CLI entry point used by `python -m server.app` and the [project.scripts]."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="CyberSelfPlay OpenEnv server")
    parser.add_argument("--host", default=os.getenv("HOST", host))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", str(port))))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
