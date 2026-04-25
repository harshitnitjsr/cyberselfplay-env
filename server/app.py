"""FastAPI app for CyberSelfPlay OpenEnv environment."""

import os
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction, CyberObservation

app = create_app(
    CyberSelfPlayEnvironment,
    CyberAction,
    CyberObservation,
    env_name="CyberSelfPlay",
    max_concurrent_envs=4,
)


def maybe_start_training() -> None:
    """
    Start background training in Space runtime when enabled.

    Enable with:
      RUN_TRAIN_ON_STARTUP=1
    Optional:
      TRAIN_SCRIPT_PATH=train/colab_trl_selfplay.py
      TRAIN_ONCE_TAG=v1   # runs once per tag
    """
    if os.getenv("RUN_TRAIN_ON_STARTUP", "0") != "1":
        return

    script_rel = os.getenv("TRAIN_SCRIPT_PATH", "train/colab_trl_selfplay.py")
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / script_rel
    if not script_path.exists():
        print(f"[train-startup] script not found: {script_path}")
        return

    once_tag = os.getenv("TRAIN_ONCE_TAG", "").strip()
    if once_tag:
        marker_dir = repo_root / ".runtime"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_file = marker_dir / f"train_done_{once_tag}.marker"
        if marker_file.exists():
            print(f"[train-startup] skipped (already ran once for tag '{once_tag}')")
            return
        marker_file.write_text("scheduled", encoding="utf-8")

    cmd = [sys.executable, str(script_path)]
    print(f"[train-startup] launching: {' '.join(cmd)}")
    # Fire-and-forget; logs appear in Space container logs.
    subprocess.Popen(cmd, cwd=str(repo_root))


maybe_start_training()


@app.get("/artifacts")
def list_artifacts() -> dict:
    """
    List locally generated training artifacts and model outputs.
    Useful for quickly checking what was produced in HF Spaces runtime.
    """
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"
    outputs_dir = repo_root / "outputs"

    if not artifacts_dir.exists() and not outputs_dir.exists():
        raise HTTPException(status_code=404, detail="No artifacts or outputs found yet.")

    def _rel_files(base: Path) -> list[str]:
        if not base.exists():
            return []
        return sorted(
            [
                str(p.relative_to(repo_root)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file()
            ]
        )

    return {
        "artifacts": _rel_files(artifacts_dir),
        "outputs": _rel_files(outputs_dir),
    }


def main(host: str = "0.0.0.0", port: int = 7870):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
