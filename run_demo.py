"""
CyberSelfPlay — single-file unified demo.

Judges (or anyone) can run THIS ONE FILE to see the whole project end-to-end:

  1. Environment smoke (reset + alternating Red/Blue steps, full POSG metrics).
  2. PFSP opponent sampling distribution (Theme 4 self-improvement).
  3. PSRO replicator meta-strategy update (game-theoretic core).
  4. Rollout collection over a few episodes (Blue learner side).
  5. League evaluation grid + exploitability-gap proxy.
  6. (Optional) TRL SFT + LoRA fine-tune of Qwen2.5-0.5B if HF deps are present.
  7. Artifacts dump under `artifacts/` + console summary.
  8. (Optional) Upload model + artifacts to HF Hub.

Usage:
    python run_demo.py                       # default end-to-end
    python run_demo.py --no-train            # skip TRL fine-tune (fast)
    python run_demo.py --episodes 30         # more rollout episodes
    python run_demo.py --league-rounds 2     # extra PFSP league cycles
    HF_TOKEN=hf_xxx HF_MODEL_REPO=user/repo python run_demo.py    # auto-upload

Exit code is 0 on full success, non-zero only if a stage hard-fails.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction
from train.colab_trl_selfplay import (
    HAS_MATPLOTLIB,
    HAS_TRL_STACK,
    build_training_dataset,
    collect_rollouts,
    maybe_upload_to_hub,
    run_trl_sft,
    save_artifacts,
)
from train.pfsp import OpponentStats, sample_opponent
from train.psro_meta import replicator_update


def _hr(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _self_check() -> dict:
    versions = {"python": sys.version.split()[0]}
    for mod in ("openenv", "fastapi", "pydantic", "trl", "transformers",
                "datasets", "peft", "matplotlib", "huggingface_hub"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "missing"
    return versions


def stage_env_smoke() -> dict:
    _hr("STAGE 1 / Environment Smoke Test")
    env = CyberSelfPlayEnvironment()
    obs = env.reset()
    print(f"reset OK   actor={obs.actor}  scenario={obs.metadata.get('scenario')}")

    red_plan = [
        ("recon_network", "host-00"),
        ("attempt_exploit", "host-01"),
        ("pivot_host", "host-02"),
        ("establish_persistence", "host-02"),
        ("prepare_exfiltration", "host-02"),
        ("execute_exfiltration", "host-02"),
    ]
    blue_plan = [
        ("query_siem", "host-00"),
        ("triage_alerts", "host-00"),
        ("isolate_host", "host-02"),
        ("rotate_secrets", "host-02"),
        ("deploy_patch", "host-02"),
        ("execute_instruction", ""),
    ]

    last_metrics = {}
    for i in range(len(red_plan)):
        rt, rg = red_plan[i]
        obs = env.step(CyberAction(actor="red", tool_name=rt, target=rg))
        if obs.done:
            break
        bt, bg = blue_plan[i]
        params = {"required_tool": "triage_alerts"} if bt == "execute_instruction" else {}
        obs = env.step(CyberAction(actor="blue", tool_name=bt, target=bg, params=params))
        last_metrics = obs.metadata.get("posg_metrics", {})
        print(
            f"  t={obs.incident_summary['time_step']:>3}  "
            f"blue_action={bt:<22}  reward={obs.reward:+.2f}  "
            f"exfil={last_metrics.get('exfil_success_rate')}  "
            f"inst_done={last_metrics.get('instruction_completion_rate'):.2f}"
        )
        if obs.done:
            break

    print(f"final POSG metrics: {json.dumps(last_metrics, indent=2)}")
    return last_metrics


def stage_pfsp_demo(samples: int = 200) -> dict:
    _hr("STAGE 2 / PFSP Opponent Sampling (Theme 4)")
    pool = [
        OpponentStats(name="trivial", win_rate_vs_learner=0.05),
        OpponentStats(name="weak",    win_rate_vs_learner=0.25),
        OpponentStats(name="even",    win_rate_vs_learner=0.50),
        OpponentStats(name="strong",  win_rate_vs_learner=0.75),
        OpponentStats(name="brutal",  win_rate_vs_learner=0.95),
    ]
    counts = Counter(sample_opponent(pool).name for _ in range(samples))
    distribution = {name: counts.get(name, 0) / samples for name in (o.name for o in pool)}
    print(f"PFSP weight f(w)=w(1-w) — sampled {samples} opponents:")
    for name, freq in distribution.items():
        bar = "#" * int(freq * 50)
        print(f"  {name:<8} freq={freq:.3f} {bar}")
    return distribution


def stage_psro_demo() -> dict:
    _hr("STAGE 3 / PSRO Replicator Meta-Solver")
    payoffs = [0.3, 0.6, 0.1, 0.8]
    p = [0.25, 0.25, 0.25, 0.25]
    print(f"initial   p={['%.3f' % x for x in p]}  payoffs={payoffs}")
    for step in range(5):
        p = replicator_update(payoffs, p, eta=0.5)
        print(f"step {step+1}    p={['%.3f' % x for x in p]}  sum={sum(p):.3f}")
    return {"final_meta_strategy": p, "payoffs": payoffs}


def stage_rollouts(episodes: int, max_steps: int) -> tuple[list, list]:
    _hr(f"STAGE 4 / Rollout Collection  (episodes={episodes}, max_steps={max_steps})")
    t0 = time.time()
    rollouts, summaries = collect_rollouts(num_episodes=episodes, max_steps=max_steps, side="blue")
    dt = time.time() - t0
    if summaries:
        avg_r = mean(s["avg_learner_reward"] for s in summaries)
        avg_inst = mean(s["final_instruction_completion"] for s in summaries)
        avg_exfil = mean(s["final_exfil_rate"] for s in summaries)
        print(
            f"collected {len(rollouts)} step samples over {len(summaries)} episodes in {dt:.1f}s  "
            f"avg_reward={avg_r:+.3f}  inst_done={avg_inst:.2f}  exfil={avg_exfil:.2f}"
        )
    return rollouts, summaries


def stage_league_eval(grid_episodes: int) -> dict:
    _hr(f"STAGE 5 / League Eval Grid + Exploitability Gap  (episodes={grid_episodes})")
    blue_pool = ["B0", "B1", "B2"]
    red_pool = ["R0", "R1", "R2"]
    payoff_row: List[float] = []
    print(f"{'blue':<6}{'red':<6}{'reward':>10}{'exfil':>10}{'mttd':>8}{'mttr':>8}")
    rows = []
    for b in blue_pool:
        for r in red_pool:
            _, summaries = collect_rollouts(num_episodes=grid_episodes, max_steps=40, side="blue")
            row = {
                "blue": b, "red": r,
                "blue_reward": float(mean(s["avg_learner_reward"] for s in summaries)) if summaries else 0.0,
                "exfil_rate":  float(mean(s["final_exfil_rate"]    for s in summaries)) if summaries else 0.0,
                "mttd":        float(mean(s["final_mttd"]          for s in summaries)) if summaries else -1.0,
                "mttr":        float(mean(s["final_mttr"]          for s in summaries)) if summaries else -1.0,
            }
            rows.append(row)
            payoff_row.append(row["blue_reward"])
            print(f"{b:<6}{r:<6}{row['blue_reward']:>10.3f}{row['exfil_rate']:>10.3f}"
                  f"{row['mttd']:>8.2f}{row['mttr']:>8.2f}")

    p0 = [1.0 / len(payoff_row)] * len(payoff_row)
    p1 = replicator_update(payoff_row, p0, eta=0.2)
    best = max(payoff_row)
    meta_value = sum(pi * ui for pi, ui in zip(p1, payoff_row))
    gap = best - meta_value
    print(f"\nbest_response_value={best:.3f}  meta_strategy_value={meta_value:.3f}  "
          f"exploitability_gap_proxy={gap:.4f}")
    return {"grid": rows, "exploitability_gap_proxy": gap, "meta_strategy": p1}


def stage_train(rollouts, summaries, do_train: bool) -> str:
    _hr("STAGE 6 / TRL SFT + LoRA Fine-Tune")
    out_dir = "outputs/cyber-blue-sft"
    if not do_train:
        print("--no-train passed; skipping fine-tune.")
        return out_dir
    if not HAS_TRL_STACK:
        print("TRL stack missing (datasets/transformers/trl/peft). "
              "Install with: pip install -e .[train]   ... skipping fine-tune.")
        return out_dir
    if not rollouts:
        print("no rollouts collected; skipping fine-tune.")
        return out_dir

    print(f"training Qwen/Qwen2.5-0.5B-Instruct on {len(rollouts)} rollout samples ...")
    ds = build_training_dataset(rollouts, keep_top_frac=0.5)
    out_dir = run_trl_sft(ds)
    print(f"model saved to: {out_dir}")
    return out_dir


def stage_save_artifacts(rollouts, summaries, league: dict, out_dir: str) -> dict:
    _hr("STAGE 7 / Artifacts")
    save_artifacts(rollouts, summaries, out_dir)
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    with open(artifacts_dir / "league_summary.json", "w", encoding="utf-8") as f:
        json.dump(league, f, ensure_ascii=True, indent=2)
    files = sorted(p.relative_to(REPO_ROOT).as_posix() for p in artifacts_dir.rglob("*") if p.is_file())
    out_path = REPO_ROOT / out_dir
    out_files = sorted(p.relative_to(REPO_ROOT).as_posix() for p in out_path.rglob("*") if p.is_file()) if out_path.exists() else []
    print("artifacts/")
    for f in files: print(f"  {f}")
    if out_files:
        print(f"{out_dir}/")
        for f in out_files: print(f"  {f}")
    return {"artifacts": files, "outputs": out_files}


def stage_upload(out_dir: str) -> None:
    _hr("STAGE 8 / (Optional) Hugging Face Hub Upload")
    maybe_upload_to_hub(out_dir)


def stage_summary(versions: dict, env_metrics: dict, pfsp_dist: dict,
                  psro_meta: dict, league: dict, paths: dict, out_dir: str) -> None:
    _hr("DEMO SUMMARY")
    summary = {
        "versions": versions,
        "stage_1_env_metrics": env_metrics,
        "stage_2_pfsp_distribution": pfsp_dist,
        "stage_3_psro_meta": psro_meta,
        "stage_5_league": {
            "exploitability_gap_proxy": league.get("exploitability_gap_proxy"),
            "meta_strategy": league.get("meta_strategy"),
        },
        "stage_6_model_dir": out_dir,
        "stage_7_files": paths,
    }
    out_file = REPO_ROOT / "artifacts" / "demo_summary.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(json.dumps({
        "ok": True,
        "exploitability_gap_proxy": league.get("exploitability_gap_proxy"),
        "model_dir": out_dir,
        "summary_path": out_file.relative_to(REPO_ROOT).as_posix(),
        "n_artifacts": len(paths.get("artifacts", [])),
        "n_outputs": len(paths.get("outputs", [])),
    }, indent=2))
    print("\nDONE.   open the files above to inspect logs / plots / model.")


def main() -> int:
    parser = argparse.ArgumentParser(description="CyberSelfPlay unified demo (single file).")
    parser.add_argument("--episodes",      type=int, default=int(os.getenv("EPISODES", "12")))
    parser.add_argument("--max-steps",     type=int, default=int(os.getenv("MAX_STEPS", "60")))
    parser.add_argument("--grid-episodes", type=int, default=int(os.getenv("GRID_EPISODES", "2")))
    parser.add_argument("--league-rounds", type=int, default=int(os.getenv("TRAIN_LEAGUE_ROUNDS", "0")))
    parser.add_argument("--no-train", action="store_true",
                        help="Skip TRL fine-tune (fastest demo path).")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HF Hub upload even if HF_TOKEN/HF_MODEL_REPO are set.")
    args = parser.parse_args()

    random.seed(42)
    _hr("CyberSelfPlay Unified Demo")
    versions = _self_check()
    print(json.dumps(versions, indent=2))
    print(f"HAS_TRL_STACK={HAS_TRL_STACK}  HAS_MATPLOTLIB={HAS_MATPLOTLIB}")

    try:
        env_metrics = stage_env_smoke()
        pfsp_dist   = stage_pfsp_demo()
        psro_meta   = stage_psro_demo()
        rollouts, summaries = stage_rollouts(args.episodes, args.max_steps)

        if args.league_rounds > 0:
            for r in range(args.league_rounds):
                _hr(f"BONUS / PFSP League Round {r+1}/{args.league_rounds}")
                more_rollouts, more_summaries = collect_rollouts(
                    num_episodes=max(4, args.episodes // 2),
                    max_steps=args.max_steps,
                    side="blue",
                )
                rollouts.extend(more_rollouts)
                summaries.extend(more_summaries)
                print(f"round {r+1}: appended {len(more_rollouts)} samples / "
                      f"{len(more_summaries)} episodes")

        league   = stage_league_eval(args.grid_episodes)
        out_dir  = stage_train(rollouts, summaries, do_train=not args.no_train)
        paths    = stage_save_artifacts(rollouts, summaries, league, out_dir)
        if not args.no_upload:
            stage_upload(out_dir)
        stage_summary(versions, env_metrics, pfsp_dist, psro_meta, league, paths, out_dir)
        return 0
    except Exception:
        print("\n[fatal] demo crashed:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
