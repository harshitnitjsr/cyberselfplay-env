# CyberSelfPlay: Long-Horizon Cyber Defense POSG (OpenEnv)

`cyber_selfplay` is an OpenEnv-compliant environment for **Theme #2 (Super Long-Horizon Planning & Instruction Following)** and **Theme #4 (Self-Improvement)**.

It trains two LLM agents in self-play:
- **Red** (attacker): multi-stage intrusion, persistence, and exfiltration.
- **Blue** (defender): noisy triage, containment, patching, instruction-following recovery.

## Why this matches the hackathon themes

- **Theme #2:** the Blue agent must follow long mission plans (up to `300` instructions in `large` scenario), checkpoint progress, recover from earlier mistakes, and operate with delayed rewards.
- **Theme #4:** the environment includes adaptive curriculum escalation and PFSP/PSRO-style league training utilities for continual self-improvement.

## Environment model (POSG)

We model the game as a two-player partially observable stochastic game:

\[
\mathcal{G}=\langle \mathcal{S},\mathcal{A}_R,\mathcal{A}_B,\mathcal{O}_R,\mathcal{O}_B,T,Z_R,Z_B,r_R,r_B,\gamma \rangle
\]

With objectives:

\[
J_i(\pi_i,\pi_{-i})=\mathbb{E}\left[\sum_{t=0}^{H}\gamma^t r_i(s_t,a_t^R,a_t^B)\right],\ i\in\{R,B\}
\]

Near-zero-sum variant with collateral term:

\[
r_B = -r_R - \lambda C_{\text{collateral}}
\]

## Reward decomposition (dense + delayed)

Red reward is event-based:
\[
r_R = w_1 I_{\text{foothold}} + w_2 I_{\text{lateral}} + w_3 I_{\text{priv}} + w_4 I_{\text{exfil}} - w_5 I_{\text{detect}} - \eta_R
\]

Blue reward combines cyber and instruction-following signals:
\[
r_B = v_1 I_{\text{detect}} + v_2 I_{\text{contain}} + v_3 I_{\text{recover}} - v_4 I_{\text{exfil}} + v_5 I_{\text{instruction\_progress}} + v_6 I_{\text{checkpoint}} - \eta_B
\]

Implemented in `cyber_selfplay_env/rubrics.py`.

## Self-improvement loop (league + PFSP + PSRO utilities)

PFSP weighting:
\[
f(w)=w(1-w)
\]
(implemented in `train/pfsp.py`)

Minimal PSRO/meta-strategy helper:
- `train/psro_meta.py` (replicator-style updates on restricted game payoffs)
- `train/evaluate_league.py` includes an exploitability-gap proxy output.

## Long-horizon mechanics

Implemented in `cyber_selfplay_env/simulator.py`:
- Scenario difficulty with increasing horizon:
  - `small`: 40 instructions
  - `medium`: 120 instructions
  - `large`: 300 instructions
- Delayed plan checkpoints (`checkpoint_every`)
- Instruction completion/violation tracking
- Blue timeout win requires enough instruction completion (not only survival)

Implemented in `cyber_selfplay_env/curriculum.py`:
- Adaptive escalation `small -> medium -> large` by rolling Blue performance.

## Built-in metrics for judging

Per-step metadata includes `posg_metrics` from `cyber_selfplay_env/metrics.py`:
- `mttd`
- `mttr`
- `exfil_success_rate`
- `critical_asset_compromise_rate`
- `false_positive_disruption_cost`
- `instruction_completion_rate`
- `instruction_violation_rate`

## Project structure

- `cyber_selfplay_env/environment.py`: OpenEnv environment class
- `cyber_selfplay_env/simulator.py`: hidden dynamics + long-horizon mission logic
- `cyber_selfplay_env/rubrics.py`: reward functions
- `cyber_selfplay_env/metrics.py`: POSG and recovery metrics
- `cyber_selfplay_env/curriculum.py`: adaptive difficulty progression
- `cyber_selfplay_env/tools_red.py`, `cyber_selfplay_env/tools_blue.py`: action registries
- `server/app.py`: OpenEnv HTTP server app
- `openenv.yaml`: environment manifest
- `train/train_blue_vs_pool.py`, `train/train_red_vs_pool.py`: league loops with PFSP
- `train/colab_trl_selfplay.py`: Colab-ready TRL loop (with fallback mode)
- `notebooks/colab_trl_selfplay.ipynb`: quick Colab notebook

## Quick start

```bash
pip install -e .[train]
python -m server.app
```

Connect client to:
`http://localhost:7870`

## Training in Colab (minimum requirement)

Run either:
- `notebooks/colab_trl_selfplay.ipynb`
- `python train/colab_trl_selfplay.py`

Both include a minimal TRL training path and a fallback path so the script still runs if full TRL stack is unavailable.

## Submission checklist (fill before deadline)

- [ ] Hugging Face Space URL: `<add-space-link>`
- [ ] Mini-blog / video / slides URL: `<add-demo-link>`
- [ ] Reward and training curves embedded in this README
- [ ] Before-vs-after qualitative trace in README

## References

- [Grandmaster level in StarCraft II using multi-agent reinforcement learning (Nature 2019)](https://www.nature.com/articles/s41586-019-1724-z)
- [PSRO: A Unified Game-Theoretic Approach to Multiagent RL (NeurIPS 2017)](https://mlanctot.info/files/papers/nips17-psro.pdf)
- [Adaptive Cyber Defense Against Multi-Stage Attacks Using Learning-Based POMDP (ACM TOPS 2021)](https://doi.org/10.1145/3418897)
- [CAGE-2 POMDP defender formulation](https://arxiv.org/html/2509.06539v1)
