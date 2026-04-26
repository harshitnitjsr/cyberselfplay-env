---
title: CyberSelfPlay (Long-Horizon Cyber POSG)
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7870
pinned: true
---

# CyberSelfPlay: Autonomous Red-vs-Blue Cyber Defense Environment

CyberSelfPlay is an OpenEnv-compatible reinforcement learning environment for long-horizon cyber defense. The setting is a partially observable, stochastic Red-vs-Blue contest where Blue must execute enterprise recovery playbooks while Red applies adversarial pressure.

## Environment on Hugging Face Space

- **Live Space:** `https://huggingface.co/spaces/HarshitShri026`

---

## Problem and Capability Gap

Most agent benchmarks are short-horizon and single-agent. Cyber defense in practice is long-horizon, partially observable, adversarial, and stochastic. CyberSelfPlay targets that gap by coupling multi-step mission execution with attacker-defender interaction and structured tool actions.

---

## Environment Design

### What the agent observes

Blue receives partial system state, mission context, and progress-related metadata.

### What the agent does

Actions are emitted as structured `CyberAction` JSON using the environment tool interface.

### What the agent is rewarded for

Rewards combine security outcomes (detection, containment, recovery, exfiltration pressure) and mission outcomes (instruction progress, checkpoints, violations).

### Formal game model

CyberSelfPlay is modeled as a two-player partially observable stochastic game (POSG):

$$
\mathcal{G}=\langle \mathcal{S},\mathcal{A}_R,\mathcal{A}_B,\mathcal{O}_R,\mathcal{O}_B,T,Z_R,Z_B,r_R,r_B,\gamma \rangle
$$

where:

$$
\begin{align*}
\mathcal{S} &: \text{hidden environment state space} \\
\mathcal{A}_R, \mathcal{A}_B &: \text{Red and Blue action spaces} \\
\mathcal{O}_R, \mathcal{O}_B &: \text{Red and Blue observation spaces} \\
T &: \text{state-transition kernel} \\
Z_R, Z_B &: \text{observation emission models for each player} \\
r_R, r_B &: \text{Red and Blue reward functions} \\
\gamma &: \text{discount factor}
\end{align*}
$$

with objective:

$$
J_i(\pi_i,\pi_{-i})=\mathbb{E}\left[\sum_{t=0}^{H}\gamma^t r_i\left(s_t,a_t^{R},a_t^{B}\right)\right],\quad i\in\{R,B\}
$$

with:

$$
\begin{aligned}
\pi_i &: \text{the policy of player } i \text{ and } \pi_{-i} \text{ the opponent policy} \\
H &: \text{the episode horizon (maximum time steps)} \\
s_t &: \text{the state at time } t \\
a_t^{R}, a_t^{B} &: \text{the Red and Blue actions at time } t \\
r_i(\cdot) &: \text{the reward received by player } i
\end{aligned}
$$

and near-zero-sum coupling:

$$
r_B=-r_R-\lambda C_{\mathrm{collateral}}.
$$

---

## Reward Structure


**Red reward**

$$
\begin{aligned}
r_R &= w_1 \mathbb{1}_{\mathrm{foothold}} + w_2 \mathbb{1}_{\mathrm{priv}} + w_3 \mathbb{1}_{\mathrm{lateral}} + w_4 \mathbb{1}_{\mathrm{exfil}} \\
&\quad - w_5 \mathbb{1}_{\mathrm{detect}} + w_6 \mathbb{1}_{\mathrm{plan\_sabotage}} - \eta_R
\end{aligned}
$$

**Blue reward**

$$
\begin{aligned}
r_B &= v_1 \mathbb{1}_{\mathrm{detect}} + v_2 \mathbb{1}_{\mathrm{contain}} + v_3 \mathbb{1}_{\mathrm{recover}} - v_4 \mathbb{1}_{\mathrm{exfil}} \\
&\quad + v_5 \mathbb{1}_{\mathrm{instr\_progress}} + v_6 \mathbb{1}_{\mathrm{checkpoint}} - v_7 \mathbb{1}_{\mathrm{instr\_violation}} \\
&\quad + v_8 \rho_{\mathrm{inst}} - \eta_B
\end{aligned}
$$

Concrete rubric implementation is in `cyber_selfplay_env/rubrics.py`.

---

## Environment Architecture

<img src="https://res.cloudinary.com/dgyebzm4w/image/upload/v1777181551/architecture_dus774.svg" width = "800"/>

## Training Flow

<img src="https://res.cloudinary.com/dgyebzm4w/image/upload/v1777181108/training-flow_a3xupo.svg" width="800"/>

---

## 🚀 Training Approaches in This Project

This project explores multiple training strategies for learning robust Blue policies in the CyberSelfPlay environment.  
We experiment across **SFT + GRPO baselines**, **reward smoothing**, **diversity shaping**, and **league-based RL**.

---

### 📊 Overview of Training Methods

| Method | Description | Colab | Metrics / Curves |
|--------|------------|-------|------------------|
| **🔹 GRPO (Single-Policy RL)** ||||
| **SFT → GRPO (Vanilla)** | Baseline using only environment reward | [Open](https://colab.research.google.com/drive/1K5771KT0-2lyU6eNghqQEStBS4OSF7D7?usp=sharing) | <img src="https://res.cloudinary.com/dp1ejt3eb/image/upload/v1777187892/SFT_GRPO_Vanilla_i88mbr.png" width="350"/>|
| **SFT → GRPO (Anti-Collapse)** | Adds diversity penalty to avoid mode collapse | [Open](https://colab.research.google.com/drive/1HivyWte1q-sugE04XsyMi1U_RY1oGkJ8?usp=sharing) | <img src="https://res.cloudinary.com/dp1ejt3eb/image/upload/v1777188452/SFT_GRPO_Anti-Collapse_Regularization_fq3mgo.png" width="350"/> |
| **🔹 League (Multi-Policy RL)** ||||
| **League (PFSP)** | Prioritized Fictitious Self-Play for opponent sampling | [Open](https://colab.research.google.com/drive/1mDk9pzeRudjmXhU0VBVJymqF5An8bHhk?usp=sharing) | Win-rate curves |
| **League (PSRO)** | Policy-Space Response Oracles (game-theoretic updates) | [Open](https://colab.research.google.com/drive/1O6IoE-_UloAeDXKve2ZA1W4OajychglP?usp=sharing) | <img src="https://res.cloudinary.com/dp1ejt3eb/image/upload/v1777188537/League_PSRO_wd3esy.png" width="350"/> |
| **League (PFSP + PSRO)** | Combines adaptive sampling + meta-policy optimization | [Open](https://colab.research.google.com/drive/1OaOQYmoq2ni2FjCUukBkpt3BpT55uhX9?usp=sharing) | Meta + Reward curves |

---

## 📐 Mathematical Formulation

### 1. GRPO (Vanilla)

$$
\mathcal{L}*{\text{GRPO}} = \mathbb{E}\left[\log \pi*\theta(a_i \mid s),(r_i - \bar{r})\right]
$$

$$
\bar{r} = \frac{1}{N}\sum_{i=1}^{N} r_i
$$

---

### 2. GRPO + Regularization (Anti-Collapse)

$$
r_i' = r_i - \lambda ,\max\big(0,; p(a_i) - \tau\big)
$$

where:
$$
\begin{aligned}
p(a_i) &= \text{frequency of action in batch} \
\tau &= \text{threshold}
\end{aligned}
$$

---

### 3. PFSP (Prioritized Fictitious Self-Play)

$$
p_j \propto f(w_j), \qquad f(w) = w(1 - w)
$$

where:
$$
\begin{aligned}
w_j &= \text{win-rate against opponent } j
\end{aligned}
$$

---

### 4. PSRO (Policy-Space Response Oracles)

$$
p_i' \propto p_i \left(1 + \eta (u_i - \bar{u}) \right)
$$

$$
\bar{u} = \sum_i p_i u_i
$$

where:
$$
\begin{aligned}
u_i &= \text{utility of policy } i \\
\eta &= \text{learning rate}
\end{aligned}
$$

---

### 5. PFSP + PSRO (Combined)

$$
p_j \propto f(w_j), \qquad f(w) = w(1 - w)
$$

$$
p_i' \propto p_i \left(1 + \eta (u_i - \bar{u}) \right)
$$

Combines opponent sampling (PFSP) with meta-policy updates (PSRO).


## Core Optimization Math

### SFT (Supervised Fine-Tuning)

Token-level cross-entropy (negative log-likelihood) on expert trajectories.

### GRPO (Group Relative Policy Optimization)

For prompt $x$, sample a group of completions:

$$
\{y^{(1)},\ldots,y^{(G)}\} \sim \pi_\theta(\cdot\mid x)
$$

Score each completion with reward $R^{(j)}$, compute group-relative advantages, and update policy parameters with optional KL regularization toward a reference policy $\pi_{\text{ref}}$.

---

## Long-Horizon Scenario Scale

| scenario | turns | instructions | checkpoint stride |
| --- | ---: | ---: | ---: |
| small | 60 | 40 | 8 |
| medium | 100 | 120 | 12 |
| large | 180 | 300 | 20 |

Instruction progress and violation signals are tracked in environment metadata.

---

## Results Summary

Across training runs, Blue policies generally move from imitation-only behavior (SFT) to stronger environment-aligned behavior after GRPO. In league mode, round-level opponent selection (PFSP / PSRO / mix) changes pressure distribution and produces distinct multi-round learning dynamics.

Common result artifacts produced by the training scripts include:

- `training_curves.png`
- `log_history.json`
- `train_metrics.log`
- `per_step_rewards.jsonl`
- per-step curve images under `curves/`
- league-specific outputs such as `training_curves_all_rounds.png`, `league_state.jsonl`, and `log_history_combined.json`

---

## Why It Matters

- **Security operations relevance:** models long-horizon defense decisions closer to real incident response.
- **Research relevance:** provides a reproducible adversarial benchmark for instruction-following under uncertainty.
- **Evaluation relevance:** combines environment dynamics, tool-structured actions, and measurable outcomes.

---

## Abbreviations

| Short form | Full form |
| --- | --- |
| SFT | Supervised Fine-Tuning |
| GRPO | Group Relative Policy Optimization |
| TRL | Transformers Reinforcement Learning |
| LoRA | Low-Rank Adaptation |
| PFSP | Prioritized Fictitious Self-Play |
| PSRO | Policy-Space Response Orbit |
| POSG | Partially Observable Stochastic Game |
| POMDP | Partially Observable Markov Decision Process |
| MTTD | Mean Time To Detect |
| MTTR | Mean Time To Repair |

---

## Project Structure (high level)

```text
cyber_selfplay/
├── cyber_selfplay_env/       # environment core, simulator, rubrics, metrics
├── server/                   # OpenEnv API server
├── train/
│   ├── kaggle_grpo.py
│   ├── kaggle_grpo_league.py
│   ├── pfsp.py
│   └── psro_meta.py
└── openenv.yaml
```

---

## References

- Vinyals et al., *Nature* 2019 — AlphaStar / league training  
- Lanctot et al., *NeurIPS* 2017 — PSRO  
- Hu et al., *ACM Transactions on Privacy and Security* (TOPS), 2021 — cyber defense POMDP  
- TTCP CAGE-2 — defender POMDP framing  
- Hugging Face TRL documentation (`GRPOTrainer`)