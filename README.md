---
title: CyberSelfPlay (Long-Horizon Cyber POSG)
emoji: "🛡️"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7870
pinned: false
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

- $\mathcal{S}$ is the hidden environment state space.
- $\mathcal{A}_R,\mathcal{A}_B$ are Red and Blue action spaces.
- $\mathcal{O}_R,\mathcal{O}_B$ are Red and Blue observation spaces.
- $T$ is the state-transition kernel.
- $Z_R,Z_B$ are observation emission models for each player.
- $r_R,r_B$ are Red and Blue reward functions.
- $\gamma$ is the discount factor.

with objective:

$$
J_i(\pi_i,\pi_{-i})=\mathbb{E}\left[\sum_{t=0}^{H}\gamma^t r_i\left(s_t,a_t^{R},a_t^{B}\right)\right],\quad i\in\{R,B\}
$$

with:

- $\pi_i$ the policy of player $i$ and $\pi_{-i}$ the opponent policy.
- $H$ the episode horizon (maximum time steps).
- $s_t$ the state at time $t$.
- $a_t^{R},a_t^{B}$ the Red and Blue actions at time $t$.
- $r_i(\cdot)$ the reward received by player $i$.

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

```mermaid
graph TD
    subgraph AgentLayer["Agent Layer"]
        RED["Red Policy π_R"]
        BLUE["Blue Policy π_B"]
    end

    subgraph Interface["OpenEnv Interface"]
        API["FastAPI / OpenEnv step()"]
        ACT["Action Validator + CyberAction Parser"]
    end

    subgraph Core["Environment Core"]
        SIM["State Transition Simulator T(s,a_R,a_B)"]
        OBSGEN["Observation Emission Z_R, Z_B"]
        RUB["Reward Rubrics r_R, r_B"]
        MET["Mission Metrics\n(ρ_inst, ν_inst, checkpoints)"]
        TERM["Termination / Truncation Logic"]
    end

    RED -->|"a_t^R"| API
    BLUE -->|"a_t^B"| API
    API --> ACT
    ACT --> SIM

    SIM --> OBSGEN
    SIM --> RUB
    SIM --> MET
    SIM --> TERM

    OBSGEN -->|"o_t^R"| RED
    OBSGEN -->|"o_t^B"| BLUE
    RUB -->|"r_t^R"| RED
    RUB -->|"r_t^B"| BLUE
    MET -->|"metadata / posg_metrics"| BLUE
    TERM -->|"done, truncated"| API
```

## Training Flow

```mermaid
sequenceDiagram
    participant M as Base Model + Tokenizer
    participant D as SFT Dataset Builder
    participant T as Trainer (GRPO)
    participant L as League Scheduler
    participant E as CyberSelfPlay Env
    participant R as Result Logger

    M->>D: Build SFT dataset from heuristic Blue rollouts
    D->>T: Initialize policy via SFT optimization

    alt Single-policy pipeline (kaggle_grpo.py)
        loop Optimization step
            T->>T: Sample G completions per prompt
            T->>E: Parse actions + env.step(...)
            E-->>T: rewards, observations, metadata
            T->>T: Group-relative advantage + KL-regularized update
            T->>R: Append step metrics and curves
        end
    else League pipeline (kaggle_grpo_league.py)
        loop League round
            L->>L: Select opponent (PFSP / PSRO / mix)
            loop Round optimization step
                T->>E: Mini-GRPO rollout and reward collection
                E-->>T: round rewards and trajectories
                T->>T: Policy update
                T->>R: Round step metrics and curves
            end
            L->>L: Replicator meta update on payoff estimates
        end
    end

    R->>R: Export artifacts (training_curves, log_history, per_step_rewards, league_state)
```

---

## 🚀 Training Approaches in This Project

This project explores multiple training strategies for learning robust Blue policies in the CyberSelfPlay environment.  
We experiment across **SFT + GRPO baselines**, **reward smoothing**, **diversity shaping**, and **league-based RL**.

---

### 📊 Overview of Training Methods

| Method | Description | Colab | Final Reward | Curves |
|--------|-------------|-------|--------------|--------|
| **🔹 GRPO (Single-Policy RL)** | |  |  |  |
| **SFT → GRPO (Vanilla)** | Baseline: environment reward only | [Open](#) | Reward ↑, stable parsing (~100%), but mode collapse observed | Reward vs Step, Loss |
| **SFT → GRPO (Regularization)** | Penalizes repeated actions (anti-collapse) | [Open](#) | Higher action diversity, reduced collapse, more stable reward | Reward + Diversity |
| **🔹 League (Multi-Policy RL)** |  |  |  |  |
| **League (PFSP)** | Prioritized opponent sampling | [Open](#) | Improved robustness, better win-rate vs diverse opponents | Win-rate curves |
| **League (PSRO)** | Meta-policy updates (game-theoretic) | [Open](#) | Reduced exploitability, adaptive strategies | Meta-policy weights |
| **League (PFSP + PSRO)** | Combines PFSP + PSRO | [Open](#) | Best overall: stability + robustness + diversity | Meta + reward curves |


---

## 📐 Mathematical Formulation (Aligned with Methods)

### 1. GRPO (Vanilla)

\[
\mathcal{L}_{GRPO} = \mathbb{E}\left[\log \pi_\theta(a_i|s)\cdot (r_i - \bar{r})\right]
\]

\[
\bar{r} = \frac{1}{N}\sum_{i=1}^{N} r_i
\]

---

### 2. GRPO + Regularization (Anti-Collapse)

\[
r_i' = r_i - \lambda \cdot \max(0, p(a_i) - \tau)
\]

where:
- \( p(a_i) \) = frequency of action in batch  
- \( \tau \) = threshold  

---

### 3. PFSP (Prioritized Fictitious Self-Play)

\[
p_j \propto f(w_j), \quad f(w) = w(1 - w)
\]

where:
- \( w_j \) = win-rate against opponent \( j \)

---

### 4. PSRO (Policy-Space Response Oracles)

\[
p_i' \propto p_i \left(1 + \eta (u_i - \bar{u}) \right)
\]

\[
\bar{u} = \sum_i p_i u_i
\]

where:
- \( u_i \) = utility of policy \( i \)  
- \( \eta \) = learning rate  

---

### 5. PFSP + PSRO (Combined)

\[
p_j \propto f(w_j), \quad f(w) = w(1 - w)
\]

\[
p_i' \propto p_i \left(1 + \eta (u_i - \bar{u}) \right)
\]

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
