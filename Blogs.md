# CyberSelfPlay: Building a Long-Horizon Cyber Defense Environment

## Why this environment exists

Most agent benchmarks are short-horizon and mostly single-agent. Real cyber defense is neither:

- decisions unfold over many steps,
- observations are partial and noisy,
- an attacker adapts while the defender is acting,
- success is not one move, but sustained containment and recovery.

CyberSelfPlay was built to model this gap directly: a stochastic Red-vs-Blue world where Blue must execute mission playbooks while Red applies adversarial pressure.

What makes this direction different is that we do not treat “good defense” as a single yes/no check. The agent must keep making good choices for many steps in a row, under changing pressure, while mission goals are still active. That combination (long horizon + partial visibility + active adversary + mission constraints) is where many current benchmarks become too easy or unrealistic.

---

## What the environment is

CyberSelfPlay is an OpenEnv-compatible two-player environment:

- **Red** (attacker) and **Blue** (defender) act in the same world state.
- Blue receives partial state + mission context + progress metadata.
- Blue outputs structured `CyberAction` JSON actions through tool APIs.
- Rewards encode both security outcomes (detect/contain/recover) and mission outcomes (instruction progress/checkpoints/violations).

Formally, the environment is modeled as a partially observable stochastic game (POSG):

$$
\mathcal{G}=\langle \mathcal{S},\mathcal{A}_R,\mathcal{A}_B,\mathcal{O}_R,\mathcal{O}_B,T,Z_R,Z_B,r_R,r_B,\gamma \rangle
$$

with objective

$$
J_i(\pi_i,\pi_{-i})=\mathbb{E}\left[\sum_{t=0}^{H}\gamma^t r_i\left(s_t,a_t^R,a_t^B\right)\right],\quad i\in\{R,B\}.
$$

Near-zero-sum coupling is represented as:

$$
r_B=-r_R-\lambda C_{\mathrm{collateral}}.
$$

In plain terms: if Red gains ground, Blue usually loses ground, and harmful side effects are also counted. This keeps the game honest and closer to what defenders face in real systems.

---

## Core components

At a high level, the system has:

1. **Environment core** (`cyber_selfplay_env/`)
   - hidden-state simulator and transitions,
   - reward rubrics,
   - metrics and progress tracking,
   - scenario definitions and tool interfaces.
2. **API server** (`server/app.py`)
   - OpenEnv endpoints for interaction.
3. **Training scripts** (`train/`)
   - `kaggle_grpo.py` (single-policy SFT -> GRPO),
   - `kaggle_grpo_league.py` (SFT -> league rounds + mini-GRPO + PFSP/PSRO updates).

Together, these parts create a full loop: simulate attack/defense interactions, score behavior with mission-aware rewards, then improve the policy using those outcomes.

---

## Observations, actions, rewards

### Observations

Blue observes partial environment state and mission/task context, including progress-related metadata.

### Actions

Blue emits `CyberAction` JSON tool calls. Red has its own attacker actions in the same timeline.

### Reward design

The reward law combines dense and delayed signals. The scripts use environment reward as the dominant term, with additional shaping in GRPO training.

Red side:

$$
\begin{aligned}
r_R &= w_1 \mathbb{1}_{\mathrm{foothold}} + w_2 \mathbb{1}_{\mathrm{priv}} + w_3 \mathbb{1}_{\mathrm{lateral}} + w_4 \mathbb{1}_{\mathrm{exfil}} \\
&\quad - w_5 \mathbb{1}_{\mathrm{detect}} + w_6 \mathbb{1}_{\mathrm{plan\_sabotage}} - \eta_R
\end{aligned}
$$

Blue side:

$$
\begin{aligned}
r_B &= v_1 \mathbb{1}_{\mathrm{detect}} + v_2 \mathbb{1}_{\mathrm{contain}} + v_3 \mathbb{1}_{\mathrm{recover}} - v_4 \mathbb{1}_{\mathrm{exfil}} \\
&\quad + v_5 \mathbb{1}_{\mathrm{instr\_progress}} + v_6 \mathbb{1}_{\mathrm{checkpoint}} - v_7 \mathbb{1}_{\mathrm{instr\_violation}} \\
&\quad + v_8 \rho_{\mathrm{inst}} - \eta_B
\end{aligned}
$$

Why this matters: these equations make it hard to “game” the benchmark with shallow tricks. The agent is pushed toward useful defense behavior across time, not just short-term score spikes.

---

## Training story: what we tried, what failed, what changed

### Step 1: Start with SFT -> vanilla GRPO

We started with the direct path:

- Supervised Fine-Tuning (SFT) on heuristic rollouts,
- then Group Relative Policy Optimization (GRPO) using environment reward.

Core GRPO intuition:

$$
\{y^{(1)},\ldots,y^{(G)}\}\sim \pi_\theta(\cdot\mid x)
$$

Sample a group of completions per prompt, score each completion, compute group-relative advantages, and update policy parameters.

At this point, we had a baseline that could parse actions and follow structure, but behavior quality still depended heavily on exploration quality.

### Step 2: We hit mode collapse pressure

During early iterations, action diversity degraded: one tool could dominate generated actions. This reduces effective exploration and hurts credit assignment.

In practice, this looked like “safe but repetitive” behavior: valid JSON, but less tactical variety. The agent was syntactically correct more often than strategically useful.

### Step 3: Add stabilization in single-policy GRPO

In `kaggle_grpo.py`, we introduced shaping aligned with this issue:

- group-level diversity penalty when one tool dominates a batch,
- additional nudge against overusing `execute_instruction` when SFT bias is high,
- continued logging of unique-tools-per-step as a diversity meter.

This improved robustness of training dynamics while keeping environment reward primary.

This step is important because it addresses a common failure mode in small-model RL: if action variety collapses too early, learning plateaus quickly.

### Step 4: Move to league training for broader robustness

Single-policy GRPO improved behavior, but robustness against varied attacker styles needed stronger pressure. We moved to `kaggle_grpo_league.py`:

- run multiple league rounds,
- pick Red archetypes using PFSP / PSRO / mix,
- run mini-GRPO each round,
- update meta-probabilities with replicator dynamics.

PFSP weighting:

$$
p_j \propto f(w_j),\qquad f(w)=w(1-w)
$$

PSRO-style replicator update:

$$
p_i' \propto p_i\left(1+\eta(u_i-\bar{u})\right),\qquad \bar{u}=\sum_i p_i u_i
$$

This made training pressure adaptive across rounds rather than fixed to one opponent profile.

This is where behavior starts to look more “field-like”: the defender is not tuned to one attacker template, but pushed by a moving mixture of attacker styles.

### Step 5: Turn logs into evidence, not just numbers

We deliberately kept artifact generation rich (`training_curves.png`, per-step JSONL logs, combined league histories) so claims can be traced back to concrete run outputs. That makes debugging, comparison, and review much more grounded.

---

## Results and evidence

Across runs, we observe the expected pattern:

- Blue moves from imitation-only behavior (SFT) to stronger reward-aligned behavior after GRPO.
- Diversity shaping reduces collapse and stabilizes learning in single-policy training.
- League mode (PFSP/PSRO/mix) produces richer multi-round dynamics and better robustness across opponent types.

A useful way to read these results is:

1. **Can the model act in valid structured form?** (SFT gives this base)
2. **Can it improve through interaction feedback?** (GRPO gives this climb)
3. **Can it hold up under varied opponents?** (league rounds test this directly)

By the final stage, improvements are not only in average reward but also in consistency across rounds and opponent profiles.

Primary artifacts produced by the training scripts:

- `training_curves.png`
- `log_history.json`
- `train_metrics.log`
- `per_step_rewards.jsonl`
- per-step curves under `curves/`
- league-specific: `training_curves_all_rounds.png`, `league_state.jsonl`, `log_history_combined.json`

These files are the evidence trail for reward trends, variance, action diversity, and round-by-round league behavior.

---

## Why it matters

CyberSelfPlay matters because it evaluates what real defenders need:

- long-horizon, instruction-conditioned recovery,
- adversarial interaction under uncertainty,
- measurable progress beyond one-step task completion.

For practitioners, it is closer to incident response realities.  
For researchers, it offers a reproducible testbed for strategic, multi-step agent behavior.

For teams building defensive copilots or autonomous responders, this kind of environment gives a safer place to test policy behavior before production deployment.  
For evaluation-focused work, it provides a bridge between toy tasks and operationally meaningful multi-step scenarios.

---

## Why this submission can stand out

- It tackles a hard setting that combines long horizon, partial observability, adversarial play, and mission objectives in one benchmark.
- It does not stop at one training recipe; it shows a full progression from baseline to stabilized training to league pressure.
- It includes mathematical grounding, system-level structure, and artifact-level evidence in one coherent package.
- The narrative from “initial approach -> failure mode -> fix -> stronger method” is explicit and reproducible.

---

## Environment link

- Hugging Face Space: `https://huggingface.co/spaces/HarshitShri026`
