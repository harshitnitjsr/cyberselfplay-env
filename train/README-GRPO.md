# SFT + GRPO — Blue defender policy (primary training path)

This document is the **reference** for the **SFT + GRPO** track: supervised warm-start, then **group-relative** on-policy RL on **environment** rewards. Entry points:

| File | When to use |
| --- | --- |
| `train/kaggle_grpo.py` | Kaggle: one cell, SFT + single GRPO phase (fastest default) |
| `train/kaggle_grpo_league.py` | Kaggle: **SFT + league rounds** — **PFSP** / **PSRO** / **mix** opponent pick (`OPP_SAMPLE_MODE`), **PSRO** replicator on heuristic payoffs, **mini-GRPO** per round, aligned Red preamble; `HF_SPACE_CLONE` env to point at your Space |
| `train/grpo_space.py` | Hugging Face Space / Docker: `ENV`-driven configuration |

The **same** repository also ships **league / PFSP / PSRO** utilities (`train/pfsp.py`, `train/psro_meta.py`, `run_demo.py`, `colab_trl_selfplay.py` with league flags) for **population-based** training. Those modules are **independent** entry points: you can run SFT+GRPO alone, run league demos alone, or later **compose** a GRPO checkpoint into a league pool.

---

## 1. What the pipeline does (two phases)

1. **Supervised fine-tuning (SFT)**  
   The model imitates a **diverse, state-conditional** heuristic on real `CyberSelfPlay` rollouts so it learns:
   - valid **JSON** `CyberAction` output;
   - use of the **13 Blue** tools (balanced data, not a single default);
   - **identical** prompt formatting in train and inference via `tokenizer.apply_chat_template`.

2. **Group relative policy optimization (GRPO, TRL)**  
   The **same** LoRA policy is updated **online**: for each prompt, the trainer samples **several** completions, assigns a **scalar** to each, and moves probability toward **higher** outcomes **relative to the same group** (advantages within the group). The **main** term is **`env.step(CyberAction).reward`** (after a short scripted Red pre-step, as implemented). **Additional** terms encourage **diversity** (penalty when one tool floods a group) and gently discourage overuse of `execute_instruction` if SFT was skewed. **Unparseable** output gets a **fixed** negative reward.

**Why this design:** SFT makes **format and tool coverage** reliable on small LMs; GRPO then **refines** behavior using the **rubric** in the env, not hand-written string heuristics for “good JSON.”

---

## 2. Core math (at a glance)

- **SFT:** cross-entropy (negative log-likelihood) on expert tokens on packed post-template text.
- **GRPO:** for each prompt \(x\), draw \(\{y^{(1)},\ldots,y^{(G)}\} \sim \pi_\theta(\cdot\mid x)\), score each \(R^{(j)}\) (env + small regularizer terms), build **group-relative** advantages, policy-gradient step, usually with **KL** toward \(\pi_{\text{ref}}\) (SFT) — `beta` in `GRPOConfig`.

*Intuition:* for each prompt, what gets reinforced is “**better than the other samples you just drew**.”

---

## 3. Reward composition (aligned in `kaggle_grpo.py` and `grpo_space.py`)

| Component | Role |
| --- | --- |
| **Environment** | `float(obs.reward)` after `env.step(blue_action)` — `blue_reward` in `cyber_selfplay_env/rubrics.py` (containment, triage, instruction progress, checkpoints, etc.). |
| **Unparseable output** | Fixed negative score so invalid strings are never optimal. |
| **In-group tool dominance** | If one `tool_name` is more than half of **valid** parses in that reward call, a **ramp** penalty reduces reward — stabilizes **exploration** across tools. |
| **Frequent `execute_instruction`** | Small extra nudge if that tool was over-represented in SFT. |

The **dominant** signal remains **environmental**; the extras are **stability** terms for **group** training.

---

## 4. Reproduction

### Kaggle

1. Open a GPU notebook, paste `train/kaggle_grpo.py` (single cell) or use it as a script source.
2. Set clone URL, `BASE_MODEL`, and Hub variables as needed.
3. Outputs: `outputs_cyber/curves/step_*.png`, `train_metrics.log`, `per_step_rewards.jsonl`, `log_history.json`, LoRA under `outputs_cyber/cyber-blue-grpo-lora/`.

### Hugging Face Space

- Root `Dockerfile` + `TRAIN_*` / SFT / GRPO env variables; `train/grpo_space.py` is the in-container trainer.

---

## 5. Artifacts useful for review

- **`curves/`** — one PNG per logging step: per-completion **reward scatter** and **unique tool** counts.
- **`train_metrics.log`**, **`per_step_rewards.jsonl`**
- **Final** `training_curves.png` + `log_history.json`
- **LoRA** directory + optional Hub upload

---

## 6. How this addresses **Theme 2** and **Theme 4**

### Theme 2 — *Long-horizon reasoning and instruction following*

- The **environment** is built for long missions, **partial** observations, and tools like **`execute_instruction`** that connect to a **playbook** of required actions (`simulator.py`, `rubrics.py`).
- **Blue** rewards include **instruction progress**, **checkpoints**, and **violation** costs — the scalar GRPO uses is **aligned** with “doing the right enterprise actions,” not only JSON validity.
- SFT+GRPO trains a **single** model to **map** an observation to a **structured** action the env **executes and scores**; the **horizon** of full multi-step episodes is defined by the **env**; GRPO, as implemented, **scores** a **step** in a fresh rollout (a common scalable pattern; **multi-step** GRPO rollouts are a natural **extension**).

### Theme 4 — *Self-improvement and co-evolution*

This path **directly** supports Theme 4 as **on-policy self-improvement**:

1. **After SFT, the policy is not frozen.** GRPO **resamples** from the **current** \(\pi_\theta\), evaluates **stochastic** simulator outcomes, and **updates** \(\theta\). The agent **improves from its own** generations and **environment** feedback.

2. **Group-relative** training compares **several** completions for the **same** prompt. That is **self-comparison** and **self-play in policy space** (branches of the same network on the same state), a standard way to get **learning signal** without a separate second network.

3. **Stochastic** Red-side **dynamics** in the simulator (success/failure, detections) mean the **Blue** policy faces a **varying** effective opponent **distribution**; **adaptation** is toward **return** under that process.

4. **Same codebase, broader Theme 4:** **PFSP** / **PSRO** / **league** scripts are available for **explicit** **multi-policy** co-evolution. A **GRPO** checkpoint can be **dropped in** as one **Blue** in a **pool** for a **second** **experimental** **phase** — the two tracks are **complementary** **layers** in one project.

**One-sentence blurb:**  
*SFT+GRPO delivers **Theme 2** (instruction-aware, env-scored actions) and **Theme 4** (**online** self-improvement of the Blue LLM from **its own** rollouts in a **stochastic** cyber sim); the repo **also** offers **league** / **PFSP** / **PSRO** for **extended** **population** **studies**.*

---

## 7. File map

| File | Role |
| --- | --- |
| `kaggle_grpo.py` | Kaggle: full training + **live** plots + file logs |
| `grpo_space.py` | Space / Docker: **env-var** **config** |
| `../requirements-train.txt` | `trl`, `unsloth`, `bitsandbytes`, … |
| `pfsp.py`, `psro_meta.py`, `colab_trl_selfplay.py`, `evaluate_league.py` | League / PFSP / PSRO — optional **second** track in the same repo; you can plug a GRPO checkpoint into a league pool when you extend the experiment. |

---

## 8. References (conceptual)

- TRL `GRPOTrainer` — **group** sampling, **group-relative** objectives.  
- CyberSelfPlay env and POSG view: `../cyber_selfplay_env/`, `../README.md`.  
- League / PFSP / PSRO: `../README.md` **§6** and `run_demo.py`.
