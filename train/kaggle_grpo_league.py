# =============================================================================
# CyberSelfPlay — Kaggle: SFT + League (PFSP/PSRO) + GRPO  (full merge, single cell)
#
#   Phase 1: same SFT as kaggle_grpo.py
#   Phase 2: for each league round (LEAGUE_ROUNDS):
#       • PFSP sample Red archetype (R-easy / R-mid / R-hard) from a pool
#       • build GRPO prompts with that Red’s rollout (exploit count range)
#       • compute_rewards uses the SAME profile (aligned env preamble)
#       • mini-GRPO: GRPO_STEPS_PER_ROUND steps, same LoRA
#       • update pool win-rates; PSRO: replicator on heuristic-eval payoffs
#
# Slower than kaggle_grpo.py — reduce LEAGUE_ROUNDS or GRPO_STEPS_PER_ROUND.
# If train.pfsp import fails, fallbacks are inlined.
# =============================================================================

# ---------- 1) Install Unsloth + TRL + OpenEnv deps ----------
get_ipython().system('pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
get_ipython().system('pip install -q --upgrade git+https://github.com/huggingface/trl.git')
get_ipython().system('pip install -q sympy scipy fastapi uvicorn datasets pydantic openenv-core huggingface_hub matplotlib')

# ---------- 2) Clone your deployed Hugging Face Space (edit URL if you fork) ----------
import os, sys, shutil
HF_SPACE_CLONE = os.environ.get(
    "HF_SPACE_CLONE",
    "https://huggingface.co/spaces/HarshitShri026/cyberselfplay-env",
)
os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/cyberselfplay-env", ignore_errors=True)
get_ipython().system("git clone " + HF_SPACE_CLONE)
os.chdir("/kaggle/working/cyberselfplay-env")
sys.path.insert(0, "/kaggle/working/cyberselfplay-env")
get_ipython().system("pip install -q -e .")

# ---------- 3) Wipe any prior checkpoint ----------
shutil.rmtree("/kaggle/working/cyberselfplay-env/outputs_cyber", ignore_errors=True)

# ---------- 4) Silence noisy warnings ----------
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("trl").setLevel(logging.WARNING)

# ---------- 5) Imports (Unsloth FIRST) ----------
import unsloth
import torch, json, random, re
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

from cyber_selfplay_env.environment import CyberSelfPlayEnvironment
from cyber_selfplay_env.models import CyberAction
from cyber_selfplay_env.tools_blue import BLUE_TOOLS

try:
    from train.pfsp import OpponentStats, sample_opponent, pfsp_weight
    from train.psro_meta import replicator_update, normalize as psro_normalize
except Exception:  # noqa: BLE001
    from dataclasses import dataclass

    @dataclass
    class OpponentStats:
        name: str
        win_rate_vs_learner: float
        games: int = 0

    def pfsp_weight(w: float) -> float:
        w = min(1.0, max(0.0, w))
        return w * (1.0 - w)

    def sample_opponent(opponents):
        pool = list(opponents)
        weights = [pfsp_weight(o.win_rate_vs_learner) for o in pool]
        t = sum(weights)
        if t <= 0:
            return random.choice(pool)
        return random.choices(pool, weights=weights, k=1)[0]

    def psro_normalize(xs):
        s = sum(max(0.0, x) for x in xs)
        if s <= 0:
            return [1.0 / len(xs)] * len(xs)
        return [max(0.0, x) / s for x in xs]

    def replicator_update(payoff_row, current, eta=0.2):
        u_bar = sum(p * u for p, u in zip(current, payoff_row))
        nxt = [p * (1.0 + eta * (u - u_bar)) for p, u in zip(current, payoff_row)]
        return psro_normalize(nxt)

# ---------- 6) Env vars ----------
os.environ["TRANSFORMERS_VERBOSITY"]    = "error"
os.environ["TOKENIZERS_PARALLELISM"]    = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["WANDB_DISABLED"]            = "true"
os.environ["BITSANDBYTES_NOWELCOME"]    = "1"
if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Kaggle T4 x2 detected -- using GPU 0 only (Unsloth single-GPU).")

# ---------- 7) Precision ----------
has_cuda = torch.cuda.is_available()
use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
use_fp16 = has_cuda and not use_bf16
print(f"CUDA: {has_cuda} | bf16: {use_bf16} | fp16: {use_fp16}")
print(f"GPU: {torch.cuda.get_device_name(0) if has_cuda else 'none'}")

# ---------- 8) Model load ----------
MAX_SEQ_LEN = 1024
LORA_RANK   = 16
BASE_MODEL  = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
tokenizer.model_max_length = MAX_SEQ_LEN
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
)

# Clear `generation_config.max_length` so it does NOT collide with `max_new_tokens`
# at every `model.generate(...)` call (silences the spammy transformers warning).
if getattr(model, "generation_config", None) is not None:
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = None

IS_LLAMA = "llama" in BASE_MODEL.lower()
IS_QWEN  = "qwen"  in BASE_MODEL.lower()

# ---------- 9) Prompt builder (single source of truth) + JSON parser ----------
# Both SFT data and GRPO inference go through tokenizer.apply_chat_template
# so the formats are byte-identical. NO hand-written prompt strings anywhere.
VALID_BLUE = list(BLUE_TOOLS)

SYSTEM_MSG = (
    "You are the Blue defender in a cyber self-play game.\n"
    "You MUST output ONLY a valid JSON object.\n"
    "No explanation. No extra text. No prefixes/suffixes.\n"
    "Output exactly ONE JSON object and stop.\n\n"
    f"Schema: {{\"actor\":\"blue\",\"tool_name\":<one of {sorted(VALID_BLUE)}>,"
    "\"target\":\"host-XX\",\"params\":{},\"rationale\":\"short\"}}"
)

def _user_msg(obs_dict: dict) -> str:
    return (
        f"Observation:\n{json.dumps(obs_dict, ensure_ascii=True)}\n\n"
        "Reply with ONE JSON line ending with '}'. Nothing else."
    )

def obs_to_prompt(obs_dict: dict) -> str:
    """Inference prompt. Same chat template SFT used, with assistant header appended."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": _user_msg(obs_dict)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_action(text: str):
    matches = JSON_RE.findall(text)
    if not matches:
        return None, False
    
    for m in matches:
        try:
            a = json.loads(m)
            if a.get("tool_name") in VALID_BLUE:
                a["actor"] = "blue"
                a.setdefault("target", "host-00")
                a.setdefault("params", {})
                a.setdefault("rationale", "grpo")
                return a, True
        except:
            continue
    
    return None, False

# =============================================================================
# PHASE 1 -- SFT warm-start (imitate heuristic Blue policy)
# =============================================================================

# ---------- 10) Heuristic Blue policy (lifted from train/colab_trl_selfplay.py) ----------
# Real catalog of `required_tool` values used by the env's instruction system
# (see cyber_selfplay_env/simulator.py::_build_instructions). Picking from this
# list means execute_instruction has a real chance (1/8) of matching.
INSTRUCTION_TOOLS = [
    "triage_alerts", "isolate_host", "deploy_patch", "rotate_secrets",
    "run_forensics", "restore_backup", "harden_policy", "publish_ioc_blocklist",
]

RATIONALE = {
    "query_siem":            "scan telemetry",
    "triage_alerts":         "investigate alert",
    "isolate_host":          "contain breach",
    "disable_account":       "lock compromised user",
    "rotate_secrets":        "remove persistence",
    "deploy_patch":          "harden vulnerable host",
    "harden_policy":         "tighten controls",
    "restore_backup":        "recover service",
    "run_forensics":         "investigate host",
    "publish_ioc_blocklist": "block known IOCs",
    "execute_instruction":   "follow playbook",
    "checkpoint_plan":       "track progress",
    "reconcile_state":       "stabilize state",
}

EPSILON_RANDOM = 0.40  # 40% pure exploration in SFT data => corpus covers all
                       # 13 tools roughly evenly, killing the "default action"
                       # bias that triggers post-SFT mode collapse.

def heuristic_blue_action(public_state: dict, telemetry: list, t_step: int) -> dict:
    """State-conditional, diverse Blue policy.

    The previous version only ever fired ~3 branches and ~80% of the data
    became `execute_instruction`, which caused SFT mode collapse. This version
    builds a *weighted candidate set* from the current observation and samples
    one, plus a 20% epsilon-random arm. Result: the SFT corpus exercises all
    13 Blue tools with realistic targets, so the model learns the schema
    BROADLY and GRPO has real diversity to do credit assignment on.
    """
    detections = telemetry or public_state.get("detections", []) or []
    known      = int(public_state.get("known_incident_count", 0) or 0)
    instr      = public_state.get("instruction_progress", {}) or {}
    instr_pending = isinstance(instr, dict) and instr.get("completed", 0) < instr.get("total", 1)

    rand_host = lambda: f"host-{random.randint(0, 5):02d}"

    # 20% pure exploration: random valid tool with a syntactically correct payload.
    if random.random() < EPSILON_RANDOM:
        tool = random.choice(VALID_BLUE)
        if tool == "execute_instruction":
            target, params = "", {"required_tool": random.choice(INSTRUCTION_TOOLS)}
        elif tool == "disable_account":
            target, params = f"user-{random.randint(0, 3):02d}", {}
        else:
            target, params = rand_host(), {}
        return {"actor": "blue", "tool_name": tool, "target": target,
                "params": params, "rationale": RATIONALE.get(tool, "explore")}

    # Build (tool, target, params, weight) candidates from the observation.
    cands: list[tuple[str, str, dict, int]] = []

    # Always-valid baseline (covers tools that always parse but pay little).
    cands += [
        ("query_siem",       rand_host(), {}, 1),
        ("checkpoint_plan",  rand_host(), {}, 1),
        ("reconcile_state",  rand_host(), {}, 1),
        ("deploy_patch",     rand_host(), {}, 1),
        ("harden_policy",    rand_host(), {}, 1),
        ("publish_ioc_blocklist", rand_host(), {}, 1),
        ("disable_account",  f"user-{random.randint(0, 3):02d}", {}, 1),
    ]

    if detections:
        cands += [("triage_alerts", rand_host(), {}, 3)]   # +2.0 reward when valid

    if known > 0:
        cands += [
            ("isolate_host",   rand_host(), {}, 3),         # +3.0
            ("rotate_secrets", rand_host(), {}, 2),         # +4.0
            ("run_forensics",  rand_host(), {}, 2),
            ("restore_backup", rand_host(), {}, 1),
        ]

    if instr_pending:
        # weight=1 (was 2) -- execute_instruction was over-represented and caused
        # the model to memorize it as the "safe default" action.
        cands += [(
            "execute_instruction", "",
            {"required_tool": random.choice(INSTRUCTION_TOOLS)},
            1,                                              # +1.2 if right, -1.0 if wrong
        )]

    if t_step > 0 and t_step % 10 == 0:
        cands += [("checkpoint_plan", rand_host(), {}, 4)]  # +2.0 on multiples of 10

    # Weighted random sample
    pool = [(tool, tgt, params) for tool, tgt, params, w in cands for _ in range(w)]
    tool, target, params = random.choice(pool)
    return {"actor": "blue", "tool_name": tool, "target": target,
            "params": params, "rationale": RATIONALE.get(tool, "respond")}

# ---------- 11) Collect SFT dataset from real env rollouts ----------
# Uses TRL "messages" format -> chat template applied automatically AND
# prompt tokens are masked from the loss (only JSON tokens contribute).
print("\n===== Phase 1: collecting SFT data from heuristic policy =====")
N_SFT_EPISODES = 50      # ~50 episodes x ~20 steps = ~1000 (obs, action) pairs
RED_TOOLS = ["recon_network", "attempt_exploit", "lateral_move", "exfiltrate_data"]
random.seed(42)

sft_pairs = []
for ep in range(N_SFT_EPISODES):
    env = CyberSelfPlayEnvironment()
    obs = env.reset()
    for t in range(40):
        if obs.done: break
        red_act = CyberAction(actor="red",
                              tool_name=RED_TOOLS[t % len(RED_TOOLS)],
                              target=f"host-{t % 6:02d}", params={})
        red_obs = env.step(red_act)
        if red_obs.done: break
        blue_action_dict = heuristic_blue_action(
            red_obs.public_state, red_obs.telemetry, t,
        )
        obs_payload = {
            "public_state":     red_obs.public_state,
            "telemetry":        red_obs.telemetry,
            "incident_summary": red_obs.incident_summary,
        }
        sft_pairs.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_MSG},
                {"role": "user",      "content": _user_msg(obs_payload)},
                {"role": "assistant", "content": json.dumps(blue_action_dict, ensure_ascii=True)},
            ]
        })
        obs = env.step(CyberAction(**blue_action_dict))
print(f"Collected {len(sft_pairs)} SFT examples in messages format.")

# Print the action distribution -- if any single tool > 50%, SFT will collapse.
from collections import Counter
_dist = Counter(json.loads(p["messages"][-1]["content"])["tool_name"] for p in sft_pairs)
print("\n[SFT] Blue tool distribution in training data:")
for tool, n in _dist.most_common():
    pct = 100 * n / len(sft_pairs)
    bar = "#" * int(pct / 2)
    print(f"  {tool:<22s} {n:>4d}  {pct:5.1f}%  {bar}")
top_pct = max(_dist.values()) / len(sft_pairs)
if top_pct > 0.5:
    print(f"\nWARNING: top tool is {top_pct:.0%} of data -- expect mode collapse.")
else:
    print(f"\n[SFT] Diversity OK -- top tool is only {top_pct:.0%} of corpus.\n")

sft_dataset = Dataset.from_list(sft_pairs)

# Pre-apply the chat template -> Unsloth's SFTTrainer wants plain text input.
def _msgs_to_text(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )}

sft_dataset = sft_dataset.map(_msgs_to_text, remove_columns=["messages"])
print("Sample SFT text:\n", sft_dataset[0]["text"][:400], "...\n")

# ---------- 12) SFT phase ----------
print("\n===== Phase 1: SFT (imitation learning, BFD-packed) =====")
sft_args = SFTConfig(
    output_dir                  = "/kaggle/working/outputs_cyber/sft",
    learning_rate               = 2e-4,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_train_epochs            = 1,
    logging_steps               = 1,
    warmup_steps                = 5,
    optim                       = "adamw_8bit",
    bf16                        = use_bf16,
    fp16                        = use_fp16,
    save_strategy               = "no",
    report_to                   = "none",
    max_length                  = MAX_SEQ_LEN,
    dataset_text_field          = "text",
    packing                     = True,
    packing_strategy            = "bfd",
)
sft_trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    args          = sft_args,
    train_dataset = sft_dataset,
)
sft_trainer.train()
print("SFT done.")

# ---------- 13) Sanity check after SFT (parses should be ~100%) ----------
# Use GREEDY decoding -- this is the truest signal of what SFT actually learned.
# Vary the env trajectory length per sample so you see different observations
# (and therefore different outputs), not 8 copies of the same one.
print("\n===== Post-SFT sanity check (greedy decoding, varied env states) =====")
FastLanguageModel.for_inference(model)
ok_count, n_check = 0, 8
example_prompt, example_gen = None, None
RED_T = ["recon_network", "attempt_exploit", "lateral_move", "exfiltrate_data"]
for i in range(n_check):
    env_t = CyberSelfPlayEnvironment(); env_t.reset()
    for t in range(random.randint(1, 6)):
        env_t.step(CyberAction(actor="red",
                               tool_name=RED_T[t % len(RED_T)],
                               target=f"host-{random.randint(0,5):02d}"))
    o = env_t.step(CyberAction(actor="red", tool_name="recon_network",
                               target=f"host-{random.randint(0,5):02d}"))
    p = obs_to_prompt({"public_state": o.public_state, "telemetry": o.telemetry,
                       "incident_summary": o.incident_summary})
    inp = tokenizer(p, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    _, ok = parse_action(gen)
    ok_count += int(ok)
    print(f"sample {i+1} (parses={ok}): {gen.strip()[:200]}")
    if example_prompt is None:
        example_prompt, example_gen = p, gen

parse_rate = ok_count / n_check
print(f"\nSFT parse rate: {ok_count}/{n_check} = {parse_rate:.0%}")

if parse_rate < 0.5:
    # Print one full prompt + generation so the user can SEE what the model is doing.
    print("\n--- DEBUG: full inference prompt the model is being given ---")
    print(example_prompt)
    print("--- DEBUG: full model output ---")
    print(example_gen)
    print("--- DEBUG: example SFT training text ---")
    print(sft_dataset[0]["text"][:1500])
    print("--- end debug ---\n")
    raise RuntimeError(
        f"SFT parse rate too low ({parse_rate:.0%}). "
        "GRPO will not learn -- aborting.\n"
        "Fixes: (a) increase N_SFT_EPISODES (50 -> 100), "
        "(b) increase num_train_epochs (4 -> 6), "
        "(c) try a stronger BASE_MODEL (Qwen2.5-Coder-1.5B-Instruct-bnb-4bit)."
    )
FastLanguageModel.for_training(model)

# =============================================================================
# PHASE 2 -- League (PFSP + PSRO) + GRPO  (per-round)
# =============================================================================
from collections import Counter
from statistics import mean
from transformers import TrainerCallback
from IPython.display import clear_output
import matplotlib.pyplot as plt

# --- Tunables (Kaggle T4) ---
LEAGUE_ROUNDS         = 2       # one PFSP/PSRO + mini-GRPO per round (must be >= 1)
GRPO_STEPS_PER_ROUND  = 50
N_GRPO_PROMPTS        = 64
PSRO_EVAL_EPISODES    = 4       # heuristic episodes per Red profile for PSRO replicator
PSRO_ETA              = 0.2
# How to pick the Red for each round: "pfsp" (f(w)=w(1-w)), "psro" (META_PROBS), "mix" (0.5/0.5)
OPP_SAMPLE_MODE = "pfsp"
assert LEAGUE_ROUNDS >= 1, "LEAGUE_ROUNDS must be >= 1 for the league+GRPO script."

# Red archetypes: (min, max) exploit pre-steps before final recon
RED_PROFILES = {
    "R-easy":  (0, 1),   # light red pressure
    "R-mid":   (0, 4),   # default spread
    "R-hard":  (3, 8),   # heavy exploit preamble
}

def env_after_red_preamble(red_name: str):
    """New env, random preamble following RED_PROFILES[red_name], then recon. Returns (env, obs before Blue)."""
    e = CyberSelfPlayEnvironment()
    e.reset()
    lo, hi = RED_PROFILES[red_name]
    n = random.randint(lo, hi)
    for _ in range(n):
        e.step(CyberAction(
            actor="red", tool_name="attempt_exploit",
            target=f"host-{random.randint(0, 5):02d}", params={},
        ))
    o = e.step(CyberAction(
        actor="red", tool_name="recon_network",
        target=f"host-{random.randint(0, 5):02d}", params={},
    ))
    return e, o

# PFSP pool (same as train_blue_vs_pool _seed_pool idea)
LEAGUE_POOL: list = [
    OpponentStats("R-easy", 0.20, 10),
    OpponentStats("R-mid", 0.50, 10),
    OpponentStats("R-hard", 0.80, 10),
]
META_PROBS = [1.0 / 3, 1.0 / 3, 1.0 / 3]  # PSRO: replicator on heuristic-eval payoffs

# Set each league round; compute_rewards reads it for aligned Red preamble
LEAGUE_CURRENT_RED = "R-mid"

def _heuristic_payoff_vector() -> list[float]:
    """Mean Blue return (heuristic policy) vs each red profile — for PSRO replicator."""
    out = []
    for name in ("R-easy", "R-mid", "R-hard"):
        s = 0.0
        for _ in range(PSRO_EVAL_EPISODES):
            e, o = env_after_red_preamble(name)
            a = heuristic_blue_action(o.public_state, o.telemetry, 0)
            o2 = e.step(CyberAction(**a))
            s += float(o2.reward or 0.0)
        out.append(s / max(1, PSRO_EVAL_EPISODES))
    return out

def _update_winrate(opp: OpponentStats, mean_r: float, n_g: int) -> None:
    blue_won = 1.0 if mean_r > 0.0 else 0.0
    new_g = opp.games + n_g
    blended = (opp.win_rate_vs_learner * opp.games + (1.0 - blue_won) * n_g) / max(1, new_g)
    opp.win_rate_vs_learner = float(min(1.0, max(0.0, blended)))
    opp.games = new_g

INVALID_PENALTY      = -1.0
DUP_TOOL_PENALTY     = -0.5
INSTR_TOOL_PENALTY   = -0.2
_REWARD_CALLS  = {"n": 0}
_PER_STEP: list  = []       # per compute_rewards call; reset at each sub-trainer
_PER_ROUND_STEP_BASE = 0   # for logging steps across mini-GRPOs

def compute_rewards(prompts, completions, **kwargs):
    n          = len(completions)
    base       = [0.0] * n
    tools      = [None] * n
    parsed_ok  = [False] * n
    debug_text = None

    for i, comp in enumerate(completions):
        text = comp if isinstance(comp, str) else comp.get("content", "")
        if debug_text is None:
            debug_text = text
        action, ok = parse_action(text)
        if not ok:
            base[i] = INVALID_PENALTY
            continue
        e, _o = env_after_red_preamble(LEAGUE_CURRENT_RED)
        try:
            obs = e.step(CyberAction(**action))
            base[i] = float(obs.reward or 0.0)
            tools[i] = action.get("tool_name")
            parsed_ok[i] = True
        except Exception:
            base[i] = INVALID_PENALTY

    counts  = Counter(t for t in tools if t)
    n_valid = sum(1 for t in tools if t)
    rewards = []
    for i in range(n):
        r = base[i]
        if parsed_ok[i] and tools[i]:
            share = counts[tools[i]] / max(1, n_valid)
            if share > 0.5:
                r += DUP_TOOL_PENALTY * (share - 0.5) * 2
            if tools[i] == "execute_instruction":
                r += INSTR_TOOL_PENALTY
        rewards.append(r)

    _REWARD_CALLS["n"] += 1
    _PER_STEP.append({
        "call": _REWARD_CALLS["n"],
        "rewards": rewards, "tools": tools, "parsed": sum(parsed_ok), "n": n,
        "tool_dist": dict(counts),
    })
    if _REWARD_CALLS["n"] % 5 == 1 and debug_text:
        top = ", ".join(f"{k}={v}" for k, v in counts.most_common(3))
        print(f"[rewards] red={LEAGUE_CURRENT_RED} call={_REWARD_CALLS['n']:3d} "
              f"parsed={sum(parsed_ok)}/{n} r_mean={sum(rewards)/n:+.2f} top={{{top}}}", flush=True)
    return rewards

def build_grpo_prompts(n: int) -> list:
    p = []
    for _ in range(n):
        _e, o = env_after_red_preamble(LEAGUE_CURRENT_RED)
        p.append({"prompt": obs_to_prompt({
            "public_state": o.public_state, "telemetry": o.telemetry,
            "incident_summary": o.incident_summary,
        })})
    return p

# Logging dirs (per-round subfolders; CURVES_DIR used by Hub upload)
OUT_DIR    = Path("/kaggle/working/outputs_cyber")
CURVES_DIR = OUT_DIR / "curves"
(OUT_DIR / "league").mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)
MAIN_LOG   = OUT_DIR / "train_metrics.log"
LEAGUE_JSONL = OUT_DIR / "league_state.jsonl"
JSONL_FILE  = OUT_DIR / "per_step_rewards.jsonl"


def pick_red_opponent():
    """Pick one of R-easy / R-mid / R-hard: PFSP, PSRO (meta-probs), or 50-50 mix."""
    pool = LEAGUE_POOL
    if OPP_SAMPLE_MODE == "psro":
        j = random.choices(range(3), weights=META_PROBS, k=1)[0]
        return pool[j]
    if OPP_SAMPLE_MODE == "mix":
        pf = [pfsp_weight(o.win_rate_vs_learner) for o in pool]
        s0 = sum(pf)
        pf = [x / s0 for x in pf] if s0 > 0 else [1 / 3, 1 / 3, 1 / 3]
        bl = [0.5 * pf[i] + 0.5 * META_PROBS[i] for i in range(3)]
        s1 = sum(bl)
        w = [x / s1 for x in bl]
        j = random.choices(range(3), weights=w, k=1)[0]
        return pool[j]
    return sample_opponent(pool)

class LivePlotCallback(TrainerCallback):
    def __init__(self, round_idx: int, red_name: str):
        self.h = []
        self.r = round_idx
        self.red_name = red_name
        self._per_step_i = 0
        self.curves = OUT_DIR / "curves" / f"round_{round_idx}_{red_name}"
        self.curves.mkdir(parents=True, exist_ok=True)
        self.latest = self.curves / "latest.png"
        self.log_file = MAIN_LOG
        if round_idx == 0:
            self.log_file.write_text("step\tloss\treward\tround\tred\tr_min\tr_max\treward_std\tkl\n")
            JSONL_FILE.write_text("")

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "reward" not in logs:
            return
        gstep = _PER_ROUND_STEP_BASE + state.global_step
        recent = _PER_STEP[self._per_step_i :]
        if not recent:
            return
        self._per_step_i = len(_PER_STEP)
        flat_rewards = [r for c in recent for r in c["rewards"]]
        flat_tools = [t for c in recent for t in c["tools"] if t]
        uq = len(set(flat_tools))
        rmin = min(flat_rewards) if flat_rewards else 0.0
        rmax = max(flat_rewards) if flat_rewards else 0.0
        loss = logs.get("loss", logs.get("train_loss", 0.0))
        row = {**logs, "step": gstep, "round": self.r, "red": self.red_name,
               "reward_min": rmin, "reward_max": rmax, "n_unique_tools": uq,
               "individuals": flat_rewards}
        self.h.append(row)
        print(
            f"[R{self.r} {self.red_name}] step {gstep:4d} loss={loss:+.4f} "
            f"rew={logs.get('reward',0):+.2f} min/max={rmin:+.2f}/{rmax:+.2f} uq={uq}",
            flush=True,
        )
        with self.log_file.open("a") as f:
            f.write(f"{gstep}\t{loss:.6f}\t{logs.get('reward',0):.6f}\t{self.r}\t{self.red_name}\t"
                    f"{rmin:.6f}\t{rmax:.6f}\t{logs.get('reward_std',0):.6f}\t{logs.get('kl',0):.6f}\n")
        with JSONL_FILE.open("a") as f:
            f.write(json.dumps({
                "step": gstep, "round": self.r, "red": self.red_name,
                "rewards": flat_rewards, "tools": flat_tools,
            }) + "\n")
        with LEAGUE_JSONL.open("a") as f:
            f.write(json.dumps({
                "round": self.r, "red": self.red_name, "global_step": gstep,
                "pool": [(o.name, o.win_rate_vs_learner) for o in LEAGUE_POOL],
                "meta_probs": list(META_PROBS),
            }) + "\n")
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        steps = [x["step"] for x in self.h]
        ax[0].plot(steps, [x.get("reward", 0) for x in self.h], "b-")
        ax[0].set_title(f"round {self.r} {self.red_name}  mean reward")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(steps, [x.get("reward_std", 0) for x in self.h], color="orange")
        ax[1].set_title("reward_std"); ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.curves / f"step_{state.global_step:04d}.png", dpi=100)
        fig.savefig(self.latest, dpi=100)
        plt.show()

# ---------- 18) League + GRPO loop ----------
all_hist: list = []
print("\n===== Phase 2: League (PFSP) + mini-GRPO + PSRO replicator =====")
for r in range(LEAGUE_ROUNDS):
    opp = sample_opponent(LEAGUE_POOL)
    LEAGUE_CURRENT_RED = opp.name
    print(f"\n--- League round {r+1}/{LEAGUE_ROUNDS}: [{OPP_SAMPLE_MODE}] Red = {LEAGUE_CURRENT_RED} "
          f"(pool w~={opp.win_rate_vs_learner:.2f}) ---")

    _REWARD_CALLS["n"] = 0
    _PER_STEP.clear()
    train_dataset = Dataset.from_list(build_grpo_prompts(N_GRPO_PROMPTS))
    print(f"  Built {len(train_dataset)} GRPO prompts for {LEAGUE_CURRENT_RED}.")

    training_args = GRPOConfig(
        output_dir                 = f"/kaggle/working/outputs_cyber/grpo_r{r}_{opp.name}",
        learning_rate              = 5e-6,
        per_device_train_batch_size= 2,
        gradient_accumulation_steps= 4,
        num_generations            = 8,
        max_completion_length      = 128,
        max_steps                  = GRPO_STEPS_PER_ROUND,
        logging_steps              = 2,
        warmup_steps               = 6,
        optim                      = "adamw_8bit",
        bf16                       = use_bf16,
        fp16                       = use_fp16,
        use_cpu                    = not has_cuda,
        report_to                  = "none",
        temperature                = 1.0,
        top_p                      = 0.95,
        beta                       = 0.02,
        max_grad_norm              = 1.0,
    )

    trainer = GRPOTrainer(
        model          = model,
        reward_funcs   = [compute_rewards],
        args           = training_args,
        train_dataset  = train_dataset,
        callbacks      = [LivePlotCallback(r, opp.name)],
    )
    trainer.train()
    all_hist.append({"round": r, "red": opp.name, "log_history": list(trainer.state.log_history or [])})
    if trainer.state.log_history:
        last_r = [x.get("reward", 0) for x in trainer.state.log_history if "reward" in x]
        round_mean = mean(last_r) if last_r else 0.0
    else:
        round_mean = 0.0
    _update_winrate(opp, round_mean, max(1, GRPO_STEPS_PER_ROUND // 2))

    payoff = _heuristic_payoff_vector()
    META_PROBS = replicator_update(payoff, META_PROBS, PSRO_ETA)  # type: ignore[assignment]
    print(f"  Heuristic payoffs (easy/mid/hard) = {[round(x,3) for x in payoff]}")
    print(f"  PSRO meta-probs  = {[round(p,3) for p in META_PROBS]}")
    print(f"  Pool win-rates: {[(o.name, round(o.win_rate_vs_learner,2)) for o in LEAGUE_POOL]}")

    _PER_ROUND_STEP_BASE += GRPO_STEPS_PER_ROUND
    (OUT_DIR / f"log_history_r{r}_{opp.name}.json").write_text(
        json.dumps(trainer.state.log_history, indent=2)
    )
    FastLanguageModel.for_training(model)  # Unsloth mode after for_inference if any

(OUT_DIR / "league_log.json").write_text(json.dumps(all_hist, indent=2))
comb = []
for h in all_hist:
    for row in h.get("log_history") or []:
        comb.append({**row, "league_round": h.get("round"), "round_red": h.get("red")})
(OUT_DIR / "log_history_combined.json").write_text(json.dumps(comb, indent=2))
print(f"\nWrote {OUT_DIR / 'league_log.json'}, {OUT_DIR / 'log_history_combined.json'}, {LEAGUE_JSONL}")
# `trainer` / `training_args` refer to the last league round (used in §19 + save)

# ---------- 19) Final summary plot + raw metrics dump (last round + optional combined) ----------
OUT = Path(training_args.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
history = list(trainer.state.log_history or [])
(OUT / "log_history.json").write_text(json.dumps(history, indent=2))
history_combined = comb  # all rounds, with league_round in each row

def _series(key):
    xs, ys = [], []
    for r in history:
        if key in r and "step" in r:
            xs.append(r["step"]); ys.append(r[key])
    return xs, ys

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("CyberSelfPlay SFT+GRPO -- training curves", fontsize=14)

ax = axes[0, 0]
rx, ry = _series("reward"); sx, sy = _series("reward_std")
if rx: ax.plot(rx, ry, color="#2563eb", linewidth=2, label="env reward")
if sx and len(ry) == len(sy):
    ax.fill_between(sx, [m-s for m,s in zip(ry,sy)], [m+s for m,s in zip(ry,sy)],
                    color="#2563eb", alpha=0.15, label="±1 std")
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.set_title("Env reward progression"); ax.set_xlabel("step")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
kx, ky = _series("kl")
if kx: ax.plot(kx, ky, color="#9333ea", linewidth=2)
ax.set_title("KL divergence"); ax.set_xlabel("step"); ax.grid(alpha=0.3)

ax = axes[1, 0]
lx, ly = _series("loss")
if not lx: lx, ly = _series("train_loss")
if lx: ax.plot(lx, ly, color="#be123c", linewidth=2)
ax.set_title("Training loss"); ax.set_xlabel("step"); ax.grid(alpha=0.3)

ax = axes[1, 1]
for k, color, ls in [("completions/mean_length", "#0891b2", "-"),
                     ("completions/min_length",  "#64748b", "--"),
                     ("completions/max_length",  "#64748b", ":")]:
    xs, ys = _series(k)
    if xs: ax.plot(xs, ys, color=color, linestyle=ls, label=k.split("/")[-1])
ax.set_title("Completion length"); ax.set_xlabel("step")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "training_curves.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved -> {OUT/'training_curves.png'} and {OUT/'log_history.json'}")

# Combined plot over all league rounds
def _series_c(rows, key):
    xs, ys = [], []
    for row in rows:
        if key in row and "step" in row:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys

if history_combined:
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    cx, cy = _series_c(history_combined, "reward")
    if cx:
        ax2.plot(cx, cy, color="#1d4ed8", linewidth=1.5)
    ax2.axhline(0, color="gray", ls=":", alpha=0.5)
    ax2.set_title("Mean reward (all league rounds, TRL log steps, combined export)")
    ax2.set_xlabel("step"); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "training_curves_all_rounds.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {OUT_DIR / 'training_curves_all_rounds.png'}")

# ---------- 20) Save adapter ----------
SAVE_DIR = "/kaggle/working/outputs_cyber/cyber-blue-grpo-lora"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nSaved LoRA adapter to {SAVE_DIR}")

# ---------- 21) (Optional) Push to HF Hub ----------
PUSH_TO_HUB    = True
HF_TARGET_REPO = "HarshitShri026/cyber-blue-grpo"

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = ""

if PUSH_TO_HUB and HF_TOKEN:
    from huggingface_hub import HfApi, login
    login(token=HF_TOKEN)
    api = HfApi()
    api.create_repo(repo_id=HF_TARGET_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=HF_TARGET_REPO, folder_path=SAVE_DIR, repo_type="model")
    # Top-level training artifacts.
    for fname in (
        "training_curves.png",
        "training_curves_all_rounds.png",
        "log_history.json",
        "log_history_combined.json",
        "league_log.json",
        "train_metrics.log",
        "per_step_rewards.jsonl",
    ):
        fpath = OUT_DIR / fname if fname in (
            "training_curves_all_rounds.png",
            "log_history_combined.json",
            "league_log.json",
            "train_metrics.log",
            "per_step_rewards.jsonl",
        ) else OUT / fname
        if fpath.exists():
            api.upload_file(path_or_fileobj=str(fpath), path_in_repo=fname,
                            repo_id=HF_TARGET_REPO, repo_type="model")
    # Per-step PNGs (one image per logging step) -> uploaded under curves/.
    if CURVES_DIR.exists():
        api.upload_folder(repo_id=HF_TARGET_REPO, folder_path=str(CURVES_DIR),
                          path_in_repo="curves", repo_type="model")
    print(f"Uploaded -> https://huggingface.co/{HF_TARGET_REPO}")
    print(f"  per-step curves -> https://huggingface.co/{HF_TARGET_REPO}/tree/main/curves")
else:
    print("[push] skipped -- set HF_TOKEN as a Kaggle Secret to enable.")

# ---------- 22) Final sanity check (varied env states) ----------
print("\n===== Post-GRPO sanity check =====")
FastLanguageModel.for_inference(model)
for i in range(3):
    env_t = CyberSelfPlayEnvironment(); env_t.reset()
    for t in range(random.randint(1, 6)):
        env_t.step(CyberAction(actor="red",
                               tool_name=RED_T[t % len(RED_T)],
                               target=f"host-{random.randint(0,5):02d}"))
    o = env_t.step(CyberAction(actor="red", tool_name="recon_network",
                               target=f"host-{random.randint(0,5):02d}"))
    p = obs_to_prompt({"public_state": o.public_state, "telemetry": o.telemetry,
                       "incident_summary": o.incident_summary})
    inp = tokenizer(p, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    action, ok = parse_action(gen)
    print(f"\n--- sample {i+1} (parse_ok={ok}) ---")
    print(gen.strip()); print("Parsed:", action)
print("\nDone.")
