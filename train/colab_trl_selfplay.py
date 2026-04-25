"""
Colab-ready TRL self-play loop scaffold.

Usage in Colab:
  !pip install "openenv-core[core]" "trl>=0.10.0" "transformers>=4.40.0" datasets accelerate peft
  !python train/colab_trl_selfplay.py
"""

from dataclasses import dataclass
from typing import Dict, List

try:
    from .pfsp import OpponentStats, sample_opponent
except ImportError:
    from pfsp import OpponentStats, sample_opponent

try:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import PPOConfig, PPOTrainer
except Exception:
    Dataset = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PPOConfig = None
    PPOTrainer = None


@dataclass
class RolloutRecord:
    prompt: str
    response: str
    reward: float
    metadata: Dict


def collect_rollouts(num_episodes: int = 8) -> List[RolloutRecord]:
    # Replace with real OpenEnv rollouts.
    rows: List[RolloutRecord] = []
    for i in range(num_episodes):
        rows.append(
            RolloutRecord(
                prompt=f"episode-{i}: choose next cyber action",
                response="triage_alerts host-03",
                reward=0.1 * (i + 1),
                metadata={"actor": "blue"},
            )
        )
    return rows


def train_with_trl_placeholder(rollouts: List[RolloutRecord]) -> None:
    # Uses TRL if available; otherwise falls back to proxy logs.
    if not (Dataset and AutoModelForCausalLM and AutoTokenizer and PPOConfig and PPOTrainer):
        avg = sum(r.reward for r in rollouts) / max(1, len(rollouts))
        print(f"[TRL fallback] batches={len(rollouts)} avg_reward={avg:.3f}")
        return

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ppo_config = PPOConfig(batch_size=2, mini_batch_size=1, learning_rate=1e-5)
    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    ds = Dataset.from_dict(
        {
            "query": [r.prompt for r in rollouts],
            "response": [r.response for r in rollouts],
            "reward": [r.reward for r in rollouts],
        }
    )
    for row in ds:
        trainer.step([row["query"]], [row["response"]], [row["reward"]])
    avg = sum(r.reward for r in rollouts) / max(1, len(rollouts))
    print(f"[TRL run] batches={len(rollouts)} avg_reward={avg:.3f}")


def main():
    red_pool = [
        OpponentStats(name="R0", win_rate_vs_learner=0.2, games=50),
        OpponentStats(name="R1", win_rate_vs_learner=0.5, games=50),
        OpponentStats(name="R2", win_rate_vs_learner=0.8, games=50),
    ]
    for round_id in range(3):
        sampled_red = sample_opponent(red_pool)
        print(f"[round {round_id}] sampled_opponent={sampled_red.name}")
        rollouts = collect_rollouts(num_episodes=8)
        train_with_trl_placeholder(rollouts)
    print("Colab loop finished. Replace placeholder blocks with PPO/GRPO trainer calls.")


if __name__ == "__main__":
    main()
