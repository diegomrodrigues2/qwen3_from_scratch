"""
PPO (Proximal Policy Optimization) Package for RLHF

This package implements PPO training for Qwen3 models with Reinforcement Learning
from Human Feedback (RLHF). It includes:

- PPOTrainer: Main training class for PPO algorithm
- PPOConfig: Configuration for PPO hyperparameters
- Qwen3RewardModel: Reward model for scoring generated text
- compute_pairwise_loss: Loss function for reward model training

USAGE:
======
Basic training example:

    from ppo import PPOTrainer, PPOConfig, Qwen3RewardModel
    from qwen3.model import Qwen3ForCausalLM
    from transformers import AutoTokenizer

    # Load models
    policy = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    ref_policy = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    reward_model = Qwen3RewardModel.from_pretrained_policy(policy)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    # Configure PPO
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        ppo_epochs=4,
    )

    # Train
    trainer = PPOTrainer(config, policy, ref_policy, reward_model, tokenizer)
    prompts = ["How do I learn Python?", "Explain quantum computing"]
    trainer.train(prompts, num_epochs=1)

For command-line training, use:
    python -m ppo.train --model_name Qwen/Qwen2.5-1.5B --batch_size 4

See ppo/train.py for full training script with all options.
"""

from ppo.trainer import PPOTrainer, PPOConfig
from ppo.reward_model import (
    Qwen3RewardModel,
    RewardModelOutput,
    compute_pairwise_loss,
)

__all__ = [
    "PPOTrainer",
    "PPOConfig",
    "Qwen3RewardModel",
    "RewardModelOutput",
    "compute_pairwise_loss",
]
