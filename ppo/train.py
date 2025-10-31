#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Training Script for Qwen3 RLHF

This script demonstrates how to train a Qwen3 model using PPO (Proximal Policy Optimization)
with a reward model for Reinforcement Learning from Human Feedback (RLHF).

TRAINING PIPELINE:
==================
1. Load pretrained Qwen3 model as the policy
2. Create a reference copy (frozen) for KL penalty
3. Load or initialize reward model
4. Set up PPO trainer with configuration
5. Train on a dataset of prompts

USAGE:
======
Basic usage:
    python ppo/train.py

With custom configuration:
    python ppo/train.py --model_name Qwen/Qwen2.5-1.5B --learning_rate 1e-5 --batch_size 4

For full list of arguments:
    python ppo/train.py --help

REQUIREMENTS:
=============
- Pretrained Qwen model (loaded from Hugging Face Hub)
- Reward model (can be initialized from policy model)
- Dataset of prompts (for generation and optimization)
- GPU recommended for reasonable training speed

TIPS FOR TRAINING:
==================
- Start with small learning rates (1e-6 to 1e-5)
- Monitor KL divergence - it should be small (< 0.1)
- Use adaptive KL coefficient to maintain target KL
- Generate diverse prompts to avoid mode collapse
- Save checkpoints frequently
"""

import torch
import argparse
import os
import json
from typing import List, Dict, Any
from datetime import datetime

from transformers import AutoTokenizer
from qwen3.model import Qwen3ForCausalLM, Qwen3Config
from ppo.trainer import PPOTrainer, PPOConfig
from ppo.reward_model import Qwen3RewardModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Qwen3 with PPO")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Name or path of pretrained model",
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default=None,
        help="Path to pretrained reward model (if None, initialized from policy)",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=4,
        help="Number of PPO update epochs per batch",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor for returns",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="Lambda for GAE (Generalized Advantage Estimation)",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.1,
        help="Coefficient for KL penalty",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.1,
        help="Coefficient for value loss",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.01,
        help="Target KL divergence for adaptive KL",
    )

    # I/O arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ppo_checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to file containing prompts (one per line)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./ppo_logs",
        help="Directory to save training logs",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model",
    )

    return parser.parse_args()


def get_default_prompts() -> List[str]:
    """
    Get default prompts for demonstration.

    In practice, you would load prompts from a dataset or file.
    These prompts should be diverse and representative of the
    desired behavior you want to reinforce.

    Returns:
        List of prompt strings
    """
    prompts = [
        "Write a helpful response to: How do I learn Python programming?",
        "Explain quantum computing to a 10-year-old.",
        "What are the benefits of regular exercise?",
        "Write a poem about artificial intelligence.",
        "How can I improve my communication skills?",
        "Explain the concept of machine learning in simple terms.",
        "What are some tips for time management?",
        "Write a short story about a robot learning to feel emotions.",
        "How does photosynthesis work?",
        "What are the best practices for software development?",
        "Explain the importance of cybersecurity.",
        "How can I reduce my carbon footprint?",
        "What are the key principles of effective leadership?",
        "Write a dialogue between two AI assistants discussing creativity.",
        "How do neural networks learn?",
        "What are the ethical considerations in AI development?",
    ]
    return prompts


def load_prompts(prompts_file: str) -> List[str]:
    """
    Load prompts from a file.

    Args:
        prompts_file: Path to file with prompts (one per line)

    Returns:
        List of prompt strings
    """
    if not os.path.exists(prompts_file):
        print(f"Warning: Prompts file {prompts_file} not found. Using default prompts.")
        return get_default_prompts()

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts


def setup_models(args):
    """
    Set up policy, reference, and reward models.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (policy_model, ref_model, reward_model, tokenizer)
    """
    print("=" * 80)
    print("SETTING UP MODELS")
    print("=" * 80)

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Check device compatibility
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Warning: bfloat16 not supported on this device, using float32")
        dtype = torch.float32

    device = args.device
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load tokenizer
    print(f"\n1. Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Load policy model
    print(f"\n2. Loading policy model from {args.model_name}...")
    policy_model = Qwen3ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        policy_model = policy_model.to(device)

    print(f"Policy model loaded on {next(policy_model.parameters()).device}")
    print(f"Number of parameters: {sum(p.numel() for p in policy_model.parameters()):,}")

    # Create reference model (frozen copy)
    print(f"\n3. Creating reference model (frozen copy)...")
    ref_model = Qwen3ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        ref_model = ref_model.to(device)

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    print("Reference model created and frozen")

    # Load or create reward model
    print(f"\n4. Setting up reward model...")
    if args.reward_model_path is not None and os.path.exists(args.reward_model_path):
        print(f"Loading reward model from {args.reward_model_path}...")
        reward_model = Qwen3RewardModel.from_pretrained(args.reward_model_path)
        reward_model = reward_model.to(device).to(dtype)
    else:
        print("Initializing reward model from policy model...")
        print("Note: In practice, you should train a reward model on human preference data first!")
        reward_model = Qwen3RewardModel.from_pretrained_policy(policy_model)
        reward_model = reward_model.to(device).to(dtype)

    # Freeze reward model
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model.eval()

    print("Reward model ready")

    return policy_model, ref_model, reward_model, tokenizer


def save_checkpoint(
    model: Qwen3ForCausalLM,
    tokenizer,
    output_dir: str,
    epoch: int,
    stats: Dict[str, Any],
):
    """
    Save model checkpoint and training statistics.

    Args:
        model: Policy model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save checkpoint
        epoch: Current epoch number
        stats: Training statistics
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nSaving checkpoint to {checkpoint_dir}...")

    # Save model
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save statistics
    stats_file = os.path.join(checkpoint_dir, "training_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Checkpoint saved successfully")


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 80)
    print("PPO TRAINING FOR QWEN3 RLHF")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up models
    policy_model, ref_model, reward_model, tokenizer = setup_models(args)

    # Create PPO configuration
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        gamma=args.gamma,
        lam=args.lam,
        clip_range=args.clip_range,
        kl_coef=args.kl_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
    )

    print("\n" + "=" * 80)
    print("PPO CONFIGURATION")
    print("=" * 80)
    print(f"Learning rate: {ppo_config.learning_rate}")
    print(f"Batch size: {ppo_config.batch_size}")
    print(f"PPO epochs: {ppo_config.ppo_epochs}")
    print(f"Gamma: {ppo_config.gamma}")
    print(f"Lambda (GAE): {ppo_config.lam}")
    print(f"Clip range: {ppo_config.clip_range}")
    print(f"KL coefficient: {ppo_config.kl_coef}")
    print(f"Value coefficient: {ppo_config.vf_coef}")
    print(f"Target KL: {ppo_config.target_kl}")

    # Create PPO trainer
    print("\n" + "=" * 80)
    print("INITIALIZING PPO TRAINER")
    print("=" * 80)
    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
    )
    print("PPO Trainer initialized successfully")

    # Load prompts
    print("\n" + "=" * 80)
    print("LOADING PROMPTS")
    print("=" * 80)
    if args.prompts_file is not None:
        prompts = load_prompts(args.prompts_file)
    else:
        print("No prompts file provided, using default prompts")
        prompts = get_default_prompts()

    print(f"Total prompts: {len(prompts)}")
    print(f"First prompt: {prompts[0][:100]}...")

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    try:
        stats = trainer.train(
            prompts=prompts,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)

        # Print final statistics
        if stats:
            print("\nFinal Statistics:")
            final_stats = stats[-1]
            for key, value in final_stats.items():
                print(f"  {key}: {value:.4f}")

        # Save final checkpoint
        save_checkpoint(
            policy_model,
            tokenizer,
            args.output_dir,
            args.num_epochs,
            {"final_stats": stats[-1] if stats else {}, "all_stats": stats},
        )

        # Save training log
        log_file = os.path.join(args.log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w') as f:
            json.dump({
                "args": vars(args),
                "stats": stats,
            }, f, indent=2)

        print(f"\nTraining log saved to {log_file}")

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("TRAINING INTERRUPTED")
        print("=" * 80)
        print("Saving checkpoint before exit...")
        save_checkpoint(
            policy_model,
            tokenizer,
            args.output_dir,
            0,
            {"interrupted": True},
        )

    except Exception as e:
        print("\n" + "=" * 80)
        print("TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
