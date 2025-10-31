#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO Training Script for Qwen3 Model

This script demonstrates how to use the GRPO (Group Relative Policy Optimization)
trainer to fine-tune a Qwen3 model with reinforcement learning from human feedback.

USAGE:
======

Basic usage with default settings:
    python grpo/train_grpo.py

With custom model:
    python grpo/train_grpo.py --model-name Qwen/Qwen2.5-1.5B-Instruct

With Weights & Biases logging:
    python grpo/train_grpo.py --use-wandb --wandb-project my-grpo-project

Full example with all options:
    python grpo/train_grpo.py \\
        --model-name Qwen/Qwen2.5-1.5B-Instruct \\
        --batch-size 4 \\
        --num-epochs 3 \\
        --learning-rate 1e-6 \\
        --num-generations 4 \\
        --output-dir ./my_grpo_model \\
        --use-wandb

DATASET:
========

This script uses a simple example dataset of prompts. For production use,
replace the example prompts with your own dataset containing:
- High-quality prompts relevant to your use case
- Diverse topics and task types
- Appropriate length and complexity

You can also load datasets from Hugging Face Hub using the `datasets` library.

REWARD MODEL:
============

By default, this script uses the policy model itself as a simple reward model.
For production training, you should:
1. Train or obtain a proper reward model
2. Load it separately using AutoModelForSequenceClassification
3. Ensure it outputs scalar rewards for prompt-response pairs

REQUIREMENTS:
============

- torch
- transformers
- datasets
- wandb (optional, for logging)
- tqdm

Install with: pip install torch transformers datasets wandb tqdm
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer import GRPOTrainer, GRPOConfig, SimplePromptDataset


def create_example_dataset():
    """
    Create a simple example dataset for demonstration.

    In production, replace this with your own dataset of prompts.
    You can load datasets from Hugging Face Hub using:

    from datasets import load_dataset
    dataset = load_dataset("your-dataset-name")
    prompts = dataset["train"]["prompt"]
    """
    example_prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing to a 10-year-old.",
        "What are the main differences between machine learning and deep learning?",
        "Describe the water cycle in nature.",
        "Write a haiku about programming.",
        "Explain the concept of recursion with an example.",
        "What are the benefits of regular exercise?",
        "How do solar panels work?",
        "Write a dialogue between two friends discussing their favorite books.",
        "Explain the greenhouse effect.",
        "What is the importance of biodiversity?",
        "Describe the process of photosynthesis.",
        "Write a poem about the changing seasons.",
        "What are some effective study techniques for students?",
        "Explain the basics of cryptocurrency.",
        "How does the internet work?",
        "Write a creative recipe for a fictional dish.",
        "What are the key principles of sustainable living?",
        "Explain the concept of neural networks.",
        "Describe a perfect day in your ideal vacation destination.",
    ]

    return SimplePromptDataset(example_prompts)


def load_models(model_name: str, device: str):
    """
    Load policy, reference, and reward models.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load models on

    Returns:
        Tuple of (model, ref_model, reward_model, tokenizer)
    """
    print(f"\n{'='*80}")
    print(f"Loading models from {model_name}")
    print(f"{'='*80}\n")

    # Determine dtype based on device capabilities
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16 precision")
    elif device == "cuda":
        dtype = torch.float16
        print("Using float16 precision")
    else:
        dtype = torch.float32
        print("Using float32 precision")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load policy model (the model we'll train)
    print("Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        model = model.to(device)

    print(f"Policy model loaded on {next(model.parameters()).device}")

    # Load reference model (frozen copy for KL penalty)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        ref_model = ref_model.to(device)

    print(f"Reference model loaded on {next(ref_model.parameters()).device}")

    # Load reward model
    # NOTE: In production, you should use a separately trained reward model
    # For this example, we'll use the policy model itself as a simple reward model
    print("\nWARNING: Using policy model as reward model (for demonstration only)")
    print("In production, load a proper reward model trained on human preferences\n")

    reward_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        reward_model = reward_model.to(device)

    print(f"Reward model loaded on {next(reward_model.parameters()).device}")

    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Information:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Vocabulary size: {model.config.vocab_size:,}")
    print(f"  Hidden size: {model.config.hidden_size:,}")

    return model, ref_model, reward_model, tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Qwen3 model using GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path"
    )

    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of responses to generate per prompt"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per response"
    )

    # GRPO-specific parameters
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.1,
        help="KL divergence coefficient"
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clipping range"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    # Optimization parameters
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./grpo_checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="qwen3-grpo",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name"
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    print("\n" + "="*80)
    print("GRPO Training for Qwen3")
    print("="*80)

    # Load models
    model, ref_model, reward_model, tokenizer = load_models(
        args.model_name,
        args.device
    )

    # Create dataset
    print("\n" + "="*80)
    print("Preparing dataset")
    print("="*80 + "\n")

    train_dataset = create_example_dataset()
    print(f"Dataset size: {len(train_dataset)} prompts")
    print(f"Example prompt: {train_dataset[0]['prompt']}")

    # Create GRPO config
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80 + "\n")

    config = GRPOConfig(
        num_generations_per_prompt=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        kl_coef=args.kl_coef,
        clip_range=args.clip_range,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        device=args.device
    )

    # Print config
    print("GRPO Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")

    # Create trainer
    print("\n" + "="*80)
    print("Initializing Trainer")
    print("="*80 + "\n")

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_dataset
    )

    # Train
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    trainer.train()

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nModel saved to: {args.output_dir}")
    print("\nTo use the trained model:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}/final_model')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}/final_model')")


if __name__ == "__main__":
    main()
