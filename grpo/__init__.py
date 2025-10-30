"""
GRPO (Group Relative Policy Optimization) Package

This package provides tools for fine-tuning language models using GRPO,
an advanced RLHF technique that optimizes policies by comparing multiple
outputs for the same prompt.

Main Components:
- GRPOTrainer: Main trainer class for GRPO training
- GRPOConfig: Configuration dataclass for training parameters
- SimplePromptDataset: Simple dataset class for prompts

Example usage:
    from grpo import GRPOTrainer, GRPOConfig, SimplePromptDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load models
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    # Create config and dataset
    config = GRPOConfig(batch_size=4, num_epochs=3)
    dataset = SimplePromptDataset(["your", "prompts", "here"])

    # Train
    trainer = GRPOTrainer(model, ref_model, model, tokenizer, config, dataset)
    trainer.train()
"""

from .trainer import GRPOTrainer, GRPOConfig, SimplePromptDataset

__all__ = ["GRPOTrainer", "GRPOConfig", "SimplePromptDataset"]
__version__ = "0.1.0"
