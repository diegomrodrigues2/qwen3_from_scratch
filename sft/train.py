"""
Supervised Fine-Tuning (SFT) training script for Qwen3.

This module provides a trainer class and utilities for fine-tuning Qwen3 models
on instruction-following datasets using Hugging Face transformers.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
import wandb
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3.model import Qwen3ForCausalLM, Qwen3Config
from sft.dataset import SFTDataset, ConversationDataset


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning.

    Args:
        model_name_or_path: Path to pretrained model or model identifier
        data_path: Path to training data
        output_dir: Directory to save model checkpoints
        max_length: Maximum sequence length (default: 2048)
        num_train_epochs: Number of training epochs (default: 3)
        per_device_train_batch_size: Batch size per device during training (default: 4)
        per_device_eval_batch_size: Batch size per device during evaluation (default: 4)
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4)
        learning_rate: Initial learning rate (default: 2e-5)
        weight_decay: Weight decay coefficient (default: 0.0)
        warmup_ratio: Ratio of total steps for learning rate warmup (default: 0.03)
        logging_steps: Log every N steps (default: 10)
        save_steps: Save checkpoint every N steps (default: 500)
        eval_steps: Evaluate every N steps (default: 500)
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
        fp16: Use 16-bit floating point precision (default: False)
        bf16: Use bfloat16 precision (default: False)
        gradient_checkpointing: Use gradient checkpointing to save memory (default: False)
        dataloader_num_workers: Number of dataloader workers (default: 4)
        seed: Random seed (default: 42)
        use_chat_template: Use model's chat template for formatting (default: True)
        format_type: Data format type ("alpaca", "sharegpt", "custom") (default: "alpaca")
        eval_data_path: Path to evaluation data (optional)
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
        report_to: List of integrations to report to (default: ["wandb"])
        run_name: Name for this run (optional)
        load_in_8bit: Load model in 8-bit precision (default: False)
        load_in_4bit: Load model in 4-bit precision (default: False)
        lora_r: LoRA rank (default: 0, disabled)
        lora_alpha: LoRA alpha parameter (default: 16)
        lora_dropout: LoRA dropout (default: 0.05)
        target_modules: LoRA target modules (default: None, auto-select)
    """

    # Model and data
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    data_path: str = ""
    eval_data_path: Optional[str] = None
    output_dir: str = "./output"

    # Training hyperparameters
    max_length: int = 2048
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = None

    # Optimization
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # Data loading
    dataloader_num_workers: int = 4
    use_chat_template: bool = True
    format_type: str = "alpaca"

    # Other
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # Evaluation
    do_eval: bool = True
    evaluation_strategy: str = "steps"


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning of Qwen3 models."""

    def __init__(self, config: SFTConfig):
        """
        Initialize the SFT trainer.

        Args:
            config: SFT configuration object
        """
        self.config = config

        # Set random seed
        torch.manual_seed(config.seed)

        # Initialize model and tokenizer
        print(f"Loading model from {config.model_name_or_path}...")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Load datasets
        print(f"Loading training data from {config.data_path}...")
        self.train_dataset = self._load_dataset(config.data_path, split="train")

        self.eval_dataset = None
        if config.do_eval and config.eval_data_path:
            print(f"Loading evaluation data from {config.eval_data_path}...")
            self.eval_dataset = self._load_dataset(config.eval_data_path, split="validation")

        # Initialize trainer
        self.trainer = self._create_trainer()

    def _load_tokenizer(self):
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _load_model(self):
        """Load model with optional quantization and LoRA."""
        # Prepare model loading arguments
        model_kwargs = {}

        if self.config.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif self.config.load_in_4bit:
            model_kwargs['load_in_4bit'] = True

        # Load model
        try:
            # Try loading our custom model first
            model = Qwen3ForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"Could not load custom model: {e}")
            print("Trying to load from Hugging Face...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
                **model_kwargs
            )

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Apply LoRA if requested
        if self.config.use_lora:
            model = self._apply_lora(model)

        return model

    def _apply_lora(self, model):
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # Auto-select target modules if not specified
            target_modules = self.config.target_modules
            if target_modules is None:
                # Default target modules for Qwen3
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            return model
        except ImportError:
            print("Warning: peft not installed. Install with: pip install peft")
            print("Continuing without LoRA...")
            return model

    def _load_dataset(self, data_path: str, split: str):
        """Load dataset."""
        dataset = SFTDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            split=split,
            format_type=self.config.format_type,
            use_chat_template=self.config.use_chat_template,
        )
        return dataset

    def _create_trainer(self):
        """Create Hugging Face Trainer."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self.config.do_eval else None,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            seed=self.config.seed,
            do_eval=self.config.do_eval,
            evaluation_strategy=self.config.evaluation_strategy if self.config.do_eval else "no",
            load_best_model_at_end=self.config.do_eval,
            metric_for_best_model="loss" if self.config.do_eval else None,
            greater_is_better=False if self.config.do_eval else None,
            save_safetensors=True,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        return trainer

    def train(self):
        """Run training."""
        print("Starting training...")

        # Check for checkpoint
        checkpoint = None
        if self.config.resume_from_checkpoint is not None:
            checkpoint = self.config.resume_from_checkpoint
        else:
            # Check for last checkpoint in output dir
            last_checkpoint = get_last_checkpoint(self.config.output_dir)
            if last_checkpoint is not None:
                print(f"Found checkpoint: {last_checkpoint}")
                checkpoint = last_checkpoint

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)

        # Save model
        print(f"Saving model to {self.config.output_dir}...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        print("Training complete!")

        return train_result

    def evaluate(self):
        """Run evaluation."""
        if self.eval_dataset is None:
            print("No evaluation dataset provided.")
            return None

        print("Starting evaluation...")
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        print(f"Evaluation results: {metrics}")
        return metrics


def main():
    """Main training function with example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Qwen3 with Supervised Fine-Tuning")

    # Model and data
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")

    # Training hyperparameters
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")

    # Optimization
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")

    # Data format
    parser.add_argument("--format_type", type=str, default="alpaca",
                        choices=["alpaca", "sharegpt", "custom"],
                        help="Data format type")
    parser.add_argument("--use_chat_template", action="store_true", default=True,
                        help="Use chat template")

    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for logging")

    args = parser.parse_args()

    # Create config
    config = SFTConfig(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        format_type=args.format_type,
        use_chat_template=args.use_chat_template,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        run_name=args.run_name,
    )

    # Create trainer
    trainer = SFTTrainer(config)

    # Train
    trainer.train()

    # Evaluate if eval data provided
    if args.eval_data_path:
        trainer.evaluate()


if __name__ == "__main__":
    main()
