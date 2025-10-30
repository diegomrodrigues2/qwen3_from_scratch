#!/usr/bin/env python3
"""
Example script for training Qwen3 with Supervised Fine-Tuning (SFT).

This script demonstrates how to use the SFT training utilities to fine-tune
a Qwen3 model on instruction-following datasets.

Usage examples:

1. Basic training with Alpaca-format dataset:
   python train_sft.py --data_path ./data/alpaca.json --output_dir ./output

2. Training with evaluation:
   python train_sft.py --data_path ./data/train.json --eval_data_path ./data/eval.json

3. Training with LoRA (efficient fine-tuning):
   python train_sft.py --data_path ./data/alpaca.json --use_lora --lora_r 8

4. Training with custom hyperparameters:
   python train_sft.py --data_path ./data/alpaca.json \
       --num_train_epochs 5 \
       --learning_rate 5e-5 \
       --per_device_train_batch_size 8 \
       --gradient_accumulation_steps 2

5. Training with mixed precision (bf16):
   python train_sft.py --data_path ./data/alpaca.json --bf16

6. Training with Hugging Face dataset:
   python train_sft.py --data_path "yahma/alpaca-cleaned" --output_dir ./output
"""

import os
import sys
import argparse
from dataclasses import asdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sft import SFTTrainer, SFTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Qwen3 with Supervised Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model and data
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Path to pretrained model or model identifier from Hugging Face"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSON/JSONL file or HuggingFace dataset name)"
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to evaluation data (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save model checkpoints and logs"
    )

    # Training hyperparameters
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device (GPU/CPU)"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Ratio of total steps for learning rate warmup"
    )

    # Optimization
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use 16-bit floating point precision (fp16)"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision (recommended for A100/H100 GPUs)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory (slower training)"
    )

    # Data format
    parser.add_argument(
        "--format_type",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt", "custom"],
        help="Data format type (alpaca: instruction-input-output, sharegpt: conversations)"
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template formatting"
    )

    # LoRA configuration
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for efficient fine-tuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (lower = fewer parameters)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (requires eval_data_path)"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run (for logging)"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=["wandb"],
        help="Reporting integrations (wandb, tensorboard, etc.)"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Disable evaluation during training"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Print configuration
    print("=" * 80)
    print("Supervised Fine-Tuning (SFT) for Qwen3")
    print("=" * 80)
    print(f"\nModel: {args.model_name_or_path}")
    print(f"Training data: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Format: {args.format_type}")
    print(f"Max length: {args.max_length}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")

    if args.use_lora:
        print(f"\nLoRA Configuration:")
        print(f"  Rank: {args.lora_r}")
        print(f"  Alpha: {args.lora_alpha}")
        print(f"  Dropout: {args.lora_dropout}")

    if args.fp16:
        print("\nUsing FP16 mixed precision training")
    elif args.bf16:
        print("\nUsing BF16 mixed precision training")

    print("\n" + "=" * 80 + "\n")

    # Create SFT configuration
    config = SFTConfig(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        format_type=args.format_type,
        use_chat_template=not args.no_chat_template,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        run_name=args.run_name,
        report_to=args.report_to,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        do_eval=not args.no_eval and args.eval_data_path is not None,
    )

    # Create trainer
    try:
        trainer = SFTTrainer(config)
    except Exception as e:
        print(f"\nError initializing trainer: {e}")
        print("\nMake sure:")
        print("1. Your data path is correct and accessible")
        print("2. The model name/path is valid")
        print("3. You have the required dependencies installed")
        sys.exit(1)

    # Run training
    try:
        train_result = trainer.train()
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"\nModel saved to: {args.output_dir}")

        # Run evaluation if requested
        if config.do_eval and args.eval_data_path:
            print("\nRunning final evaluation...")
            eval_result = trainer.evaluate()
            print(f"Evaluation results: {eval_result}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial results may be saved in: {args.output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
