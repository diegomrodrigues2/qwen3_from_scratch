#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple GRPO Training Example

This script demonstrates a minimal working example of GRPO training.
It's designed to be run quickly for testing and demonstration purposes.

Run with:
    python grpo/example.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer import GRPOTrainer, GRPOConfig, SimplePromptDataset


def main():
    print("\n" + "="*80)
    print("GRPO Training - Minimal Example")
    print("="*80 + "\n")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Training on CPU will be slow. Use a GPU for faster training.")

    # Use a very small model for quick demonstration
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading model: {model_name}")
    print("(Using the smallest Qwen model for quick demonstration)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )

    print("Using policy model as reward model (demo only)...")
    reward_model = model  # In production, use a separate reward model

    # Create a tiny dataset for quick demo
    print("\nCreating demo dataset...")
    demo_prompts = [
        "Write a haiku about coding.",
        "Explain recursion simply.",
        "What is machine learning?",
        "Describe the water cycle.",
        "Write a short poem about AI.",
    ]

    dataset = SimplePromptDataset(demo_prompts)
    print(f"Dataset size: {len(dataset)} prompts")

    # Create a minimal config for quick training
    print("\nConfiguring GRPO trainer...")
    config = GRPOConfig(
        # Small batch size for quick demo
        batch_size=1,
        num_epochs=1,

        # Minimal generations to save time
        num_generations_per_prompt=2,
        max_new_tokens=50,

        # Small number of steps
        gradient_accumulation_steps=2,

        # Frequent logging for demo
        logging_steps=1,
        save_steps=10,

        # Output directory
        output_dir="./grpo_demo_output",

        # Device
        device=device
    )

    print("\nConfiguration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Generations per prompt: {config.num_generations_per_prompt}")
    print(f"  Max new tokens: {config.max_new_tokens}")

    # Create trainer
    print("\n" + "="*80)
    print("Initializing GRPO Trainer")
    print("="*80 + "\n")

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset
    )

    # Run training
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    try:
        trainer.train()

        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)

        # Test the trained model
        print("\n" + "="*80)
        print("Testing Trained Model")
        print("="*80 + "\n")

        test_prompt = "Write a haiku about programming:"
        print(f"Prompt: {test_prompt}")

        model.eval()
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nGenerated response:\n{response}")

        print(f"\n\nModel saved to: {config.output_dir}/final_model")
        print("You can load it with:")
        print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{config.output_dir}/final_model')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{config.output_dir}/final_model')")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial checkpoints may have been saved to:", config.output_dir)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
