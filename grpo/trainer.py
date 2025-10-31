# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) Trainer Implementation

GRPO is an advanced RLHF technique that optimizes language models by comparing
multiple outputs (a group) for the same prompt, using relative rankings to reduce
variance and improve sample efficiency compared to traditional PPO.

ARCHITECTURAL OVERVIEW:
======================

GRPO builds upon the insights from policy gradient methods but introduces several
key innovations:

1. **Group-Based Learning**: Instead of optimizing individual responses, GRPO
   generates multiple responses per prompt and learns from their relative quality.

2. **Variance Reduction**: By using within-group comparisons, GRPO significantly
   reduces the variance of gradient estimates, leading to more stable training.

3. **Simplified Algorithm**: Unlike PPO, GRPO doesn't require a separate value
   network, making it simpler to implement and train.

KEY COMPONENTS:
==============

1. **Policy Model**: The language model being optimized (actor)
2. **Reference Model**: Frozen copy of the initial policy for KL penalty
3. **Reward Model**: Scores the quality of generated responses
4. **Group Generation**: Generate K responses per prompt
5. **Advantage Estimation**: Compute relative advantages within each group
6. **Policy Update**: Update policy to maximize expected relative reward

ALGORITHM FLOW:
==============

For each training step:
1. Sample a batch of prompts from the dataset
2. For each prompt, generate K different responses using the current policy
3. Score all responses using the reward model
4. Compute advantages relative to the group mean
5. Calculate policy loss with KL divergence penalty
6. Update the policy model using gradient descent

Mathematical Formulation:
L = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL[π_θ || π_ref]

Where:
- r(θ) = π_θ(a|s) / π_ref(a|s) is the probability ratio
- A is the advantage (reward - baseline)
- β is the KL penalty coefficient
- ε is the clipping parameter

This implementation is designed to work with the Qwen3 model architecture
and integrates seamlessly with the Hugging Face ecosystem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import wandb
import os
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import get_linear_schedule_with_warmup


@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training.

    Hyperparameters controlling the GRPO training process, including
    generation settings, optimization parameters, and regularization.

    Args:
        num_generations_per_prompt (int): Number of responses to generate per prompt (K)
        learning_rate (float): Learning rate for the policy optimizer
        batch_size (int): Number of prompts per training batch
        num_epochs (int): Number of training epochs
        max_new_tokens (int): Maximum tokens to generate per response
        temperature (float): Sampling temperature for generation
        kl_coef (float): Coefficient for KL divergence penalty (β)
        clip_range (float): PPO-style clipping range (ε)
        gamma (float): Discount factor for rewards
        gradient_accumulation_steps (int): Steps to accumulate gradients
        max_grad_norm (float): Maximum gradient norm for clipping
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        logging_steps (int): Log metrics every N steps
        save_steps (int): Save checkpoint every N steps
        output_dir (str): Directory to save checkpoints and logs
        use_wandb (bool): Whether to use Weights & Biases for logging
    """
    # Generation parameters
    num_generations_per_prompt: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    # Training parameters
    learning_rate: float = 1e-6
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # GRPO-specific parameters
    kl_coef: float = 0.1
    clip_range: float = 0.2
    gamma: float = 1.0

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "./grpo_checkpoints"
    use_wandb: bool = False
    wandb_project: str = "qwen3-grpo"
    wandb_run_name: Optional[str] = None

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32


class GRPOTrainer:
    """
    GRPO Trainer for fine-tuning language models with reinforcement learning.

    This trainer implements the Group Relative Policy Optimization algorithm,
    which learns to improve a language model's outputs by comparing multiple
    generations for the same prompt and optimizing based on their relative quality.

    The trainer handles:
    - Multiple response generation per prompt
    - Reward model integration for scoring responses
    - Group-based advantage calculation
    - Policy optimization with KL regularization
    - Checkpointing and logging

    Args:
        model (PreTrainedModel): The policy model to train
        ref_model (PreTrainedModel): Reference model for KL penalty (frozen)
        reward_model (PreTrainedModel): Model to score responses
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing
        config (GRPOConfig): Training configuration
        train_dataset (Dataset): Dataset containing prompts
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: GRPOConfig,
        train_dataset: Dataset,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset

        # Move models to device
        self.model = self.model.to(config.device)
        self.ref_model = self.ref_model.to(config.device)
        self.reward_model = self.reward_model.to(config.device)

        # Set models to appropriate modes
        self.model.train()
        self.ref_model.eval()
        self.reward_model.eval()

        # Freeze reference and reward models
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Setup learning rate scheduler
        total_steps = (len(train_dataset) // config.batch_size) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Setup data loader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config)
            )

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for batching prompts.

        Args:
            batch: List of samples from dataset

        Returns:
            Dictionary containing batched prompts and metadata
        """
        prompts = [item["prompt"] for item in batch]
        return {"prompts": prompts}

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str]
    ) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses for each prompt.

        For each prompt, generate K different responses using the current policy.
        Also compute log probabilities for both the policy and reference models.

        Args:
            prompts: List of input prompts

        Returns:
            Tuple of:
            - responses: List of K responses for each prompt
            - policy_logprobs: Log probabilities under current policy
            - ref_logprobs: Log probabilities under reference policy
        """
        self.model.eval()

        all_responses = []
        all_policy_logprobs = []
        all_ref_logprobs = []

        for prompt in prompts:
            prompt_responses = []
            prompt_policy_logprobs = []
            prompt_ref_logprobs = []

            # Generate K responses for this prompt
            for _ in range(self.config.num_generations_per_prompt):
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.config.device)

                # Generate response
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                # Decode response (only new tokens)
                response_ids = output.sequences[0][inputs.input_ids.shape[1]:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                prompt_responses.append(response_text)

                # Compute log probabilities for the generated sequence
                # Policy model logprobs
                with torch.enable_grad():
                    policy_outputs = self.model(
                        input_ids=output.sequences,
                        attention_mask=torch.ones_like(output.sequences)
                    )
                    policy_logits = policy_outputs.logits[:, inputs.input_ids.shape[1]-1:-1, :]
                    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
                    policy_logprobs = torch.gather(
                        policy_logprobs,
                        dim=-1,
                        index=response_ids.unsqueeze(0).unsqueeze(-1)
                    ).squeeze(-1).sum().item()
                    prompt_policy_logprobs.append(policy_logprobs)

                # Reference model logprobs
                ref_outputs = self.ref_model(
                    input_ids=output.sequences,
                    attention_mask=torch.ones_like(output.sequences)
                )
                ref_logits = ref_outputs.logits[:, inputs.input_ids.shape[1]-1:-1, :]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = torch.gather(
                    ref_logprobs,
                    dim=-1,
                    index=response_ids.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1).sum().item()
                prompt_ref_logprobs.append(ref_logprobs)

            all_responses.append(prompt_responses)
            all_policy_logprobs.append(prompt_policy_logprobs)
            all_ref_logprobs.append(prompt_ref_logprobs)

        self.model.train()

        return (
            all_responses,
            torch.tensor(all_policy_logprobs, device=self.config.device),
            torch.tensor(all_ref_logprobs, device=self.config.device)
        )

    @torch.no_grad()
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[List[str]]
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses using the reward model.

        Args:
            prompts: List of prompts
            responses: List of K responses for each prompt

        Returns:
            Tensor of shape (num_prompts, K) containing rewards
        """
        self.reward_model.eval()

        all_rewards = []

        for prompt, prompt_responses in zip(prompts, responses):
            prompt_rewards = []

            for response in prompt_responses:
                # Combine prompt and response
                full_text = prompt + response

                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.config.device)

                # Get reward (assuming reward model outputs a scalar reward)
                # If using a different reward model architecture, adjust this
                outputs = self.reward_model(**inputs)

                # Extract reward - this depends on your reward model architecture
                # Common approaches:
                # 1. Last token logit: reward = outputs.logits[0, -1, 0]
                # 2. Mean pooling: reward = outputs.last_hidden_state.mean(dim=1)
                # For this implementation, we'll use a simple approach
                if hasattr(outputs, 'score'):
                    reward = outputs.score.item()
                else:
                    # Fallback: use last hidden state norm as a simple reward
                    reward = outputs.logits[0, -1, 0].item()

                prompt_rewards.append(reward)

            all_rewards.append(prompt_rewards)

        return torch.tensor(all_rewards, device=self.config.device)

    def compute_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        GRPO uses advantages computed relative to the mean reward within each group.
        This reduces variance compared to using a global baseline.

        Args:
            rewards: Tensor of shape (num_prompts, K) containing rewards

        Returns:
            Tensor of same shape containing advantages
        """
        # Compute mean reward for each prompt group
        mean_rewards = rewards.mean(dim=1, keepdim=True)

        # Advantages are rewards minus the group mean
        advantages = rewards - mean_rewards

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def compute_policy_loss(
        self,
        prompts: List[str],
        responses: List[List[str]],
        old_policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the GRPO policy loss.

        The loss combines:
        1. Clipped policy gradient loss (PPO-style)
        2. KL divergence penalty from reference model

        Args:
            prompts: List of prompts
            responses: List of responses for each prompt
            old_policy_logprobs: Log probs from generation time
            ref_logprobs: Reference model log probs
            advantages: Computed advantages

        Returns:
            Tuple of (loss, metrics_dict)
        """
        total_loss = 0.0
        num_responses = 0

        kl_divergences = []
        policy_ratios = []

        for i, (prompt, prompt_responses) in enumerate(zip(prompts, responses)):
            for j, response in enumerate(prompt_responses):
                # Combine prompt and response
                full_text = prompt + response

                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.config.device)

                # Get current policy logprobs
                outputs = self.model(**inputs)
                logits = outputs.logits[:, :-1, :]
                target_ids = inputs.input_ids[:, 1:]

                # Compute log probabilities
                logprobs = F.log_softmax(logits, dim=-1)
                current_logprobs = torch.gather(
                    logprobs,
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1).sum()

                # Compute probability ratio
                old_logprob = old_policy_logprobs[i, j]
                ratio = torch.exp(current_logprobs - old_logprob)

                # Compute clipped surrogate loss (PPO-style)
                advantage = advantages[i, j]
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range
                )

                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                )

                # Compute KL divergence from reference model
                ref_logprob = ref_logprobs[i, j]
                kl_div = current_logprobs - ref_logprob

                # Total loss for this response
                loss = policy_loss + self.config.kl_coef * kl_div
                total_loss += loss

                # Track metrics
                kl_divergences.append(kl_div.item())
                policy_ratios.append(ratio.item())
                num_responses += 1

        # Average loss
        avg_loss = total_loss / num_responses

        # Metrics
        metrics = {
            "loss": avg_loss.item(),
            "kl_div": np.mean(kl_divergences),
            "policy_ratio": np.mean(policy_ratios),
        }

        return avg_loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch of prompts

        Returns:
            Dictionary of training metrics
        """
        prompts = batch["prompts"]

        # 1. Generate responses
        responses, policy_logprobs, ref_logprobs = self.generate_responses(prompts)

        # 2. Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # 3. Compute advantages
        advantages = self.compute_advantages(rewards)

        # 4. Compute policy loss and update
        loss, metrics = self.compute_policy_loss(
            prompts,
            responses,
            policy_logprobs,
            ref_logprobs,
            advantages
        )

        # 5. Backpropagation
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # 6. Gradient clipping and optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Add reward metrics
        metrics["mean_reward"] = rewards.mean().item()
        metrics["max_reward"] = rewards.max().item()
        metrics["min_reward"] = rewards.min().item()
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        return metrics

    def train(self):
        """
        Main training loop.
        """
        print(f"Starting GRPO training for {self.config.num_epochs} epochs...")
        print(f"Total steps: {len(self.train_dataloader) * self.config.num_epochs}")
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Generations per prompt: {self.config.num_generations_per_prompt}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_metrics = []

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )

            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.4f}",
                    "kl": f"{metrics['kl_div']:.4f}"
                })

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb:
                        wandb.log(metrics, step=self.global_step)

                    print(f"\nStep {self.global_step}:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value:.4f}")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                self.global_step += 1

            # Epoch summary
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            print(f"\nEpoch {epoch + 1} Summary:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Save final model
        self.save_checkpoint(final=True)
        print("\nTraining completed!")

    def save_checkpoint(self, final: bool = False):
        """
        Save model checkpoint.

        Args:
            final: Whether this is the final checkpoint
        """
        checkpoint_name = "final_model" if final else f"checkpoint-{self.global_step}"
        checkpoint_path = os.path.join(self.config.output_dir, checkpoint_name)

        print(f"\nSaving checkpoint to {checkpoint_path}...")

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, state_path)

        print(f"Checkpoint saved successfully!")


class SimplePromptDataset(Dataset):
    """
    Simple dataset for GRPO training containing prompts.

    Args:
        prompts: List of prompt strings
    """

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}
