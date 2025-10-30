# -*- coding: utf-8 -*-
"""
PPO Trainer for Reinforcement Learning from Human Feedback (RLHF)

This module implements the Proximal Policy Optimization (PPO) algorithm for fine-tuning
large language models with reward signals. PPO is the key algorithm used in RLHF
pipelines like ChatGPT, where models are optimized to maximize rewards from a
learned reward model.

PPO ALGORITHM OVERVIEW:
=======================
PPO is a policy gradient method that constrains policy updates to stay close to
the previous policy, preventing destructively large updates. This is achieved through
a clipped surrogate objective.

KEY CONCEPTS:
=============
1. **Policy**: The language model that generates text (π_θ)
2. **Reference Policy**: Frozen copy of the initial policy (π_ref)
3. **Reward Model**: Provides scalar rewards for generated text
4. **Value Model**: Estimates expected future rewards (used for advantage estimation)
5. **KL Penalty**: Keeps policy close to reference to prevent reward hacking

PPO OBJECTIVE:
==============
L_CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

Where:
- r_t(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)
- A_t: advantage estimate (how much better than expected)
- ε: clip parameter (typically 0.2)

ADDITIONAL COMPONENTS:
======================
- KL penalty: -β * KL(π_θ || π_ref)  (keeps policy close to reference)
- Value loss: MSE between value predictions and actual returns
- Entropy bonus: encourages exploration

TRAINING FLOW:
==============
1. Generate responses with current policy
2. Compute rewards using reward model
3. Compute KL penalty against reference policy
4. Calculate advantages using GAE (Generalized Advantage Estimation)
5. Update policy using clipped PPO objective
6. Update value model to predict returns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from qwen3.model import Qwen3ForCausalLM, Qwen3Config
from ppo.reward_model import Qwen3RewardModel


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    Args:
        model_name: Name or path of the pretrained model
        learning_rate: Learning rate for policy optimization
        batch_size: Number of sequences per batch
        mini_batch_size: Size of mini-batches for PPO updates
        ppo_epochs: Number of PPO epochs per batch
        gamma: Discount factor for returns
        lam: Lambda parameter for GAE (Generalized Advantage Estimation)
        clip_range: Clipping parameter for PPO (epsilon)
        clip_range_value: Clipping parameter for value loss
        vf_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        kl_coef: Coefficient for KL penalty
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence (for early stopping)
        init_kl_coef: Initial KL coefficient (for adaptive KL)
        adap_kl_ctrl: Whether to use adaptive KL coefficient
    """
    model_name: str = "Qwen/Qwen2.5-1.5B"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    ppo_epochs: int = 4
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.1
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    max_grad_norm: float = 1.0
    target_kl: float = 0.01
    init_kl_coef: float = 0.1
    adap_kl_ctrl: bool = True


class PPOTrainer:
    """
    Trainer for Proximal Policy Optimization (PPO) with language models.

    This class implements the core PPO algorithm for RLHF, handling:
    - Policy updates with clipped surrogate objective
    - Value function updates
    - KL penalty computation
    - Advantage estimation using GAE
    - Adaptive KL coefficient

    TRAINING PROCESS:
    =================
    1. **Rollout Phase**: Generate responses with current policy
    2. **Evaluation Phase**: Compute rewards and advantages
    3. **Update Phase**: Multiple PPO epochs over the batch
    4. **Logging Phase**: Track metrics and KL divergence

    Args:
        config: PPO configuration
        model: The policy model to train (Qwen3ForCausalLM)
        ref_model: Reference model for KL penalty (frozen)
        reward_model: Model that provides reward signals
        tokenizer: Tokenizer for the model
    """

    def __init__(
        self,
        config: PPOConfig,
        model: Qwen3ForCausalLM,
        ref_model: Qwen3ForCausalLM,
        reward_model: Qwen3RewardModel,
        tokenizer,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()

        # Value head: estimates expected future rewards
        # Added on top of the policy model
        self.value_head = nn.Linear(model.config.n_embd, 1, bias=False)
        self.value_head.to(model.device)

        # Optimizer for both policy and value head
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
        )

        # Adaptive KL coefficient
        self.kl_coef = config.init_kl_coef

    def compute_rewards(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute rewards for generated sequences using the reward model.

        Args:
            input_ids: Generated token IDs
            attention_mask: Attention mask for the sequences

        Returns:
            Scalar rewards for each sequence
        """
        with torch.no_grad():
            rewards = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).rewards

        return rewards

    def compute_values(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute value estimates for each token position.

        The value function estimates expected future rewards from each state,
        which is used for advantage estimation.

        Args:
            hidden_states: Hidden states from the model, shape (batch, seq_len, hidden_dim)
            attention_mask: Attention mask

        Returns:
            Value estimates for each position, shape (batch, seq_len)
        """
        # Apply value head to get value estimates
        values = self.value_head(hidden_states).squeeze(-1)
        return values

    def get_log_probs(
        self,
        model: Qwen3ForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute log probabilities for the given sequences.

        Args:
            model: Model to use (policy or reference)
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (log_probs, hidden_states)
            - log_probs: Log probabilities for each token, shape (batch, seq_len)
            - hidden_states: Last layer hidden states
        """
        # Forward pass through model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer

        # Compute log probabilities
        # Shift logits and input_ids for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Get log probs for the actual tokens
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather log probs for the selected tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs, hidden_states

    def compute_advantages(
        self,
        rewards: torch.FloatTensor,
        values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        GAE provides a way to estimate advantages that balances bias and variance.
        It uses a parameter λ to interpolate between TD(0) and Monte Carlo estimates.

        MATHEMATICAL FORMULATION:
        =========================
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        Where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error

        Args:
            rewards: Rewards for each sequence, shape (batch,)
            values: Value estimates for each position, shape (batch, seq_len)
            attention_mask: Attention mask

        Returns:
            Tuple of (advantages, returns)
            - advantages: GAE advantages, shape (batch, seq_len)
            - returns: TD(λ) returns, shape (batch, seq_len)
        """
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        # Process each sequence in the batch
        for b in range(batch_size):
            # Get sequence length (excluding padding)
            if attention_mask is not None:
                seq_length = attention_mask[b].sum().item()
            else:
                seq_length = seq_len

            # Initialize
            last_gae = 0
            reward = rewards[b].item()

            # Backward pass through the sequence
            for t in reversed(range(seq_length)):
                if t == seq_length - 1:
                    # Last position: reward comes from reward model
                    next_value = 0.0
                    delta = reward + self.config.gamma * next_value - values[b, t]
                else:
                    # Middle positions: bootstrap from next value
                    next_value = values[b, t + 1]
                    delta = self.config.gamma * next_value - values[b, t]

                # GAE computation
                last_gae = delta + self.config.gamma * self.config.lam * last_gae
                advantages[b, t] = last_gae
                returns[b, t] = advantages[b, t] + values[b, t]

        return advantages, returns

    def ppo_loss(
        self,
        old_log_probs: torch.FloatTensor,
        log_probs: torch.FloatTensor,
        advantages: torch.FloatTensor,
        values: torch.FloatTensor,
        returns: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Compute PPO loss with clipped surrogate objective.

        LOSS COMPONENTS:
        ================
        1. Policy Loss (L_CLIP): Clipped surrogate objective
        2. Value Loss: MSE between predicted values and returns
        3. Entropy Bonus: Encourages exploration

        Args:
            old_log_probs: Log probs from old policy (rollout)
            log_probs: Log probs from current policy
            advantages: Advantage estimates
            values: Value predictions
            returns: Target returns
            attention_mask: Attention mask

        Returns:
            Tuple of (total_loss, stats_dict)
        """
        # Compute probability ratio: π_new / π_old
        ratio = torch.exp(log_probs - old_log_probs)

        # Normalize advantages (for stability)
        if attention_mask is not None:
            # Only normalize over valid (non-padded) positions
            mask = attention_mask[:, 1:].float()  # Shift for next-token prediction
            normalized_advantages = (advantages - (advantages * mask).sum() / mask.sum()) / (
                (advantages * mask).std() + 1e-8
            )
        else:
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped surrogate objective
        policy_loss_unclipped = ratio * normalized_advantages
        policy_loss_clipped = (
            torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
            * normalized_advantages
        )
        policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped)

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()
            policy_loss = (policy_loss * mask).sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()

        # Value loss (clipped)
        value_loss_unclipped = (values - returns) ** 2
        value_loss_clipped = (
            values - returns.clamp(
                values - self.config.clip_range_value,
                values + self.config.clip_range_value,
            )
        ) ** 2
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        if attention_mask is not None:
            value_loss = (value_loss * attention_mask.float()).sum() / attention_mask.float().sum()
        else:
            value_loss = value_loss.mean()

        # Total loss
        total_loss = policy_loss + self.config.vf_coef * value_loss

        # Collect statistics
        stats = {
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/total": total_loss.item(),
            "ratio/mean": ratio.mean().item(),
            "ratio/max": ratio.max().item(),
            "ratio/min": ratio.min().item(),
            "advantages/mean": advantages.mean().item(),
            "advantages/std": advantages.std().item(),
        }

        return total_loss, stats

    def step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform one PPO training step.

        This method implements the complete PPO training loop:
        1. Generate rollout data (log probs, values, rewards)
        2. Compute advantages using GAE
        3. Perform multiple PPO update epochs
        4. Track and return statistics

        Args:
            input_ids: Generated sequences
            attention_mask: Attention mask

        Returns:
            Dictionary of training statistics
        """
        device = self.model.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # === ROLLOUT PHASE ===
        with torch.no_grad():
            # Get log probs and hidden states from current policy
            old_log_probs, hidden_states = self.get_log_probs(
                self.model, input_ids, attention_mask
            )

            # Get log probs from reference policy (for KL penalty)
            ref_log_probs, _ = self.get_log_probs(
                self.ref_model, input_ids, attention_mask
            )

            # Compute rewards
            rewards = self.compute_rewards(input_ids, attention_mask)

            # Compute KL penalty
            kl_div = (old_log_probs - ref_log_probs).sum(dim=-1)
            rewards = rewards - self.kl_coef * kl_div

            # Compute values (detach hidden states)
            values = self.compute_values(hidden_states.detach(), attention_mask)

            # Compute advantages and returns
            advantages, returns = self.compute_advantages(rewards, values, attention_mask)

        # === UPDATE PHASE ===
        all_stats = []

        for epoch in range(self.config.ppo_epochs):
            # Forward pass with current policy
            log_probs, hidden_states = self.get_log_probs(
                self.model, input_ids, attention_mask
            )

            # Compute values with gradient
            values = self.compute_values(hidden_states, attention_mask)

            # Compute PPO loss
            loss, stats = self.ppo_loss(
                old_log_probs.detach(),
                log_probs,
                advantages.detach(),
                values,
                returns.detach(),
                attention_mask,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.config.max_grad_norm,
            )

            # Optimizer step
            self.optimizer.step()

            all_stats.append(stats)

        # === LOGGING ===
        # Average statistics across epochs
        final_stats = {
            key: np.mean([s[key] for s in all_stats])
            for key in all_stats[0].keys()
        }

        # Add additional metrics
        final_stats["kl_div"] = kl_div.mean().item()
        final_stats["rewards"] = rewards.mean().item()
        final_stats["kl_coef"] = self.kl_coef

        # Adaptive KL coefficient
        if self.config.adap_kl_ctrl:
            if kl_div.mean().item() > self.config.target_kl * 1.5:
                self.kl_coef *= 1.5
            elif kl_div.mean().item() < self.config.target_kl / 1.5:
                self.kl_coef /= 1.5

        return final_stats

    def train(
        self,
        prompts: List[str],
        num_epochs: int = 1,
        max_length: int = 128,
    ) -> List[Dict[str, Any]]:
        """
        Full training loop for PPO.

        Args:
            prompts: List of prompt strings to generate from
            num_epochs: Number of training epochs
            max_length: Maximum generation length

        Returns:
            List of statistics for each step
        """
        self.model.train()
        all_stats = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Process prompts in batches
            for i in tqdm(range(0, len(prompts), self.config.batch_size)):
                batch_prompts = prompts[i : i + self.config.batch_size]

                # Tokenize prompts
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)

                # Generate responses
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Create attention mask for generated sequences
                gen_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()

                # PPO step
                stats = self.step(generated_ids, gen_attention_mask)
                all_stats.append(stats)

                # Print stats periodically
                if (i // self.config.batch_size) % 10 == 0:
                    print(f"Step {i // self.config.batch_size}: {stats}")

        return all_stats
