# -*- coding: utf-8 -*-
"""
Reward Model for PPO Training

This module implements a reward model used in Reinforcement Learning from Human Feedback (RLHF).
The reward model predicts scalar rewards for given text sequences, which are used by PPO
to optimize the language model to generate more desirable outputs.

REWARD MODEL ARCHITECTURE:
=========================
The reward model is built on top of the base Qwen3 model and adds a scalar head
that predicts a single reward value for each sequence. The reward represents how
"good" or "desirable" the generated text is according to human preferences.

TRAINING PARADIGM:
==================
1. The reward model is typically trained on human preference data (comparisons)
2. Given pairs of responses (chosen vs rejected), it learns to assign higher rewards
   to preferred responses
3. During PPO training, this model provides reward signals for policy optimization

KEY COMPONENTS:
===============
- Base transformer model for encoding sequences
- Reward head that produces scalar rewards
- Methods for computing rewards for generation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from qwen3.model import Qwen3Config, Qwen3ForCausalLM


@dataclass
class RewardModelOutput(ModelOutput):
    """
    Output class for reward model.

    Args:
        rewards (torch.FloatTensor): Scalar rewards for each sequence, shape (batch_size,)
        hidden_states (torch.FloatTensor): Last hidden states from the model
    """
    rewards: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None


class Qwen3RewardModel(PreTrainedModel):
    """
    Reward Model for RLHF based on Qwen3 architecture.

    This model takes a sequence of tokens as input and outputs a single scalar reward
    value representing the quality or desirability of that sequence according to
    learned human preferences.

    ARCHITECTURE:
    =============
    Input Tokens → Qwen3 Encoder → Last Token Hidden State → Reward Head → Scalar Reward

    WHY LAST TOKEN?
    ===============
    In causal language models, the last token's hidden state contains information about
    the entire sequence due to the causal attention mechanism. This makes it an ideal
    representation for scoring the complete sequence.

    TRAINING OBJECTIVE:
    ===================
    During training, the reward model learns to maximize the log-likelihood of human
    preferences using a pairwise ranking loss:

    Loss = -log(σ(r_chosen - r_rejected))

    Where:
    - r_chosen: reward for the preferred response
    - r_rejected: reward for the rejected response
    - σ: sigmoid function

    USAGE IN PPO:
    =============
    During PPO training, this model provides reward signals for generated responses:
    1. Generate response with current policy
    2. Compute reward for generated response
    3. Use reward to compute advantage estimates
    4. Update policy to maximize expected reward

    Args:
        config (Qwen3Config): Configuration for the base model
    """
    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        # Base model for encoding sequences (without LM head)
        # We'll use the Qwen3 transformer layers
        from qwen3.model import RMSNorm, Qwen3Block

        # Token embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        # Stack of transformer blocks
        self.layers = nn.ModuleList([Qwen3Block(config) for _ in range(config.n_layer)])
        # Final normalization
        self.norm = RMSNorm(config.n_embd)

        # Reward head: projects last hidden state to scalar reward
        # Using a simple linear layer for the reward head
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> RewardModelOutput:
        """
        Forward pass to compute rewards for input sequences.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask (not used in this implementation)
            return_dict: Whether to return structured output

        Returns:
            RewardModelOutput containing scalar rewards for each sequence
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        hidden_states = self.embed_tokens(input_ids)

        # 2. Pass through transformer blocks
        for block in self.layers:
            layer_outputs = block(
                hidden_states,
                attention_mask=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        # 3. Final normalization
        hidden_states = self.norm(hidden_states)

        # 4. Extract last token hidden state for each sequence
        # In causal LMs, the last token contains information about the full sequence
        # Shape: (batch_size, seq_len, n_embd) -> (batch_size, n_embd)

        # If attention_mask is provided, use it to find the last non-padded token
        if attention_mask is not None:
            # Get the index of the last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]
        else:
            # If no mask, just use the last token
            last_hidden_states = hidden_states[:, -1, :]

        # 5. Compute scalar reward from last hidden state
        # Shape: (batch_size, n_embd) -> (batch_size, 1) -> (batch_size,)
        rewards = self.reward_head(last_hidden_states).squeeze(-1)

        return RewardModelOutput(
            rewards=rewards,
            hidden_states=hidden_states,
        )

    @classmethod
    def from_pretrained_policy(cls, policy_model: Qwen3ForCausalLM):
        """
        Initialize reward model from a pretrained policy model.

        This method creates a reward model and copies weights from a pretrained
        Qwen3ForCausalLM model. This is useful for initializing the reward model
        with the same pretrained representations as the policy.

        Args:
            policy_model: Pretrained Qwen3ForCausalLM model

        Returns:
            Qwen3RewardModel with weights copied from policy model
        """
        # Create reward model with same config
        reward_model = cls(policy_model.config)

        # Copy weights from policy model (excluding LM head)
        reward_model.embed_tokens.load_state_dict(policy_model.embed_tokens.state_dict())
        for i in range(len(reward_model.layers)):
            reward_model.layers[i].load_state_dict(policy_model.layers[i].state_dict())
        reward_model.norm.load_state_dict(policy_model.norm.state_dict())

        # Reward head is randomly initialized
        # It will be trained to predict human preferences

        return reward_model


def compute_pairwise_loss(
    reward_model: Qwen3RewardModel,
    chosen_input_ids: torch.LongTensor,
    rejected_input_ids: torch.LongTensor,
    chosen_attention_mask: Optional[torch.Tensor] = None,
    rejected_attention_mask: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    """
    Compute pairwise ranking loss for reward model training.

    This function implements the standard ranking loss used in RLHF:
    Loss = -log(sigmoid(reward_chosen - reward_rejected))

    The loss encourages the model to assign higher rewards to chosen (preferred)
    responses compared to rejected responses.

    MATHEMATICAL FORMULATION:
    =========================
    Given human preference data (chosen, rejected), we want:
    P(chosen > rejected) = σ(r_chosen - r_rejected)

    Maximizing log-likelihood gives us the loss function:
    L = -log(σ(r_chosen - r_rejected))

    Args:
        reward_model: The reward model to train
        chosen_input_ids: Token IDs for chosen/preferred responses
        rejected_input_ids: Token IDs for rejected responses
        chosen_attention_mask: Attention mask for chosen responses
        rejected_attention_mask: Attention mask for rejected responses

    Returns:
        Scalar loss value
    """
    # Compute rewards for chosen and rejected responses
    chosen_rewards = reward_model(
        input_ids=chosen_input_ids,
        attention_mask=chosen_attention_mask,
    ).rewards

    rejected_rewards = reward_model(
        input_ids=rejected_input_ids,
        attention_mask=rejected_attention_mask,
    ).rewards

    # Compute ranking loss: -log(sigmoid(r_chosen - r_rejected))
    # Using logsigmoid for numerical stability: log(sigmoid(x)) = -log(1 + exp(-x))
    loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss
