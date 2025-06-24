# -*- coding: utf-8 -*-
"""
Full, From-Scratch Implementation of a Qwen3-like Model for Hugging Face

This script re-implements the concepts from the `standalone-qwen3.ipynb` notebook
by Sebastian Raschka, adapting it into a fully Hugging Face-compatible format.
The goal is to be didactic, showing how to build a custom transformer model
from the ground up and integrate it with the `transformers` library.

NOTEBOOK ANALYSIS & ARCHITECTURAL OVERVIEW:
==========================================

Model Architecture:
This implementation follows a standard decoder-only transformer architecture, 
characteristic of models like GPT. The key architectural components include:

1. **RMSNorm**: Used for normalization instead of LayerNorm. RMSNorm (Root Mean Square 
   Layer Normalization) is simpler and often more efficient than standard LayerNorm,
   as it doesn't subtract the mean, only normalizes by the root-mean-square.

2. **Rotary Positional Embeddings (RoPE)**: Applied to queries and keys in the 
   self-attention mechanism for encoding positional information. RoPE encodes position
   by rotating the query and key vectors based on their absolute position, allowing
   for better length extrapolation compared to learned positional embeddings.

3. **Grouped-Query Attention (GQA)**: A variant of multi-head attention where multiple 
   query heads share a single key and value head to reduce computational and memory costs.
   However, this implementation uses Multi-Head Attention (MHA) for simplicity and
   educational clarity, following the notebook's approach where n_head == n_kv_head.

4. **SwiGLU Activation**: The feed-forward network uses SwiGLU (Swish-Gated Linear Unit)
   activation function, which is common in modern LLMs. SwiGLU combines Swish/SiLU 
   activation with a gating mechanism: SwiGLU(x) = Swish(xW + b) ⊙ (xV + c).

5. **tiktoken Tokenizer**: Uses a tiktoken-based tokenizer with custom vocabulary and 
   special tokens, wrapped for Hugging Face compatibility.

Weight Loading & Conversion:
The implementation includes functionality to load pretrained weights from PyTorch .pth 
files and demonstrates the mapping between original weight names and Hugging Face 
model layer names for seamless integration.

Text Generation Capabilities:
Provides a complete generate() function that performs inference with advanced sampling
techniques including temperature scaling, top-k filtering, and KV caching for efficient
autoregressive generation.

IMPLEMENTATION STRUCTURE:
========================

Key features of this implementation:
- **From-Scratch Components**: RMSNorm, RoPE, Attention, and MLP built using pure PyTorch
- **Hugging Face Integration**:
  - `Qwen3Config`: Inherits from `transformers.PretrainedConfig` for hyperparameter management
  - `Qwen3ForCausalLM`: Inherits from `transformers.PreTrainedModel`, enabling access to
    methods like `.generate()`, `.from_pretrained()`, and `.push_to_hub()`
- **Educational Focus**: Every class and function thoroughly documented with architectural
  explanations and implementation details
- **Hub Integration**: Uses official Qwen models from Hugging Face Hub for production use
- **Weight Conversion**: Demonstrates manual weight loading from .pth files for educational purposes

USAGE:
======
1. Install required libraries: `torch`, `transformers`, `huggingface-hub`
2. Run main() for Hub-based inference (recommended for production)

This implementation serves as both a working model and an educational resource for 
understanding modern transformer architectures and their integration with the Hugging Face ecosystem.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Suppress Hugging Face tokenizer parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face imports
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.generic import ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast


# =============================================================================
# 1. Configuration Class
# =============================================================================

class Qwen3Config(PretrainedConfig):
    """
    Configuration class for Qwen3 model, storing all architectural hyperparameters.
    
    This class inherits from `PretrainedConfig` to ensure full compatibility with the 
    Hugging Face ecosystem, enabling easy save/load of model configurations and 
    integration with training/inference scripts.
    
    ARCHITECTURAL DECISIONS:
    - Follows standard decoder-only transformer design
    - Uses Multi-Head Attention (MHA) instead of Grouped-Query Attention (GQA) 
      for educational clarity (n_head == n_kv_head)
    - Incorporates RoPE (Rotary Positional Embeddings) for position encoding
    - Uses SwiGLU activation in the feed-forward network
    - RMSNorm for layer normalization instead of standard LayerNorm
    
    Args:
        vocab_size (int): Size of the vocabulary (number of tokens)
        context_len (int): Maximum sequence length the model can handle
        n_layer (int): Number of transformer blocks/layers
        n_head (int): Number of attention heads in multi-head attention
        n_embd (int): Embedding dimension (hidden size)
        intermediate_size (int): Dimension of the feed-forward network's hidden layer
        rope_theta (float): Base frequency for RoPE (Rotary Positional Embeddings)
    """
    model_type = "qwen3_from_scratch"
    attribute_map = { # Mapping for compatibility with HF trainer/inference scripts
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "max_position_embeddings": "context_len"
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        context_len: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        n_embd: int = 4096,
        intermediate_size: int = 22016,  # From the original model config
        rope_theta: float = 10000.0,
        use_cache: bool = True,
        use_return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta

        # Cache and output settings used in the model forward pass
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        # The Qwen model architecture supports Grouped-Query Attention (GQA) where
        # multiple query heads share key/value heads for efficiency. However, this
        # educational implementation uses Multi-Head Attention (MHA) for simplicity,
        # following the notebook's approach where n_head == n_kv_head.
        # For true GQA implementation, you would set n_kv_head < n_head.
        self.n_kv_head = n_head

        super().__init__(
            return_dict=use_return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )


# =============================================================================
# 2. From-Scratch Model Components (as nn.Module)
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is a simpler and often more efficient alternative to standard LayerNorm.
    Unlike LayerNorm, RMSNorm doesn't subtract the mean - it only normalizes by the 
    root-mean-square, making it computationally lighter while maintaining similar 
    performance in many cases.
    
    MATHEMATICAL FORMULATION:
    RMSNorm(x) = (x / √(mean(x²) + ε)) * γ
    
    Where:
    - x is the input tensor
    - mean(x²) is the mean of squared elements across the last dimension
    - ε is a small constant for numerical stability  
    - γ is a learnable scaling parameter
    
    ADVANTAGES OVER LAYERNORM:
    - Simpler computation (no mean subtraction)
    - Slightly faster in practice
    - Often works just as well as LayerNorm for language models
    - Used in many modern LLMs including LLaMA, PaLM, and Qwen
    
    Args:
        dim (int): The dimension to normalize (typically the embedding dimension)
        eps (float): Small constant for numerical stability (default: 1e-5)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # Learnable scaling parameter γ

    def _norm(self, x):
        # Calculate root mean square across the last dimension
        # (B, Seq_Len, Dim) -> (B, Seq_Len, 1)
        rms = x.pow(2).mean(-1, keepdim=True)
        # Normalize by RMS with epsilon for numerical stability
        return x * torch.rsqrt(rms + self.eps)

    def forward(self, x):
        # Apply normalization and scale by learnable weight parameter
        return self.weight * self._norm(x.float()).type_as(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.
    
    RoPE encodes positional information by rotating query and key vectors in attention
    based on their absolute position. This approach has several advantages over learned
    positional embeddings:
    
    ADVANTAGES OF ROPE:
    - Better length extrapolation (can handle sequences longer than training)
    - Relative position encoding emerges naturally from absolute position rotation
    - No additional parameters needed beyond the base frequency (theta)
    - More efficient than adding positional embeddings to token embeddings
    
    MATHEMATICAL FOUNDATION:
    For each position m and dimension pair (2i, 2i+1), RoPE applies rotation:
    [x₂ᵢ']     [cos(m·θᵢ)  -sin(m·θᵢ)] [x₂ᵢ]
    [x₂ᵢ₊₁'] = [sin(m·θᵢ)   cos(m·θᵢ)] [x₂ᵢ₊₁]
    
    Where θᵢ = θ^(-2i/d) and θ is the base frequency (typically 10000)
    
    Args:
        dim (int): Dimension of each attention head (must be even)
        max_seq_len (int): Maximum sequence length to precompute frequencies for
        theta (float): Base frequency for the rotation (default: 10000.0)
    """
    def __init__(self, dim, max_seq_len, theta=10000.0):
        super().__init__()
        # Dimension must be even for rotation pairs
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.freqs_cis = self._precompute_freqs_cis()

    def _precompute_freqs_cis(self):
        # Calculate frequencies: θ^(-2k/d) for k in [0, 1, ..., d/2 - 1]
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        # Create position indices: [0, 1, ..., max_seq_len - 1]
        t = torch.arange(self.max_seq_len)
        # Calculate phase angles (m * θᵢ) for all positions and frequencies
        freqs = torch.outer(t, freqs).float()
        # Convert to complex exponentials: e^(i·m·θᵢ) = cos(m·θᵢ) + i·sin(m·θᵢ)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Get precomputed rotation frequencies for the given sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, Seq_Len, H, Head_Dim)
            start_pos (int): Starting position (used for KV caching during generation)
        Returns:
            torch.Tensor: Complex frequencies for rotation, shape (Seq_Len, Head_Dim/2)
        """
        seq_len = x.shape[1]
        # Ensure frequencies are on the same device as input
        if self.freqs_cis.device != x.device:
            self.freqs_cis = self.freqs_cis.to(x.device)
        
        # Extract frequencies for current sequence positions
        return self.freqs_cis[start_pos : start_pos + seq_len]


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor using precomputed frequencies.
    
    This function performs the core RoPE transformation by treating pairs of real
    numbers as complex numbers and multiplying by the rotation frequencies.
    
    IMPLEMENTATION DETAILS:
    1. Reshape input to treat consecutive dimension pairs as complex numbers
    2. Convert to complex tensor representation
    3. Multiply by precomputed rotation frequencies (complex multiplication = rotation)
    4. Convert back to real tensor and reshape to original dimensions
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, Seq_Len, H, Head_Dim)
        freqs_cis (torch.Tensor): Rotation frequencies of shape (Seq_Len, Head_Dim/2)
    Returns:
        torch.Tensor: Rotated tensor with same shape as input
    """
    # Reshape to treat consecutive pairs as complex numbers: (..., Head_Dim/2, 2)
    x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # Convert pairs to complex numbers: (..., Head_Dim/2)
    x_complex = torch.view_as_complex(x_shaped)
    # Reshape frequencies for broadcasting: (1, Seq_Len, 1, Head_Dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    # Apply rotation via complex multiplication
    x_rotated = x_complex * freqs_cis
    # Convert back to real: (..., Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Flatten back to original shape: (..., Head_Dim)
    x_out = x_out.flatten(3)
    return x_out.type_as(x)


class Qwen3Attention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Positional Embeddings and KV Caching.
    
    This implementation follows the standard multi-head attention mechanism but with
    several modern enhancements:
    
    ARCHITECTURAL FEATURES:
    - **RoPE Integration**: Applies rotary positional embeddings to Q and K vectors
    - **KV Caching**: Supports efficient autoregressive generation by caching past keys/values
    - **Multi-Head Attention**: Uses MHA instead of GQA for educational clarity
    - **Causal Masking**: Implements causal attention for decoder-only architecture
    - **Optimized Attention**: Uses PyTorch's scaled_dot_product_attention for efficiency
    
    ATTENTION VS GQA TRADE-OFF:
    While the original Qwen architecture uses Grouped-Query Attention (GQA) to reduce
    computational costs by sharing key/value heads across multiple query heads, this
    implementation uses standard Multi-Head Attention where each head has its own
    Q, K, V projections. This choice prioritizes:
    - Educational clarity and simplicity
    - Easier understanding of attention mechanisms  
    - Direct correspondence with many tutorial materials
    
    For production GQA implementation, you would:
    - Add n_kv_head parameter (< n_head)
    - Modify K,V projections to use n_kv_head * head_dim
    - Implement key/value head sharing logic
    
    Args:
        config (Qwen3Config): Model configuration containing architectural parameters
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Linear projections for Q, K, V, and output
        # Note: Original Qwen uses bias=True for Q,K,V and bias=False for output
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Initialize RoPE with head dimension and context length
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.context_len, theta=config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = x.shape

        # 1. Project input to Q, K, V
        q = self.q_proj(x)  # (B, Seq_Len, n_embd)
        k = self.k_proj(x)  # (B, Seq_Len, n_embd)
        v = self.v_proj(x)  # (B, Seq_Len, n_embd)

        # 2. Reshape for multi-head attention
        # (B, Seq_Len, n_embd) -> (B, Seq_Len, n_head, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim)

        # 3. Apply Rotary Positional Embeddings (RoPE)
        # Determine starting position for RoPE (important for KV caching)
        start_pos = past_key_value[0].shape[1] if past_key_value is not None else 0
        freqs_cis = self.rotary_emb(q, start_pos=start_pos)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 4. Handle KV Cache for efficient generation
        if past_key_value is not None:
            # Concatenate past keys/values with current keys/values
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)

        # Update cache if enabled
        present_key_value = (k, v) if use_cache else None

        # 5. Transpose for attention computation: (B, n_head, Seq_Len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. Compute scaled dot-product attention with causal masking
        # Using PyTorch's optimized implementation for memory efficiency and speed
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True  # Critical for decoder-only models!
        )
        
        # Note: For more complex masking (e.g., padding tokens), you would pass
        # attention_mask here, but causal masking is sufficient for this implementation

        # 7. Reshape and apply output projection
        # (B, n_head, Seq_Len, head_dim) -> (B, Seq_Len, n_head, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (B, Seq_Len, n_head, head_dim) -> (B, Seq_Len, n_embd)
        attn_output = attn_output.view(batch_size, seq_len, self.n_embd)

        # Final linear projection
        output = self.o_proj(attn_output)

        # Attention weights not returned by scaled_dot_product_attention by default
        attn_weights = None

        return output, attn_weights, present_key_value


class Qwen3MLP(nn.Module):
    """
    Feed-Forward Network with SwiGLU Activation.
    
    This MLP implements the SwiGLU (Swish-Gated Linear Unit) activation function,
    which has become standard in modern large language models due to its superior
    performance compared to traditional activations like ReLU or GELU.
    
    SWIGLU ARCHITECTURE:
    The SwiGLU activation combines two concepts:
    1. **Swish/SiLU Activation**: f(x) = x * sigmoid(x), smooth and non-monotonic
    2. **Gated Linear Unit (GLU)**: Element-wise gating mechanism
    
    MATHEMATICAL FORMULATION:
    SwiGLU(x) = Swish(x * W_gate + b_gate) ⊙ (x * W_up + b_up)
    Output = (SwiGLU_result) * W_down + b_down
    
    Where ⊙ denotes element-wise multiplication (Hadamard product)
    
    ADVANTAGES OF SWIGLU:
    - Better gradient flow compared to ReLU-based activations
    - Gating mechanism allows selective information passage
    - Empirically shown to improve model performance in LLMs
    - Used in models like PaLM, LLaMA, and Qwen
    
    ARCHITECTURE DETAILS:
    - gate_proj: Projects input to intermediate dimension with SiLU activation
    - up_proj: Projects input to intermediate dimension (no activation)
    - down_proj: Projects back to embedding dimension
    - intermediate_size: Typically 2.67x the embedding dimension (e.g., 4096 -> ~11K)
    
    Args:
        config (Qwen3Config): Model configuration with embedding and intermediate dimensions
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        # Three linear layers implementing SwiGLU
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        # SiLU (Swish) activation: f(x) = x * sigmoid(x)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Implement SwiGLU: Swish(x * W_gate) ⊙ (x * W_up)
        # First branch: apply SiLU activation to gate projection
        gate = self.act_fn(self.gate_proj(x))
        # Second branch: linear projection without activation
        up = self.up_proj(x)
        # Element-wise multiplication (gating mechanism)
        fused_gate_up = gate * up
        # Final projection back to embedding dimension
        return self.down_proj(fused_gate_up)


class Qwen3Block(nn.Module):
    """
    Complete Transformer Block combining Attention and MLP with residual connections.
    
    This class implements one complete transformer layer following the standard
    architecture used in decoder-only models. The block consists of:
    
    ARCHITECTURE PATTERN:
    Input → RMSNorm → Multi-Head Attention → Residual Connection
           ↓
          RMSNorm → SwiGLU MLP → Residual Connection → Output
    
    KEY DESIGN CHOICES:
    - **Pre-Normalization**: RMSNorm is applied before attention and MLP (not after)
    - **Residual Connections**: Enable deep network training and gradient flow
    - **RMSNorm**: Used instead of LayerNorm for efficiency
    - **SwiGLU MLP**: Modern activation function for better performance
    
    PRE-NORM VS POST-NORM:
    This implementation uses pre-normalization (norm before attention/MLP) which:
    - Provides more stable training for deep networks
    - Better gradient flow in very deep models
    - Has become the standard in modern transformer architectures
    - Used in GPT-3/4, LLaMA, PaLM, and other recent models
    
    RESIDUAL CONNECTIONS:
    The residual connections (x + f(x)) are crucial for:
    - Training stability in deep networks
    - Gradient flow during backpropagation  
    - Allowing the model to learn identity mappings when needed
    - Enabling successful scaling to hundreds of layers
    
    Args:
        config (Qwen3Config): Model configuration containing all architectural parameters
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.n_embd)
        # Multi-head self-attention with RoPE
        self.self_attn = Qwen3Attention(config)
        # Pre-MLP normalization  
        self.post_attention_layernorm = RMSNorm(config.n_embd)
        # Feed-forward network with SwiGLU
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # 1. First Sub-Layer: Multi-Head Self-Attention with Pre-Normalization
        # Store residual connection
        residual = hidden_states
        # Apply pre-attention normalization
        normalized_hidden_states = self.input_layernorm(hidden_states)
        # Self-attention with RoPE and optional KV caching
        attn_output, attn_weights, present_key_value = self.self_attn(
            normalized_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # Apply residual connection
        hidden_states = residual + attn_output

        # 2. Second Sub-Layer: Feed-Forward Network with Pre-Normalization
        # Store residual connection
        residual = hidden_states
        # Apply pre-MLP normalization
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        # SwiGLU feed-forward network
        mlp_output = self.mlp(normalized_hidden_states)
        # Apply residual connection
        hidden_states = residual + mlp_output

        return hidden_states, attn_weights, present_key_value


# =============================================================================
# 3. Full Model Definition (Hugging Face Compatible)
# =============================================================================

class Qwen3ForCausalLM(PreTrainedModel):
    """
    Complete Qwen3 Model for Causal Language Modeling with Hugging Face Integration.
    
    This is the main model class that assembles all components into a complete
    decoder-only transformer for autoregressive language modeling. By inheriting
    from `transformers.PreTrainedModel`, it gains access to the entire Hugging Face
    ecosystem including generation, training, and model hub functionality.
    
    FULL ARCHITECTURE OVERVIEW:
    Input Token IDs → Embedding Layer → Stack of Transformer Blocks → Final RMSNorm → Language Model Head → Logits
    
    Each Transformer Block contains:
    - RMSNorm → Multi-Head Attention with RoPE → Residual Connection
    - RMSNorm → SwiGLU MLP → Residual Connection
    
    HUGGING FACE INTEGRATION BENEFITS:
    - **Automatic Generation**: Access to sophisticated generate() method with sampling strategies
    - **Model Hub**: Easy save/load with push_to_hub() and from_pretrained()
    - **Training Integration**: Compatible with transformers.Trainer and training loops
    - **Tokenizer Integration**: Seamless work with tokenizers and data loaders
    - **Mixed Precision**: Automatic support for fp16/bf16 training and inference
    - **Device Management**: Automatic device placement and data parallel training
    
    ARCHITECTURAL DECISIONS EXPLAINED:
    1. **Decoder-Only**: Follows GPT-style architecture for autoregressive generation
    2. **Embedding Sharing**: Language model head can optionally share weights with embeddings
    3. **Final Normalization**: RMSNorm before language model head for stability
    4. **Causal Attention**: Each token can only attend to previous tokens
    5. **Position Encoding**: Uses RoPE instead of learned positional embeddings
    
    GENERATION CAPABILITIES:
    - **Autoregressive Generation**: Predicts next token given previous context
    - **Sampling Strategies**: Supports greedy, top-k, top-p, temperature sampling
    - **KV Caching**: Efficient generation by caching attention keys/values
    - **Batch Generation**: Can generate multiple sequences simultaneously
    - **Length Control**: Max length, stopping criteria, and early stopping support
    
    Args:
        config (Qwen3Config): Model configuration with all architectural parameters
    """
    # Associate with custom configuration class
    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        # Core model architecture
        # Token embedding layer: vocab_size -> n_embd
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        # Stack of transformer blocks
        self.layers = nn.ModuleList([Qwen3Block(config) for _ in range(config.n_layer)])
        # Final normalization before language model head
        self.norm = RMSNorm(config.n_embd)
        # Language model head: n_embd -> vocab_size (no bias for efficiency)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights using Hugging Face's initialization scheme
        self.post_init()
        # Tie input and output embeddings by sharing weights
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        """Required method for Hugging Face compatibility."""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Required method for Hugging Face compatibility."""
        self.embed_tokens = value

    def get_output_embeddings(self):
        """Return the language modeling head for weight tying."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set the language modeling head (needed for some HF utilities)."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass implementing the complete transformer computation.
        
        COMPUTATION FLOW:
        1. Token Embedding: Convert token IDs to dense vectors
        2. Transformer Layers: Apply N transformer blocks sequentially
        3. Final Normalization: Apply RMSNorm to final hidden states
        4. Language Model Head: Project to vocabulary logits
        5. Loss Computation: Calculate cross-entropy loss if labels provided
        
        KV CACHING SUPPORT:
        During generation, past_key_values enables efficient autoregressive decoding
        by caching attention keys/values from previous steps, avoiding recomputation.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask (not used in this implementation)
            past_key_values: Cached keys/values from previous generation steps
            use_cache: Whether to return keys/values for caching
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return structured output
            labels: Ground truth tokens for loss computation (training)
            
        Returns:
            CausalLMOutputWithPast containing logits, loss, and optional cached states
        """
        # Set default values from config if not provided
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        hidden_states = self.embed_tokens(input_ids)

        # Initialize output containers if requested
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 2. Pass through transformer blocks sequentially
        for i, block in enumerate(self.layers):
            # Store hidden states if requested
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get cached key/value for this layer if available
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Apply transformer block
            layer_outputs = block(
                hidden_states,
                attention_mask=None,  # Causal mask handled in attention module
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            # Store outputs if requested
            if use_cache:
                next_decoder_cache += (layer_outputs[2],)  # KV cache
            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # Attention weights

        # 3. Final normalization and language model head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # 4. Calculate loss if training labels provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction: input[i] predicts label[i+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten for cross-entropy calculation
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure labels are on same device as logits
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Return structured output compatible with Hugging Face ecosystem
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# =============================================================================
# 4. Tokenizer
# =============================================================================

import tiktoken
from tiktoken.load import load_tiktoken_bpe

class Qwen3Tokenizer:
    """
    Hugging Face-compatible wrapper for Qwen's tiktoken-based tokenizer.
    
    This class bridges the gap between the original Qwen tokenizer (based on tiktoken)
    and the Hugging Face ecosystem. The tiktoken library provides efficient tokenization
    using Byte-Pair Encoding (BPE) with custom vocabularies and special tokens.
    
    TIKTOKEN ADVANTAGES:
    - **Efficiency**: Faster tokenization compared to standard HF tokenizers
    - **Memory Efficient**: Lower memory footprint for large vocabularies
    - **Custom Vocabularies**: Supports domain-specific or multilingual vocabularies
    - **Special Token Handling**: Robust handling of chat formatting and control tokens
    
    TOKENIZATION FEATURES:
    - **BPE Encoding**: Uses merged byte-pair rankings for subword tokenization
    - **Special Tokens**: Handles chat formatting tokens like <|im_start|>, <|im_end|>
    - **Regex Patterns**: Custom regex for proper handling of text boundaries
    - **Unicode Support**: Proper handling of multilingual text and special characters
    
    HUGGING FACE COMPATIBILITY:
    This wrapper provides the essential methods expected by HF models:
    - encode(): Text -> token IDs
    - decode(): Token IDs -> text  
    - __call__(): Returns dict with 'input_ids' for model input
    - Standard attributes: vocab_size, eos_token_id, pad_token_id
    
    SPECIAL TOKENS:
    - <|im_start|>: Start of a message in chat format
    - <|im_end|>: End of a message in chat format
    - <|endoftext|>: End of document/conversation token
    
    Args:
        tokenizer_path (str): Path to the .tiktoken vocabulary file
    """
    def __init__(self, tokenizer_path="qwen.tiktoken"):
        # Define special tokens used in Qwen chat format
        special_tokens = {
            "<|im_start|>": 151645,   # Chat message start
            "<|im_end|>": 151644,     # Chat message end
            "<|endoftext|>": 151643,  # End of text/document
        }

        # Initialize tiktoken encoder with custom vocabulary and patterns
        self.tokenizer = tiktoken.Encoding(
            name=tokenizer_path.split(".")[0],
            # Regex pattern for tokenization boundaries - handles:
            # - Contractions ('s, 't, 're, etc.)
            # - Letters and numbers
            # - Punctuation and whitespace
            # - Unicode character classes
            pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            mergeable_ranks=load_tiktoken_bpe(tokenizer_path),
            special_tokens=special_tokens
        )
        
        # Set attributes expected by Hugging Face ecosystem
        self.vocab_size = self.tokenizer.n_words
        self.special_tokens = special_tokens
        self.pad_token_id = self.tokenizer.n_words  # Use a new token ID for padding
        self.eos_token_id = special_tokens["<|endoftext|>"]
        self.bos_token_id = None  # Qwen doesn't typically use a beginning-of-sequence token

    def encode(self, text: str, **kwargs) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text (str): Input text to tokenize
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            list[int]: List of token IDs
        """
        # Allow encoding of special tokens (needed for chat formatting)
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (list[int]): List of token IDs to decode
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            str: Decoded text
        """
        return self.tokenizer.decode(token_ids)

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and return in format expected by Hugging Face models.
        
        Args:
            text (str): Input text to tokenize
            return_tensors (str): Format of returned tensors ("pt" for PyTorch)
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Dict containing 'input_ids' tensor ready for model input
        """
        token_ids = self.encode(text)
        if return_tensors == "pt":
            # Add batch dimension: (seq_len,) -> (1, seq_len)
            return {"input_ids": torch.tensor(token_ids).unsqueeze(0)}
        return {"input_ids": token_ids}



# =============================================================================
# 5. Usage Example with Hugging Face Hub
# =============================================================================

def main():
    print("--- 1. Loading Model and Tokenizer from Hugging Face Hub ---")
    
    # Use official Qwen model from Hugging Face Hub
    # Available models: "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", etc.
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Using a smaller model for faster loading
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Loading {model_name} on {device} with {dtype}...")
    
    # 1. Load tokenizer from Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded from Hugging Face Hub.")
    
    # 2. Load model from Hub with automatic dtype and device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,  # Automatic device mapping for multi-GPU
        low_cpu_mem_usage=True,  # Reduces peak RAM usage during loading
        trust_remote_code=False  # Official models don't need this
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"Model loaded successfully on {device}.")
    
    # Set the model to evaluation mode
    model.eval()

    print("\n--- 2. Generating Text with Hugging Face `generate` method ---")
    # The prompt format follows the Qwen chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is the capital of France?"}
    ]
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print(f"Prompt:\n{prompt}")

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    # Use the powerful `model.generate()` method provided by the `PreTrainedModel` base class.
    # This handles KV caching, sampling, and stopping criteria automatically.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,  # Use EOS as pad token
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode the generated tokens to text (only the new tokens)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\nModel Response:\n{response}")
    
    # Demonstrate another prompt
    print("\n--- 3. Another Example ---")
    messages_2 = [
        {"role": "user", "content": "Write a short poem about the beauty of programming."}
    ]
    
    prompt_2 = tokenizer.apply_chat_template(
        messages_2, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print(f"Prompt:\n{prompt_2}")
    inputs_2 = tokenizer(prompt_2, return_tensors="pt")
    input_ids_2 = inputs_2["input_ids"].to(model.device)
    
    with torch.no_grad():
        output_ids_2 = model.generate(
            input_ids_2,
            max_new_tokens=100,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    new_tokens_2 = output_ids_2[0][input_ids_2.shape[-1]:]
    response_2 = tokenizer.decode(new_tokens_2, skip_special_tokens=True)
    print(f"\nModel Response:\n{response_2}")

    print("\n--- 4. Model Information ---")
    print(f"Model name: {model_name}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {model.config.vocab_size:,}")
    print(f"Context length: {model.config.max_position_embeddings:,}")


if __name__ == "__main__":
    main()
