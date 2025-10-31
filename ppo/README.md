# PPO (Proximal Policy Optimization) for Qwen3 RLHF

This module implements **Proximal Policy Optimization (PPO)** for fine-tuning Qwen3 models using **Reinforcement Learning from Human Feedback (RLHF)**. PPO is the key algorithm used in training models like ChatGPT to align with human preferences.

## 🎯 Overview

PPO is a policy gradient method that enables stable and efficient training of language models with reward signals. The algorithm constrains policy updates to prevent destructively large changes, making it well-suited for fine-tuning large language models.

### Key Components

1. **Reward Model** (`reward_model.py`): Predicts scalar rewards for generated text based on learned human preferences
2. **PPO Trainer** (`trainer.py`): Implements the core PPO algorithm with clipped surrogate objective
3. **Training Script** (`train.py`): Complete training pipeline with command-line interface

## 🏗️ Architecture

### PPO Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      PPO Training Loop                       │
└─────────────────────────────────────────────────────────────┘

1. Generate Responses
   Prompt → Policy Model (π_θ) → Generated Text

2. Compute Rewards
   Generated Text → Reward Model → Scalar Reward
   + KL Penalty: -β * KL(π_θ || π_ref)

3. Calculate Advantages
   Rewards + Values → GAE → Advantages

4. Update Policy
   Clipped Surrogate Objective → Policy Gradient → Update π_θ

5. Update Value Function
   MSE Loss → Value Gradient → Update V(s)
```

### Reward Model Architecture

```
Input Tokens → Qwen3 Encoder → Last Token Hidden State → Reward Head → Scalar Reward
```

## 📚 Algorithm Details

### PPO Objective

The core PPO loss combines three components:

```
L_total = L_CLIP + c₁·L_VF - c₂·H

Where:
- L_CLIP: Clipped surrogate objective for policy
- L_VF: Value function loss (MSE)
- H: Entropy bonus for exploration
```

**Clipped Surrogate Objective:**

```
L_CLIP = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]

Where:
- r_t = π_θ(a|s) / π_old(a|s)  (probability ratio)
- A_t: advantage estimate
- ε: clip parameter (default: 0.2)
```

### Generalized Advantage Estimation (GAE)

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

Where:
- δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
- γ: discount factor
- λ: GAE parameter
```

## 🚀 Usage

### Basic Usage

```python
from ppo import PPOTrainer, PPOConfig, Qwen3RewardModel
from qwen3.model import Qwen3ForCausalLM
from transformers import AutoTokenizer

# Load models
policy = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
ref_policy = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
reward_model = Qwen3RewardModel.from_pretrained_policy(policy)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Configure PPO
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    ppo_epochs=4,
    clip_range=0.2,
    kl_coef=0.1,
)

# Create trainer
trainer = PPOTrainer(
    config=config,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    tokenizer=tokenizer,
)

# Train on prompts
prompts = [
    "How do I learn Python programming?",
    "Explain quantum computing in simple terms.",
    "What are the benefits of exercise?",
]

trainer.train(prompts, num_epochs=1, max_length=128)
```

### Command-Line Training

```bash
# Basic training
python -m ppo.train --model_name Qwen/Qwen2.5-1.5B

# Custom configuration
python -m ppo.train \
    --model_name Qwen/Qwen2.5-1.5B \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --num_epochs 3 \
    --ppo_epochs 4 \
    --kl_coef 0.1 \
    --output_dir ./ppo_checkpoints

# With custom prompts
python -m ppo.train \
    --model_name Qwen/Qwen2.5-1.5B \
    --prompts_file prompts.txt \
    --output_dir ./my_ppo_model
```

### Training with Custom Reward Model

```bash
# First, train a reward model on preference data
# Then use it for PPO training
python -m ppo.train \
    --model_name Qwen/Qwen2.5-1.5B \
    --reward_model_path ./my_reward_model \
    --learning_rate 1e-6 \
    --batch_size 8
```

## ⚙️ Configuration

### PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-5 | Learning rate for policy optimization |
| `batch_size` | 4 | Number of sequences per batch |
| `ppo_epochs` | 4 | PPO update epochs per batch |
| `gamma` | 1.0 | Discount factor for returns |
| `lam` | 0.95 | Lambda for GAE |
| `clip_range` | 0.2 | PPO clipping parameter (ε) |
| `kl_coef` | 0.1 | Coefficient for KL penalty |
| `vf_coef` | 0.1 | Coefficient for value loss |
| `target_kl` | 0.01 | Target KL divergence |

### Recommended Settings

**Small models (< 1B parameters):**
```python
PPOConfig(
    learning_rate=1e-5,
    batch_size=8,
    ppo_epochs=4,
    kl_coef=0.1,
)
```

**Medium models (1B-7B parameters):**
```python
PPOConfig(
    learning_rate=5e-6,
    batch_size=4,
    ppo_epochs=4,
    kl_coef=0.05,
)
```

**Large models (> 7B parameters):**
```python
PPOConfig(
    learning_rate=1e-6,
    batch_size=2,
    ppo_epochs=2,
    kl_coef=0.02,
)
```

## 📊 Training Tips

### 1. Monitor KL Divergence

Keep KL divergence small (< 0.1) to prevent the policy from deviating too far from the reference:

```python
# Adaptive KL is enabled by default
config = PPOConfig(
    kl_coef=0.1,
    target_kl=0.01,
    adap_kl_ctrl=True,  # Automatically adjust kl_coef
)
```

### 2. Start Small

Begin with a small learning rate and gradually increase if training is stable:

```python
# Conservative settings for initial experiments
config = PPOConfig(
    learning_rate=1e-6,
    batch_size=2,
    ppo_epochs=2,
)
```

### 3. Diverse Prompts

Use diverse prompts to prevent mode collapse and ensure robust training:

```python
prompts = [
    "Explain [topic] in simple terms",
    "Write a helpful response to: [question]",
    "Compare and contrast [A] and [B]",
    "What are the pros and cons of [X]?",
]
```

### 4. Reward Model Quality

The reward model is critical for successful RLHF. In practice:

1. **Train reward model first** on human preference data
2. **Validate reward model** on held-out preferences
3. **Monitor reward distribution** during PPO training

### 5. Save Checkpoints

Save checkpoints frequently to recover from training issues:

```bash
python -m ppo.train \
    --output_dir ./checkpoints \
    --save_steps 100
```

## 🔬 Advanced Features

### Custom Reward Functions

You can implement custom reward functions:

```python
class CustomRewardModel(Qwen3RewardModel):
    def compute_custom_reward(self, text: str) -> float:
        # Implement custom logic
        # E.g., combine multiple reward signals
        base_reward = super().forward(...)
        length_penalty = -0.01 * len(text)
        return base_reward + length_penalty
```

### Multi-Objective Optimization

Combine multiple reward signals:

```python
# Combine helpfulness, harmlessness, and honesty
total_reward = (
    0.5 * helpfulness_reward +
    0.3 * harmlessness_reward +
    0.2 * honesty_reward
)
```

## 📈 Monitoring Training

Track these key metrics:

1. **Policy Loss**: Should decrease over time
2. **Value Loss**: Should stabilize
3. **KL Divergence**: Should stay small (< 0.1)
4. **Mean Reward**: Should increase
5. **Advantage Mean/Std**: Should be stable

## 🐛 Troubleshooting

### Problem: KL divergence exploding

**Solution**: Reduce learning rate or increase `kl_coef`

```python
config = PPOConfig(
    learning_rate=1e-6,  # Reduce from 1e-5
    kl_coef=0.2,         # Increase from 0.1
)
```

### Problem: Rewards not improving

**Solution**: Check reward model, increase learning rate, or adjust advantages

```python
# Verify reward model is working
with torch.no_grad():
    rewards = reward_model(input_ids)
    print(f"Reward range: {rewards.min():.2f} to {rewards.max():.2f}")
```

### Problem: Training unstable

**Solution**: Reduce batch size, clip gradients, use smaller PPO epochs

```python
config = PPOConfig(
    batch_size=2,        # Reduce batch size
    ppo_epochs=2,        # Fewer PPO epochs
    max_grad_norm=0.5,   # Aggressive gradient clipping
)
```

## 📝 Citation

If you use this PPO implementation, please cite:

```bibtex
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
```

## 🔗 References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [Anthropic RLHF](https://arxiv.org/abs/2204.05862)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2309.16609)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Distributed training support
- [ ] More reward model architectures
- [ ] Integration with preference datasets
- [ ] Advanced sampling strategies
- [ ] Reward model training utilities

## 📄 License

This implementation is part of the Qwen3 from scratch project and follows the same license.
