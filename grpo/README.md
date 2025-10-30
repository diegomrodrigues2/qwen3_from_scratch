# GRPO (Group Relative Policy Optimization)

This module implements GRPO, an advanced reinforcement learning from human feedback (RLHF) technique for fine-tuning language models.

## What is GRPO?

GRPO (Group Relative Policy Optimization) is a policy gradient method that improves upon traditional PPO by:

1. **Group-Based Learning**: Generates multiple responses (K) for each prompt and learns from their relative quality
2. **Variance Reduction**: Uses within-group comparisons instead of global baselines, reducing gradient variance
3. **Simplicity**: Doesn't require a separate value network, making it simpler than PPO
4. **Efficiency**: Achieves better sample efficiency through group-relative advantage estimation

### Algorithm Overview

For each training step:
1. Sample a batch of prompts
2. Generate K responses per prompt using the current policy
3. Score all responses with a reward model
4. Compute advantages relative to each group's mean reward
5. Update policy to maximize expected relative reward with KL penalty

## Installation

Make sure you have the required dependencies:

```bash
pip install torch transformers datasets wandb tqdm
```

## Quick Start

### Basic Usage

```python
from grpo import GRPOTrainer, GRPOConfig, SimplePromptDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create dataset
prompts = ["Your prompt 1", "Your prompt 2", "..."]
dataset = SimplePromptDataset(prompts)

# Configure training
config = GRPOConfig(
    batch_size=4,
    num_epochs=3,
    learning_rate=1e-6,
    num_generations_per_prompt=4,
    output_dir="./grpo_output"
)

# Train
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=model,  # Use proper reward model in production
    tokenizer=tokenizer,
    config=config,
    train_dataset=dataset
)

trainer.train()
```

### Using the Training Script

The easiest way to get started is using the provided training script:

```bash
# Basic training with defaults
python grpo/train_grpo.py

# Custom configuration
python grpo/train_grpo.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 1e-6 \
    --num-generations 4 \
    --output-dir ./my_grpo_model

# With Weights & Biases logging
python grpo/train_grpo.py \
    --use-wandb \
    --wandb-project my-project \
    --wandb-run-name grpo-experiment-1
```

## Configuration Parameters

### Generation Parameters

- `num_generations_per_prompt` (int, default=4): Number of responses to generate per prompt (K)
- `max_new_tokens` (int, default=256): Maximum tokens to generate per response
- `temperature` (float, default=0.7): Sampling temperature for generation
- `top_k` (int, default=50): Top-k sampling parameter
- `top_p` (float, default=0.9): Nucleus sampling parameter

### Training Parameters

- `learning_rate` (float, default=1e-6): Learning rate for optimizer
- `batch_size` (int, default=4): Number of prompts per batch
- `num_epochs` (int, default=3): Number of training epochs
- `gradient_accumulation_steps` (int, default=4): Steps to accumulate gradients
- `max_grad_norm` (float, default=1.0): Maximum gradient norm for clipping
- `warmup_steps` (int, default=100): Warmup steps for learning rate scheduler

### GRPO-Specific Parameters

- `kl_coef` (float, default=0.1): KL divergence penalty coefficient (β)
- `clip_range` (float, default=0.2): PPO-style clipping range (ε)
- `gamma` (float, default=1.0): Discount factor for rewards

### Logging Parameters

- `logging_steps` (int, default=10): Log metrics every N steps
- `save_steps` (int, default=500): Save checkpoint every N steps
- `output_dir` (str, default="./grpo_checkpoints"): Output directory
- `use_wandb` (bool, default=False): Enable Weights & Biases logging
- `wandb_project` (str): W&B project name
- `wandb_run_name` (str): W&B run name

## Command Line Options

```
--model-name              HuggingFace model name or path
--batch-size              Training batch size
--num-epochs              Number of training epochs
--learning-rate           Learning rate
--num-generations         Number of responses per prompt
--max-new-tokens          Maximum tokens to generate
--kl-coef                 KL divergence coefficient
--clip-range              PPO clipping range
--temperature             Sampling temperature
--gradient-accumulation-steps  Gradient accumulation steps
--max-grad-norm           Maximum gradient norm
--warmup-steps            Number of warmup steps
--output-dir              Output directory
--logging-steps           Log every N steps
--save-steps              Save every N steps
--use-wandb               Enable W&B logging
--wandb-project           W&B project name
--wandb-run-name          W&B run name
--device                  Device (cuda/cpu)
```

## Using Custom Datasets

Replace the example dataset with your own:

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("your-dataset-name")
prompts = dataset["train"]["prompt"]

# Create GRPO dataset
grpo_dataset = SimplePromptDataset(prompts)
```

## Using a Custom Reward Model

For production training, use a proper reward model:

```python
from transformers import AutoModelForSequenceClassification

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "your-reward-model-name",
    num_labels=1  # For scalar rewards
)

# Use in trainer
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,  # Proper reward model
    tokenizer=tokenizer,
    config=config,
    train_dataset=dataset
)
```

## Important Notes

### Reward Model

The example script uses the policy model itself as a reward model for demonstration. **In production, you must use a properly trained reward model** that has been fine-tuned on human preference data.

### Memory Requirements

GRPO is memory-intensive because it:
- Loads 3 models (policy, reference, reward)
- Generates multiple responses per prompt
- Computes gradients for all generated sequences

Tips for reducing memory usage:
- Use smaller batch sizes
- Reduce `num_generations_per_prompt`
- Use gradient checkpointing
- Use smaller models
- Use mixed precision (bfloat16/float16)

### Computational Cost

Each training step involves:
- K generations per prompt (expensive)
- Reward model inference (K times per prompt)
- Policy model forward passes for all sequences

For faster iteration during development:
- Use fewer generations per prompt (2-3)
- Reduce max_new_tokens
- Use smaller models
- Test on a subset of data first

## Output

After training, the model and tokenizer are saved to the output directory:

```
output_dir/
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── training_state.pt
└── checkpoint-{step}/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── training_state.pt
```

Load the trained model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./grpo_checkpoints/final_model")
tokenizer = AutoTokenizer.from_pretrained("./grpo_checkpoints/final_model")

# Generate with the fine-tuned model
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## References

GRPO is based on recent advances in RLHF and policy optimization:

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **RLHF**: Christiano et al., "Deep Reinforcement Learning from Human Preferences" (2017)
- **InstructGPT**: Ouyang et al., "Training language models to follow instructions with human feedback" (2022)

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size: `--batch-size 1`
- Reduce generations: `--num-generations 2`
- Reduce max tokens: `--max-new-tokens 64`
- Use gradient accumulation: `--gradient-accumulation-steps 8`

### Slow Training

- Use smaller model initially
- Reduce number of generations per prompt
- Test on subset of data first
- Ensure CUDA is being used: check `--device cuda`

### Poor Results

- Check your reward model is properly trained
- Increase training epochs
- Adjust KL coefficient (too high = stays close to reference, too low = diverges)
- Verify prompts are high quality and diverse
- Check learning rate (try 5e-7 to 5e-6)

## License

This implementation is provided for educational and research purposes.
