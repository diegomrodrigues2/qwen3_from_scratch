# Supervised Fine-Tuning (SFT) for Qwen3

This module provides a complete implementation for supervised fine-tuning of Qwen3 models on instruction-following datasets.

## Features

- **Multiple Data Formats**: Support for Alpaca, ShareGPT, and custom formats
- **Flexible Data Sources**: Load from local JSON/JSONL files or Hugging Face datasets
- **Chat Template Support**: Automatic formatting using model's chat template
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **Mixed Precision**: FP16 and BF16 training support
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Full HuggingFace Integration**: Built on top of transformers Trainer
- **Experiment Tracking**: WandB and TensorBoard support

## Installation

Make sure you have all required dependencies:

```bash
pip install -r requirements.txt
```

For LoRA support, install PEFT:

```bash
pip install peft
```

## Quick Start

### 1. Prepare Your Data

Create a dataset in Alpaca format (JSON):

```json
[
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Translate to Spanish",
        "input": "Hello, how are you?",
        "output": "Hola, ¿cómo estás?"
    }
]
```

Or use the ShareGPT conversation format:

```json
[
    {
        "conversations": [
            {"from": "user", "value": "Hello!"},
            {"from": "assistant", "value": "Hi! How can I help you today?"},
            {"from": "user", "value": "Tell me about AI."},
            {"from": "assistant", "value": "AI stands for Artificial Intelligence..."}
        ]
    }
]
```

### 2. Run Training

Basic training:

```bash
python train_sft.py \
    --data_path ./examples/sample_alpaca.json \
    --output_dir ./output/qwen-sft \
    --num_train_epochs 3
```

Training with LoRA (recommended for efficiency):

```bash
python train_sft.py \
    --data_path ./data/alpaca.json \
    --output_dir ./output/qwen-lora \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_train_epochs 3
```

Training with evaluation:

```bash
python train_sft.py \
    --data_path ./data/train.json \
    --eval_data_path ./data/eval.json \
    --output_dir ./output/qwen-sft \
    --num_train_epochs 5 \
    --bf16
```

Training with Hugging Face dataset:

```bash
python train_sft.py \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir ./output/qwen-alpaca \
    --num_train_epochs 3
```

## Usage

### Command-Line Interface

```bash
python train_sft.py --help
```

Key arguments:

- `--data_path`: Path to training data (JSON/JSONL or HuggingFace dataset)
- `--output_dir`: Directory to save model checkpoints
- `--model_name_or_path`: Pretrained model to fine-tune (default: Qwen/Qwen2.5-1.5B)
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--max_length`: Maximum sequence length
- `--use_lora`: Enable LoRA for efficient fine-tuning
- `--bf16`: Use bfloat16 mixed precision
- `--format_type`: Data format (alpaca, sharegpt, custom)

### Python API

```python
from sft import SFTTrainer, SFTConfig

# Create configuration
config = SFTConfig(
    model_name_or_path="Qwen/Qwen2.5-1.5B",
    data_path="./data/alpaca.json",
    output_dir="./output",
    max_length=2048,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    use_lora=True,
    lora_r=8,
)

# Create trainer and run
trainer = SFTTrainer(config)
trainer.train()
```

### Using the Dataset Loader

```python
from sft import SFTDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Load Alpaca format dataset
dataset = SFTDataset(
    data_path="./data/alpaca.json",
    tokenizer=tokenizer,
    max_length=2048,
    format_type="alpaca",
    use_chat_template=True,
)

# Access data
example = dataset[0]
print(example.keys())  # ['input_ids', 'attention_mask', 'labels']
```

## Data Formats

### Alpaca Format

```json
{
    "instruction": "Task description",
    "input": "Optional input (can be empty)",
    "output": "Expected output",
    "system": "Optional system prompt"
}
```

### ShareGPT Format

```json
{
    "conversations": [
        {"from": "user", "value": "User message"},
        {"from": "assistant", "value": "Assistant response"}
    ]
}
```

### Custom Format

The loader can adapt to custom formats with fields like:
- `question` → instruction
- `answer` → output
- `response` → output

## Training Recommendations

### Small Models (1.5B - 3B parameters)

```bash
python train_sft.py \
    --data_path ./data/train.json \
    --output_dir ./output \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --bf16
```

### Large Models (7B+ parameters)

Use LoRA for efficiency:

```bash
python train_sft.py \
    --data_path ./data/train.json \
    --output_dir ./output \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16
```

### Memory-Constrained Environment

```bash
python train_sft.py \
    --data_path ./data/train.json \
    --output_dir ./output \
    --use_lora \
    --lora_r 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --max_length 1024 \
    --bf16
```

## Configuration Options

### SFTConfig Parameters

```python
@dataclass
class SFTConfig:
    # Model and data
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    data_path: str = ""
    eval_data_path: Optional[str] = None
    output_dir: str = "./output"

    # Training
    max_length: int = 2048
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5

    # Optimization
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False

    # LoRA
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
```

## Examples

See `examples/sample_alpaca.json` for a sample dataset.

## Troubleshooting

### Out of Memory

1. Reduce batch size: `--per_device_train_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use LoRA: `--use_lora`
4. Reduce sequence length: `--max_length 1024`
5. Use mixed precision: `--bf16`

### Slow Training

1. Increase batch size if you have memory
2. Use mixed precision (bf16 on modern GPUs)
3. Increase gradient accumulation steps
4. Disable gradient checkpointing if memory allows

### Poor Results

1. Increase training epochs
2. Adjust learning rate (try 1e-5 to 1e-4 range)
3. Check data quality and format
4. Ensure chat template is being used correctly
5. Try different LoRA ranks (8, 16, 32)

## Advanced Usage

### Resume from Checkpoint

```bash
python train_sft.py \
    --data_path ./data/train.json \
    --output_dir ./output \
    --resume_from_checkpoint ./output/checkpoint-1000
```

### Custom WandB Project

```bash
python train_sft.py \
    --data_path ./data/train.json \
    --output_dir ./output \
    --run_name "qwen-sft-experiment-1" \
    --report_to wandb
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 train_sft.py \
    --data_path ./data/train.json \
    --output_dir ./output \
    --per_device_train_batch_size 4
```

## License

This code is part of the Qwen3 from scratch project and follows the same license.
