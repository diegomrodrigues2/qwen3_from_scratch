# Qwen3-like Model From Scratch

This repository contains a full, from-scratch implementation of a Qwen3-like transformer model in PyTorch. The implementation is designed for educational purposes and is fully compatible with the Hugging Face `transformers` library. This project is based on the concepts from Sebastian Raschka's `standalone-qwen3.ipynb` notebook, refactored into a clean, well-documented, and reusable Python script.

## 🚀 Overview

The goal of this project is to provide a didactic guide on how to build a custom transformer model from the ground up and integrate it with the `transformers` library. It covers the implementation of modern architectural components, weight loading, and text generation.

## 🏛️ Architectural Features

This implementation follows a standard decoder-only transformer architecture with several key features found in modern Large Language Models:

-   **RMSNorm**: Root Mean Square Layer Normalization is used for its efficiency and performance compared to standard LayerNorm.
-   **Rotary Positional Embeddings (RoPE)**: Applied to queries and keys to encode positional information, allowing for better length extrapolation.
-   **Multi-Head Attention (MHA)**: For simplicity and educational clarity, standard Multi-Head Attention is used. The architecture is compatible with Grouped-Query Attention (GQA).
-   **SwiGLU Activation**: The feed-forward network uses the Swish-Gated Linear Unit activation function, common in high-performance LLMs.
-   **Hugging Face Integration**: The model is built as a `PreTrainedModel`, providing seamless integration with the Hugging Face ecosystem, including methods like `.generate()`, `.from_pretrained()`, and `.push_to_hub()`.
-   **Custom Tokenizer Wrapper**: A wrapper for `tiktoken` is included to make Qwen's efficient tokenizer compatible with Hugging Face workflows.

## 📂 Project Structure

The entire implementation is self-contained in `model.py`, which includes:

-   `Qwen3Config`: A `PretrainedConfig` class for managing model hyperparameters.
-   `Qwen3ForCausalLM`: The main model class inheriting from `PreTrainedModel`.
-   **From-Scratch Modules**: `RMSNorm`, `RotaryEmbedding`, `Qwen3Attention`, `Qwen3MLP`, and `Qwen3Block`.
-   `Qwen3Tokenizer`: A wrapper for the `tiktoken` tokenizer.
-   A `main()` function that demonstrates how to load and use an official Qwen model from the Hugging Face Hub.

## 💻 Usage

The `main()` function in `model.py` provides a complete example of how to load an official Qwen model from the Hugging Face Hub and use it for text generation.

### Using Hugging Face Hub (Recommended)

#### 1. Installation

Install the required libraries:

```bash
pip install torch transformers huggingface-hub tiktoken
```

#### 2. Running the script

To run the demonstration:

```bash
python model.py
```

The script will:
1.  Load the `Qwen/Qwen2.5-1.5B-Instruct` model and tokenizer from the Hugging Face Hub.
2.  Set up the model for inference on the appropriate device (CUDA or CPU).
3.  Generate text in response to a sample prompt using the `model.generate()` method.

### Using Local Weights (From Scratch)

This repository also allows you to load original Qwen `.pth` weights and run the from-scratch model directly. A utility function is provided in `util.py` to handle the weight conversion.

**1. Download Original Weights**

First, you need to download the original model weights (`.pth` file) and the `qwen.tiktoken` tokenizer file.

**2. Create a script to run the model**

Create a Python script (e.g., `run_local.py`) and add the following code. This script will initialize the from-scratch model, load the tokenizer, and use the `convert_and_load_weights` utility to load your local weights.

```python
import torch
import os
from model import Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer
from util import convert_and_load_weights

def run_local():
    print("--- 1. Initializing Model and Tokenizer from local files ---")

    # Define file paths. Change these if your files are located elsewhere.
    ORIGINAL_WEIGHTS_PATH = "qwen_3b_instruct.pth"  # Change to your .pth file
    TOKENIZER_PATH = "qwen.tiktoken"

    # Check if necessary files exist
    if not os.path.exists(ORIGINAL_WEIGHTS_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("\\nERROR: Model weights or tokenizer file not found.")
        print(f"Please make sure '{ORIGINAL_WEIGHTS_PATH}' and '{TOKENIZER_PATH}' are in the current directory.")
        return

    # Create a configuration for our model.
    # These parameters should match the model weights you are using.
    config = Qwen3Config(
        vocab_size=151936,
        context_len=4096,
        n_layer=32,
        n_head=32,
        n_embd=3456,
        intermediate_size=18944
    )

    # Initialize the tokenizer
    tokenizer = Qwen3Tokenizer(tokenizer_path=TOKENIZER_PATH)

    # Initialize the Hugging Face model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = Qwen3ForCausalLM(config).to(device).to(dtype)

    print("\\n--- 2. Loading and Converting Pretrained Weights ---")
    convert_and_load_weights(model, ORIGINAL_WEIGHTS_PATH)
    model.eval()

    print("\\n--- 3. Generating Text ---")
    prompt = (
        "<|im_start|>system\\n"
        "You are a helpful assistant.<|im_end|>\\n"
        "<|im_start|>user\\n"
        "Hello, what is the capital of France?<|im_end|>\\n"
        "<|im_start|>assistant\\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0].tolist())
    print(f"\\nModel Response:\\n{response}")

if __name__ == "__main__":
    run_local()
```

**3. Run the Local Script**

Execute the script from your terminal:
```bash
python run_local.py
```

## 🎓 Educational Goals

This repository is intended as a learning resource for those looking to:
-   Understand the architecture of modern decoder-only transformers.
-   Implement key components like RoPE, SwiGLU, and RMSNorm from scratch.
-   Learn how to integrate a custom model into the Hugging Face ecosystem.
-   See an example of weight conversion and model loading for custom architectures.
-   Understand the difference between using a pre-packaged Hub model and loading local weights. 
## Development

Run pre-commit to format and lint code:
```bash
pre-commit run --all-files
```

Execute the test suite with:
```bash
pytest -q
```
