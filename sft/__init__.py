"""
Supervised Fine-Tuning (SFT) Package

This package provides utilities for fine-tuning Qwen3 models on instruction-following datasets.

Main components:
- SFTDataset: Dataset loader for various instruction formats
- ConversationDataset: Dataset loader for multi-turn conversations
- SFTTrainer: Trainer class for supervised fine-tuning
- SFTConfig: Configuration for training
"""

from sft.dataset import SFTDataset, ConversationDataset, SFTExample
from sft.train import SFTTrainer, SFTConfig

__all__ = [
    "SFTDataset",
    "ConversationDataset",
    "SFTExample",
    "SFTTrainer",
    "SFTConfig",
]
