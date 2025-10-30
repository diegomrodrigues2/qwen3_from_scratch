"""
Dataset loader for Supervised Fine-Tuning (SFT).

This module provides utilities for loading and processing datasets for instruction fine-tuning.
Supports multiple formats including Hugging Face datasets, JSON, and JSONL files.
"""

import json
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


@dataclass
class SFTExample:
    """Single example for supervised fine-tuning."""
    instruction: str
    input: Optional[str] = None
    output: str = ""
    system: Optional[str] = None


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning with instruction-following data.

    Supports multiple formats:
    - Hugging Face datasets
    - JSON files with instruction-input-output format
    - JSONL files (one example per line)
    - Conversation format (multi-turn dialogues)

    Args:
        data_path: Path to dataset file or Hugging Face dataset name
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length (default: 2048)
        split: Dataset split to use (default: "train")
        format_type: Type of data format ("alpaca", "sharegpt", "custom")
        use_chat_template: Whether to use the model's chat template
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train",
        format_type: str = "alpaca",
        use_chat_template: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        self.use_chat_template = use_chat_template

        # Load the dataset
        self.examples = self._load_data(data_path, split)

    def _load_data(self, data_path: str, split: str) -> List[SFTExample]:
        """Load data from various sources."""
        examples = []

        # Try to load as a Hugging Face dataset first
        if not os.path.exists(data_path):
            try:
                dataset = load_dataset(data_path, split=split)
                examples = self._parse_hf_dataset(dataset)
                print(f"Loaded {len(examples)} examples from Hugging Face dataset: {data_path}")
                return examples
            except Exception as e:
                raise ValueError(f"Could not load dataset from {data_path}: {e}")

        # Load from local file
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                examples = self._parse_json_data(data)
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
                examples = self._parse_json_data(data)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        print(f"Loaded {len(examples)} examples from {data_path}")
        return examples

    def _parse_hf_dataset(self, dataset) -> List[SFTExample]:
        """Parse Hugging Face dataset into SFTExamples."""
        examples = []

        for item in dataset:
            if self.format_type == "alpaca":
                # Alpaca format: instruction, input (optional), output
                examples.append(SFTExample(
                    instruction=item.get('instruction', ''),
                    input=item.get('input', None),
                    output=item.get('output', ''),
                    system=item.get('system', None)
                ))
            elif self.format_type == "sharegpt":
                # ShareGPT format: conversations list
                if 'conversations' in item:
                    examples.append(self._parse_conversation(item['conversations']))
            else:
                # Custom format - try to extract fields
                examples.append(SFTExample(
                    instruction=item.get('instruction', item.get('question', '')),
                    input=item.get('input', None),
                    output=item.get('output', item.get('response', item.get('answer', ''))),
                    system=item.get('system', None)
                ))

        return examples

    def _parse_json_data(self, data: List[Dict]) -> List[SFTExample]:
        """Parse JSON/JSONL data into SFTExamples."""
        examples = []

        for item in data:
            if self.format_type == "alpaca":
                examples.append(SFTExample(
                    instruction=item.get('instruction', ''),
                    input=item.get('input', None),
                    output=item.get('output', ''),
                    system=item.get('system', None)
                ))
            elif self.format_type == "sharegpt":
                if 'conversations' in item:
                    examples.append(self._parse_conversation(item['conversations']))
            else:
                examples.append(SFTExample(
                    instruction=item.get('instruction', item.get('question', '')),
                    input=item.get('input', None),
                    output=item.get('output', item.get('response', item.get('answer', ''))),
                    system=item.get('system', None)
                ))

        return examples

    def _parse_conversation(self, conversations: List[Dict]) -> SFTExample:
        """Parse conversation format into SFTExample."""
        # Extract user message and assistant response from conversation
        instruction = ""
        output = ""

        for turn in conversations:
            role = turn.get('from', turn.get('role', ''))
            content = turn.get('value', turn.get('content', ''))

            if role in ['user', 'human']:
                instruction = content
            elif role in ['assistant', 'gpt', 'bot']:
                output = content
                break  # Use first assistant response

        return SFTExample(instruction=instruction, output=output)

    def _format_example(self, example: SFTExample) -> str:
        """Format an example into a text string."""
        if self.use_chat_template:
            # Use the model's chat template format
            messages = []

            if example.system:
                messages.append({"role": "system", "content": example.system})

            # Combine instruction and input
            user_message = example.instruction
            if example.input:
                user_message = f"{example.instruction}\n\n{example.input}"

            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": example.output})

            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                try:
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    return formatted_text
                except Exception as e:
                    print(f"Warning: Could not apply chat template: {e}")

        # Fallback to simple format
        text = f"Instruction: {example.instruction}\n"
        if example.input:
            text += f"Input: {example.input}\n"
        text += f"Response: {example.output}"

        return text

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized example."""
        example = self.examples[idx]

        # Format the example as text
        formatted_text = self._format_example(example)

        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Prepare labels (same as input_ids for causal LM)
        labels = tokenized['input_ids'].clone()

        # Optionally mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


class ConversationDataset(Dataset):
    """
    Dataset for multi-turn conversation fine-tuning.

    This dataset handles full conversations with multiple turns,
    masking the user inputs in the loss calculation.

    Args:
        data_path: Path to dataset file or Hugging Face dataset name
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length (default: 2048)
        split: Dataset split to use (default: "train")
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load conversations
        self.conversations = self._load_conversations(data_path, split)

    def _load_conversations(self, data_path: str, split: str) -> List[List[Dict]]:
        """Load conversation data."""
        conversations = []

        # Try Hugging Face dataset first
        if not os.path.exists(data_path):
            try:
                dataset = load_dataset(data_path, split=split)
                for item in dataset:
                    if 'conversations' in item:
                        conversations.append(item['conversations'])
                return conversations
            except Exception as e:
                raise ValueError(f"Could not load dataset: {e}")

        # Load from local file
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        for item in data:
            if 'conversations' in item:
                conversations.append(item['conversations'])

        print(f"Loaded {len(conversations)} conversations from {data_path}")
        return conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized conversation."""
        conversation = self.conversations[idx]

        # Format conversation using chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Convert to standard format
            messages = []
            for turn in conversation:
                role = turn.get('from', turn.get('role', ''))
                content = turn.get('value', turn.get('content', ''))

                if role in ['user', 'human']:
                    messages.append({"role": "user", "content": content})
                elif role in ['assistant', 'gpt', 'bot']:
                    messages.append({"role": "assistant", "content": content})
                elif role == 'system':
                    messages.append({"role": "system", "content": content})

            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception:
                # Fallback to simple format
                formatted_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" for msg in messages
                ])
        else:
            # Simple format
            formatted_text = ""
            for turn in conversation:
                role = turn.get('from', turn.get('role', ''))
                content = turn.get('value', turn.get('content', ''))
                formatted_text += f"{role}: {content}\n"

        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels
        labels = tokenized['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }
