from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


class TextProcessor:
    """HFTokenizer."""

    def __init__(self, tokenizer_name: str = "t5-base", tokenizer_kwargs: Optional[Dict] = None):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            }

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_kwargs = tokenizer_kwargs

    def encode(self, strings: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode strings to token IDs and attention masks

        Args:
            strings: List of strings to encode

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        return self.tokenizer(strings, **self.tokenizer_kwargs)
