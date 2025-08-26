import re
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import T5EncoderModel

from octo_pytorch.model.components.base import TokenGroup


def generate_proper_pad_mask(
    tokens: torch.Tensor,
    pad_mask_dict: Optional[Dict[str, torch.Tensor]],
    keys: Sequence[str],
) -> torch.Tensor:
    """Generate proper padding mask for tokens."""
    if pad_mask_dict is None:
        print("No pad_mask_dict found. Nothing will be masked.")
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    if not all([key in pad_mask_dict for key in keys]):
        print(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}. Nothing will be masked."
        )
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], dim=-1)
    pad_mask = torch.any(pad_mask, dim=-1)
    pad_mask = pad_mask.unsqueeze(-1).expand(tokens.shape[:-1])

    return pad_mask


def regex_match(regex_keys, x):
    """Match a string against a list of regex patterns."""
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    """Filter a list of strings using regex patterns."""
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


class ImageTokenizer(nn.Module):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning.

    Args:
        encoder (nn.Module): Encoder class (PatchEncoder, SmallStem, or ViTResnet).
        use_token_learner (bool): Whether to use token learner. Defaults to False.
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for
            encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for
            encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into
            FiLM conditioning. Supports regex.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_tokens: int = 8,
        conditioning_type: str = "none",
        obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*"),
        task_stack_keys: Sequence[str] = tuple(),
        task_film_keys: Sequence[str] = tuple(),
        proper_pad_mask: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys
        self.proper_pad_mask = proper_pad_mask

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        tasks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            observations: Dictionary of tensors with shape (batch_size, timesteps, height, width, channels)
            tasks: Optional dictionary of task tensors

        Returns:
            TokenGroup containing tokens and mask
        """

        def extract_inputs(keys, inputs, check_spatial=False):
            """Extract and concatenate inputs based on keys."""
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])
            return torch.cat(extracted_outputs, dim=-1)

        # Filter observation keys using regex
        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # Stack all spatial observation inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)

        # Stack task inputs if specified
        if self.task_stack_keys and tasks is not None:
            needed_task_keys = regex_filter(self.task_stack_keys, observations.keys())
            # If any task inputs are missing, replace with zero padding
            for k in needed_task_keys:
                if k not in tasks:
                    # Create a copy of tasks with the missing key added
                    if isinstance(tasks, dict):
                        tasks = {**tasks, k: torch.zeros_like(observations[k][:, 0])}
                    else:
                        # Handle case where tasks is not a dict (e.g., None)
                        tasks = {k: torch.zeros_like(observations[k][:, 0])}

            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(f"No task inputs matching {self.task_stack_keys} were found.")

            task_inputs = extract_inputs(task_stack_keys, tasks, check_spatial=True)
            # Repeat task inputs for each timestep
            task_inputs = task_inputs.unsqueeze(1).repeat(1, enc_inputs.shape[1], 1, 1, 1)
            enc_inputs = torch.cat([enc_inputs, task_inputs.to(enc_inputs.device)], dim=-1)

        # Get shape information
        b, t, h, w, c = enc_inputs.shape

        # Reshape for encoder
        enc_inputs = enc_inputs.reshape(b * t, h, w, c)

        # Extract non-spatial FiLM inputs
        encoder_input_kwargs = {}

        # Run visual encoder
        image_tokens = self.encoder(enc_inputs, **encoder_input_kwargs)

        # Reshape back to batch, timestep format
        if isinstance(image_tokens, torch.Tensor):
            # Get spatial dimensions from encoder output
            spatial_dims = image_tokens.shape[1:-1]  # Exclude batch and channel dims
            token_dim = image_tokens.shape[-1]

            # Reshape from (b*t, h', w', c) to (b, t, h'*w', c)
            num_spatial_tokens = np.prod(spatial_dims)
            image_tokens = image_tokens.reshape(b, t, num_spatial_tokens, token_dim)

        # Generate padding mask
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                image_tokens,
                observations.get("pad_mask_dict", None),
                obs_stack_keys,
            )
        else:
            pad_mask = torch.ones(image_tokens.shape[:-1], dtype=torch.bool, device=image_tokens.device)

        # Return TokenGroup
        return TokenGroup(image_tokens, pad_mask)


class LanguageTokenizer(nn.Module):
    """Language tokenizer that embeds text input IDs into continuous language embeddings.
    Supports pre-trained HF models."""

    def __init__(self, finetune_encoder: bool = False, proper_pad_mask: bool = True):
        super().__init__()
        self.proper_pad_mask = proper_pad_mask

        # Load pretrained weights directly
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-base", torch_dtype=torch.float32)
        self.finetune_encoder = finetune_encoder

        if not self.finetune_encoder:
            for param in self.t5_encoder.parameters():
                param.requires_grad = False
            print("T5 encoder frozen (finetune_encoder=False)")
        else:
            print("T5 encoder trainable (finetune_encoder=True)")

    def forward(self, language_input: Dict[str, torch.Tensor], tasks=None) -> TokenGroup:
        # Ensure T5 encoder is on the same device as inputs
        device = language_input["input_ids"].device
        self.t5_encoder = self.t5_encoder.to(device)
        
        # Convert inputs to float32 to avoid precision issues
        outputs = self.t5_encoder(input_ids=language_input["input_ids"].long(), attention_mask=language_input["attention_mask"].long())
        tokens = outputs.last_hidden_state.float()

        # Generate padding mask
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

        # # TODO (lilkm): check this
        # # All true attention mask, simple torch.ones
        # mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)

        # # TODO (lilkm): this more correct
        # # mask = language_input["attention_mask"].bool()

        return TokenGroup(tokens, pad_mask.to(tokens.device))
