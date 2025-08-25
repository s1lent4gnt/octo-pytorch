from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from octo_pytorch.model.components.base import TokenGroup
from octo_pytorch.model.components.transformer import Transformer


class AttentionRule(Enum):
    """Enum describing when to attend to another token group."""

    NEVER = "never"
    CAUSAL = "other.timestep <= self.timestep"
    CURRENT = "other.timestep == self.timestep"
    STRICT_PAST = "other.timestep < self.timestep"
    ALL = "all"


def find_match(pattern_dict: Dict[str, Any], name: str, default: Any) -> Any:
    """Find the first matching pattern in the dictionary, or return the default value."""
    for pattern, value in pattern_dict.items():
        if fnmatch(name, pattern):
            return value
    return default


@dataclass
class PrefixGroup(TokenGroup):
    """A group of tokens that will be at the beginning of the token sequence (e.g. task tokens)."""

    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        super().__post_init__()
        if len(self.tokens.shape) != 3:
            raise ValueError(
                f"PrefixGroup tokens must be (batch, n_tokens, d), but got shape {self.tokens.shape}"
            )
        if len(self.mask.shape) != 2:
            raise ValueError(f"PrefixGroup mask must be (batch, n_tokens), but got shape {self.mask.shape}")


@dataclass
class TimestepGroup(TokenGroup):
    """A group of tokens that is repeated for each timestep (e.g. observation tokens)."""

    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        super().__post_init__()
        if len(self.tokens.shape) != 4:
            raise ValueError(
                f"TimestepGroup tokens must be (batch, horizon, n_tokens, d), "
                f"but got shape {self.tokens.shape}"
            )
        if len(self.mask.shape) != 3:
            raise ValueError(
                f"TimestepGroup mask must be (batch, horizon, n_tokens), but got shape {self.mask.shape}"
            )


@dataclass
class TokenMetadata:
    """Attention mask logic supported by AttentionRule."""

    name: str
    timestep: int  # -1 for prefix tokens
    attention_rules: Mapping[str, AttentionRule]

    @classmethod
    def create(cls, group: Union["PrefixGroup", "TimestepGroup"], timestep: int):
        return cls(
            timestep=timestep,
            name=group.name,
            attention_rules=group.attention_rules,
        )

    def should_attend_to(self, other_metadata: "TokenMetadata") -> bool:
        attention_rule = find_match(self.attention_rules, other_metadata.name, AttentionRule.NEVER)

        if attention_rule == AttentionRule.CAUSAL:
            return other_metadata.timestep <= self.timestep
        elif attention_rule == AttentionRule.CURRENT:
            return other_metadata.timestep == self.timestep
        elif attention_rule == AttentionRule.STRICT_PAST:
            return other_metadata.timestep < self.timestep
        elif attention_rule == AttentionRule.ALL:
            return True
        elif attention_rule == AttentionRule.NEVER:
            return False
        else:
            raise ValueError(f"Invalid attention rule: {attention_rule}")


class BlockTransformer(nn.Module):
    """A transformer that acts on multiple groups of tokens, which may attend to each other
    (in complex patterns)."""

    def __init__(self, transformer_kwargs: Dict[str, Any], enforce_causal: bool = True):
        super().__init__()
        self.transformer_kwargs = transformer_kwargs
        self.enforce_causal = enforce_causal

        # Create transformer
        self.transformer = Transformer(**transformer_kwargs)

    def forward(
        self, prefix_groups: List[PrefixGroup], timestep_groups: List[TimestepGroup]
    ) -> Tuple[List[PrefixGroup], List[TimestepGroup]]:
        horizon = timestep_groups[0].tokens.shape[1]
        assert all(group.tokens.shape[1] == horizon for group in timestep_groups)

        token_dim = timestep_groups[0].tokens.shape[-1]
        assert all(group.tokens.shape[-1] == token_dim for group in prefix_groups)
        assert all(group.tokens.shape[-1] == token_dim for group in timestep_groups)

        # Assemble input tokens
        input_tokens = self._assemble_input_tokens(prefix_groups, timestep_groups)

        # Generate attention mask
        attention_mask = self._generate_attention_mask(prefix_groups, timestep_groups)

        # Run transformer
        output = self.transformer(input_tokens, attention_mask)

        # Split output into prefix and timestep groups
        prefix_outputs, timestep_outputs = self._split_output_tokens(output, prefix_groups, timestep_groups)

        return prefix_outputs, timestep_outputs

    def _assemble_input_tokens(
        self, prefix_groups: List[PrefixGroup], timestep_groups: List[TimestepGroup]
    ) -> torch.Tensor:
        """Assemble input tokens from prefix and timestep groups."""
        batch_size = timestep_groups[0].tokens.shape[0]
        token_dim = timestep_groups[0].tokens.shape[-1]

        # Concatenate prefix tokens
        if len(prefix_groups) > 0:
            all_prefix_tokens = torch.cat([group.tokens for group in prefix_groups], dim=1)
        else:
            all_prefix_tokens = torch.zeros(
                (batch_size, 0, token_dim),
                dtype=timestep_groups[0].tokens.dtype,
                device=timestep_groups[0].tokens.device,
            )

        # Concatenate timestep tokens and fold horizon into sequence dimension
        all_timestep_tokens = torch.cat([group.tokens for group in timestep_groups], dim=2)
        # Reshape from (batch, horizon, n_tokens, d) to (batch, horizon * n_tokens, d)
        batch_size, horizon, n_tokens, d = all_timestep_tokens.shape
        all_timestep_tokens = all_timestep_tokens.view(batch_size, horizon * n_tokens, d)

        # Concatenate prefix and timestep tokens
        tokens = torch.cat([all_prefix_tokens, all_timestep_tokens], dim=1)

        return tokens

    def _split_output_tokens(
        self,
        output_tokens: torch.Tensor,
        prefix_groups: List[PrefixGroup],
        timestep_groups: List[TimestepGroup],
    ) -> Tuple[List[PrefixGroup], List[TimestepGroup]]:
        """Split output tokens back into prefix and timestep groups."""
        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        n_prefix_tokens = sum(tokens_per_prefix_group)

        # Split prefix and timestep tokens
        prefix_embeddings, timestep_embeddings = torch.split(
            output_tokens, [n_prefix_tokens, output_tokens.shape[1] - n_prefix_tokens], dim=1
        )

        # Process prefix group outputs
        all_prefix_outputs = []
        if len(prefix_groups) > 0:
            prefix_splits = torch.split(prefix_embeddings, tokens_per_prefix_group, dim=1)
            for group, embeddings in zip(prefix_groups, prefix_splits):
                all_prefix_outputs.append(
                    PrefixGroup(
                        tokens=embeddings,
                        mask=group.mask,
                        name=group.name,
                        attention_rules=group.attention_rules,
                    )
                )

        # Process timestep group outputs
        # Reshape from (batch, horizon * n_tokens, d) to (batch, horizon, n_tokens, d)
        batch_size, total_timestep_tokens, d = timestep_embeddings.shape
        n_tokens_per_timestep = total_timestep_tokens // horizon
        timestep_embeddings = timestep_embeddings.view(batch_size, horizon, n_tokens_per_timestep, d)

        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]
        timestep_splits = torch.split(timestep_embeddings, tokens_per_timestep_group, dim=2)

        all_timestep_outputs = []
        for group, embeddings in zip(timestep_groups, timestep_splits):
            all_timestep_outputs.append(
                TimestepGroup(
                    tokens=embeddings, mask=group.mask, name=group.name, attention_rules=group.attention_rules
                )
            )

        return all_prefix_outputs, all_timestep_outputs

    def _generate_attention_mask(
        self, prefix_groups: List[PrefixGroup], timestep_groups: List[TimestepGroup]
    ) -> torch.Tensor:
        """Generate attention mask based on group attention rules."""
        if self.enforce_causal:
            self._verify_causality(prefix_groups, timestep_groups)

        def _get_position(i, tokens_per_elem):
            return np.searchsorted(np.cumsum(tokens_per_elem), i, side="right")

        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]

        tokens_for_prefix = sum(tokens_per_prefix_group)
        tokens_per_time_step = sum(tokens_per_timestep_group)
        total_tokens = tokens_for_prefix + tokens_per_time_step * horizon

        # Create attention mask using numpy for compatibility with JAX implementation
        attention_mask = np.zeros((total_tokens, total_tokens), dtype=int)

        def get_token_metadata(i):
            if i < tokens_for_prefix:
                position = _get_position(i, tokens_per_prefix_group)
                return TokenMetadata.create(prefix_groups[position], timestep=-1)

            i -= tokens_for_prefix
            timestep, i = divmod(i, tokens_per_time_step)
            position = _get_position(i, tokens_per_timestep_group)
            return TokenMetadata.create(timestep_groups[position], timestep)

        # Apply attention rules
        for i in range(total_tokens):  # Token attending
            for j in range(total_tokens):  # Token being attended to
                metadata_i = get_token_metadata(i)
                metadata_j = get_token_metadata(j)
                mask = int(metadata_i.should_attend_to(metadata_j))
                attention_mask[i, j] = mask

        # Convert to torch tensor and move to correct device
        device = timestep_groups[0].tokens.device
        attention_mask = torch.from_numpy(attention_mask).bool().to(device)

        # Combine with padding mask
        pad_attention_mask = self._generate_pad_attention_mask(prefix_groups, timestep_groups)

        # The attention mask from rules is (total_tokens, total_tokens)
        # The padding mask is (batch, total_tokens, total_tokens)
        # We need to combine them properly
        batch_size = pad_attention_mask.shape[0]
        attention_mask = attention_mask.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch, total_tokens, total_tokens)
        # attention_mask = attention_mask.unsqueeze(1)  # (batch, 1, total_tokens, total_tokens)

        # Combine with padding mask using logical AND
        attention_mask = attention_mask & pad_attention_mask

        num_attention_heads = self.transformer_kwargs["num_attention_heads"]

        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, self.transformer_kwargs["num_attention_heads"], total_tokens, total_tokens)
        attention_mask = attention_mask.reshape(batch_size * num_attention_heads, total_tokens, total_tokens)

        return attention_mask

    def _generate_pad_attention_mask(
        self, prefix_groups: List[PrefixGroup], timestep_groups: List[TimestepGroup]
    ) -> torch.Tensor:
        """Generate padding attention mask."""
        batch_size = timestep_groups[0].tokens.shape[0]
        horizon = timestep_groups[0].tokens.shape[1]

        # Concatenate prefix masks
        if len(prefix_groups) > 0:
            prefix_pad_mask = torch.cat([group.mask for group in prefix_groups], dim=1)
        else:
            prefix_pad_mask = torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=timestep_groups[0].tokens.device
            )

        # Concatenate timestep masks and flatten
        timestep_pad_mask = torch.cat([group.mask for group in timestep_groups], dim=2)
        # Reshape from (batch, horizon, n_tokens) to (batch, horizon * n_tokens)
        batch_size, horizon, n_tokens = timestep_pad_mask.shape[:3]
        timestep_pad_mask = timestep_pad_mask.view(batch_size, -1)

        # Combine masks
        pad_mask = torch.cat([prefix_pad_mask, timestep_pad_mask], dim=1)

        # Broadcast to attention mask shape (batch, 1, total_tokens, total_tokens)
        # This matches the JAX implementation's broadcasting
        total_tokens = pad_mask.shape[1]
        pad_mask = pad_mask.unsqueeze(1) # (batch, 1, total_tokens)
        pad_mask = pad_mask.expand(batch_size, total_tokens, total_tokens)

        return pad_mask

    def _verify_causality(self, prefix_groups: List[PrefixGroup], timestep_groups: List[TimestepGroup]):
        """Verify that attention rules don't break causality."""
        # Simplified verification - in full implementation would check all attention rules
        pass
