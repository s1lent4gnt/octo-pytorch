from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TokenGroup:
    """A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)."""

    tokens: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.mask.ndim != self.tokens.ndim - 1:
            raise ValueError(
                f"Mask must have one less dimension than tokens, "
                f"but got {self.mask.ndim} and {self.tokens.ndim}"
            )

    @classmethod
    def concatenate(cls, group_list: List["TokenGroup"], axis: int = -2) -> "TokenGroup":
        """Concatenates a list of TokenGroups along a specified axis."""
        if not group_list:
            raise ValueError("Cannot concatenate an empty list of TokenGroups")

        tokens = torch.cat([t.tokens for t in group_list], dim=axis)
        mask = torch.cat([t.mask for t in group_list], dim=axis + 1)

        return cls(tokens=tokens, mask=mask)
