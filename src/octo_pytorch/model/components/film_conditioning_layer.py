import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning layer."""

    def __init__(self):
        super().__init__()

    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        """
        Applies FiLM conditioning to a convolutional feature map.

        Args:
            conv_filters: A tensor of shape [batch_size, height, width, channels].
            conditioning: A tensor of shape [batch_size, conditioning_size].

        Returns:
            A tensor of shape [batch_size, height, width, channels].
        """
        channels = conv_filters.shape[-1]
        cond_size = conditioning.shape[-1]

        self.proj_add = nn.Linear(cond_size, channels)
        self.proj_mult = nn.Linear(cond_size, channels)

        projected_cond_add = self.proj_add(conditioning)
        projected_cond_mult = self.proj_mult(conditioning)

        # Reshape for broadcasting
        projected_cond_add = projected_cond_add.unsqueeze(1).unsqueeze(1)
        projected_cond_mult = projected_cond_mult.unsqueeze(1).unsqueeze(1)

        return conv_filters * (1 + projected_cond_mult) + projected_cond_add
