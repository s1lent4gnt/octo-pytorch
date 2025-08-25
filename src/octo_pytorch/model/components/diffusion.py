import math
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Implementation of FourierFeatures with learnable kernel"""

    def __init__(self, output_size: int, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable

        if self.learnable:
            # JAX: (output_size // 2, x.shape[-1]) where x.shape[-1] = 1 for time
            self.kernel = nn.Parameter(torch.normal(0, 0.2, size=(output_size // 2, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_dims..., 1) time values
        Returns:
            fourier_features: (batch_dims..., output_size)
        """
        if self.learnable:
            # f = 2 * π * x @ w.T
            f = 2 * math.pi * torch.matmul(x.to(self.kernel.dtype), self.kernel.T)
        else:
            # Non-learnable version (not used in Octo but included for completeness)
            half_dim = self.output_size // 2
            f = math.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim, device=x.device) * -f)
            f = x * f

        # Concatenate cos and sin
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class MLPResNetBlock(nn.Module):
    """Implementation of MLPResNetBlock."""

    def __init__(
        self, features: int, activation, dropout_rate: Optional[float] = None, use_layer_norm: bool = False
    ):
        super().__init__()
        self.features = features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features, eps=1e-6)

        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)

        return residual + x


class ScoreActor(nn.Module):
    """Implementation of ScoreActor."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        time_dim: int,
        num_blocks: int,
        dropout_rate: float,
        hidden_dim: int,
        use_layer_norm: bool,
    ):
        super().__init__()

        # Time preprocessing (FourierFeatures)
        self.time_preprocess = FourierFeatures(time_dim, learnable=True)

        # Condition encoder (MLP)
        self.cond_encoder = nn.Sequential(
            nn.Linear(time_dim, 2 * time_dim), nn.SiLU(), nn.Linear(2 * time_dim, time_dim)
        )

        # Reverse network (MLPResNet)
        self.reverse_network = MLPResNet(
            num_blocks=num_blocks,
            out_dim=out_dim,
            in_dim=in_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, obs_enc: torch.Tensor, actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Time preprocessing
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff)

        # Broadcast obs_enc if needed
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            obs_enc = obs_enc.expand(new_shape)

        # Concatenate inputs
        reverse_input = torch.cat([cond_enc, obs_enc, actions], dim=-1)

        # Run reverse network
        eps_pred = self.reverse_network(reverse_input)

        return eps_pred


class MLPResNet(nn.Module):
    """Implementation of MLPResNet."""

    def __init__(
        self,
        num_blocks: int,
        out_dim: int,
        in_dim: int,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activation=nn.SiLU,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activation = activation()

        # Initial projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ResNet blocks
        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock(hidden_dim, self.activation, dropout_rate, use_layer_norm)
                for _ in range(num_blocks)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.activation(x)
        x = self.output_proj(x)

        return x
