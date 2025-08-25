import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from octo_pytorch.model.components.base import TokenGroup
from octo_pytorch.model.components.diffusion import ScoreActor


def masked_mean(x, mask):
    mask = torch.broadcast_to(mask, x.shape)
    return torch.mean(x * mask) / torch.clamp(torch.mean(mask.to(x.dtype)), min=1e-5, max=None)


def continous_loss(pred_value, ground_truth_value, mask, loss_type="mse"):
    if loss_type == "mse":
        loss = torch.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = torch.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = torch.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {"loss": loss, "mse": mse}


class DiffusionActionHead(nn.Module):
    """Implementation of diffusion action head."""

    def __init__(
        self,
        readout_key: str,
        use_map: bool = False,
        input_dim: int = 768,  # T5 d_model
        action_horizon: int = 1,
        action_dim: int = 7,
        max_action: float = 5.0,
        loss_type: str = "mse",
        time_dim: int = 32,
        num_blocks: int = 3,
        dropout_rate: float = 0.0,
        hidden_dim: int = 256,
        use_layer_norm: bool = True,
        diffusion_steps: int = 20,
        n_diffusion_samples: int = 1,
    ):
        super().__init__()

        self.readout_key = readout_key
        self.use_map = use_map
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_action = max_action
        self.loss_type = loss_type
        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.diffusion_steps = diffusion_steps
        self.n_diffusion_samples = n_diffusion_samples

        # Initialize MAP head if needed (skipped since use_map=False)
        if self.use_map:
            raise NotImplementedError("MAP head not implemented as use_map=False")

        # Diffusion network
        input_size = input_dim + action_dim * action_horizon + time_dim  # obs_enc + actions + cond_enc
        self.diffusion_model = ScoreActor(
            action_dim * action_horizon,
            in_dim=input_size,
            time_dim=time_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )

        # Create beta schedule
        self.register_buffer("betas", self._cosine_beta_schedule(diffusion_steps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_hats", torch.cumprod(self.alphas, dim=0))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def forward(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass through the diffusion model."""

        # Extract token group using readout_key
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            "Expected token_group.tokens to have shape "
            "(batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        # Process embeddings
        if self.use_map:  # Multi-head attention pooling
            raise NotImplementedError("MAP head not implemented as use_map=False")
        else:  # mean pooling
            embeddings = token_group.tokens.mean(dim=-2)

        # Run diffusion model
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time)
        return pred_eps

    def loss(self, transformer_outputs, actions, timestep_pad_mask, action_pad_mask):
        """Compute the loss for the diffusion objective."""
        batch_size, window_size = timestep_pad_mask.shape
        device = actions.device
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = torch.clamp(actions_flat, -self.max_action, self.max_action)

        time = torch.randint(
            0, self.diffusion_steps, (self.n_diffusion_samples, batch_size, window_size, 1), device=device
        )
        noise = torch.randn((self.n_diffusion_samples,) + actions_flat.shape, device=device)

        alpha_hats = self.alpha_hats.to(device=device)

        scale = torch.sqrt(alpha_hats[time])
        std = torch.sqrt(1 - alpha_hats[time])
        noisy_actions = scale * actions_flat[None] + std * noise
        pred_eps = self.forward(transformer_outputs, time=time, noisy_actions=noisy_actions)

        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask
        mask = rearrange(mask, "b w h a -> b w (h a)")
        mask = mask[None]

        loss, metrics = continous_loss(pred_eps, noise, mask, loss_type=self.loss_type)

        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        embodiment_action_dim: Optional[int] = None,
        sample_shape: tuple = (),
    ) -> torch.Tensor:
        """Convenience method for predicting actions for the final timestep."""

        if embodiment_action_dim is None:
            print(
                "embodiment_action_dim is highly recommended for diffusion action head"
                " if any action dimensions were masked during training"
            )

        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        device = transformer_outputs[self.readout_key].tokens.device

        # Create action mask
        action_mask = torch.ones(
            (*sample_shape, batch_size, window_size, self.action_horizon, self.action_dim),
            dtype=torch.bool,
            device=device,
        )
        if embodiment_action_dim is not None:
            action_mask = action_mask.clone()
            action_mask[..., embodiment_action_dim:] = False

        flat_action_mask = action_mask.view(
            *sample_shape, batch_size, window_size, self.action_horizon * self.action_dim
        )

        # Initialize with noise
        noise = torch.randn(
            (*sample_shape, batch_size, window_size, self.action_horizon * self.action_dim),
            device=device,
        )

        # DDPM sampling loop
        # Run reverse diffusion
        current_x = noise
        for time in reversed(range(self.diffusion_steps)):
            input_time = torch.full((*current_x.shape[:-1], 1), time, device=device, dtype=torch.float32)

            eps_pred = self.forward(transformer_outputs, input_time, current_x)

            alpha_1 = 1 / torch.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / torch.sqrt(1 - self.alpha_hats[time])
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            z = torch.randn(current_x.shape, device=device)
            current_x = current_x + (time > 0) * (torch.sqrt(self.betas[time]) * z)

            current_x = torch.clamp(current_x, -self.max_action, self.max_action)

            # Set non-eval actions to the noise that would have been seen during training
            current_x = torch.where(flat_action_mask, current_x, torch.sqrt(1 - self.alpha_hats[time]) * z)

        flat_action = current_x

        # Reshape and return last timestep
        actions = flat_action.view(
            *sample_shape, batch_size, window_size, self.action_horizon, self.action_dim
        )
        # Only get the last timestep in the window
        return actions[..., -1, :, :]
