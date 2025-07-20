import logging
import re
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def regex_match(regex_keys, x):
    """Match a string against a list of regex patterns."""
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    """Filter a list of strings using regex patterns."""
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


def generate_proper_pad_mask(
    tokens: torch.Tensor,
    pad_mask_dict: Optional[Dict[str, torch.Tensor]],
    keys: Sequence[str],
) -> torch.Tensor:
    """Generate proper padding mask for tokens."""
    if pad_mask_dict is None:
        logging.warning("No pad_mask_dict found. Nothing will be masked.")
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}. Nothing will be masked."
        )
        return torch.ones(tokens.shape[:-1], dtype=torch.bool, device=tokens.device)

    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], dim=-1)
    pad_mask = torch.any(pad_mask, dim=-1)
    pad_mask = pad_mask.unsqueeze(-1).expand(tokens.shape[:-1])

    return pad_mask


@torch.no_grad()
def normalize_images(img, img_norm_type="default"):
    """Normalize images according to the specified normalization type."""
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.float() / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.float() / 255.0
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1, 1, 1, 3)

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = mean.repeat(*num_tile)
        std_tile = std.repeat(*num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile
    raise ValueError(f"Unknown img_norm_type: {img_norm_type}")


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


class WeightStandardizedConv2d(nn.Conv2d):
    """Convolution with weight standardization."""

    def forward(self, x):
        weight = self.weight

        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        # NOTE: the use of unbiased estimator
        weight_std = weight.std(dim=(1, 2, 3), keepdim=True, unbiased=False) + 1e-5
        standardized_weight = (weight - weight_mean) / weight_std

        return F.conv2d(x, standardized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SmallStem(nn.Module):
    """Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computer vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """

    def __init__(
        self,
        use_film: bool = False,
        patch_size: int = 32,
        kernel_sizes: Tuple[int, ...] = (3, 3, 3, 3),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        features: Tuple[int, ...] = (32, 96, 192, 384),
        padding: Tuple[int, ...] = (1, 1, 1, 1),
        num_features: int = 512,
        img_norm_type: str = "default",
    ):
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.features = features
        self.padding = padding
        self.num_features = num_features
        self.img_norm_type = img_norm_type

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 6  # Assuming RGB input

        for n, (kernel_size, stride, out_features, conv_padding) in enumerate(
            zip(kernel_sizes, strides, features, padding)
        ):
            self.conv_layers.append(
                nn.Sequential(
                    WeightStandardizedConv2d(
                        in_channels=in_channels,
                        out_channels=out_features,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=conv_padding,
                    ),
                    nn.GroupNorm(32, out_features, eps=1e-06),
                    nn.ReLU(),
                )
            )
            in_channels = out_features

        # Final embedding layer
        final_patch_size = patch_size // 16
        # Use the last element of the features tuple
        last_feature = features[-1] if isinstance(features, tuple) else features
        self.embedding = nn.Conv2d(
            in_channels=last_feature,
            out_channels=num_features,
            kernel_size=(final_patch_size, final_patch_size),
            stride=(final_patch_size, final_patch_size),
            padding=0,
        )

        # FiLM conditioning layer
        self.film = FilmConditioning() if use_film else None

    def forward(
        self, observations: torch.Tensor, train: bool = True, cond_var: Optional[torch.Tensor] = None
    ):
        """
        Args:
            observations: Tensor of shape [batch_size, height, width, channels]
            train: Whether in training mode
            cond_var: Optional conditioning variable for FiLM

        Returns:
            Tensor of shape [batch_size, n_patches_h, n_patches_w, num_features]
        """
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert expecting_cond_var == received_cond_var, "Only pass in cond var iff model expecting cond var"

        # Normalize images
        x = normalize_images(observations, self.img_norm_type)

        # Convert from NHWC to NCHW format for PyTorch
        x = x.permute(0, 3, 1, 2)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Apply final embedding
        x = self.embedding(x)

        # Convert back to NHWC format
        x = x.permute(0, 2, 3, 1)

        # Apply FiLM conditioning if needed
        if self.use_film:
            assert cond_var is not None, "Cond var is None, nothing to condition on"
            x = self.film(x, cond_var)

        return x


class SmallStem16(SmallStem):
    """SmallStem with patch_size=16."""

    def __init__(
        self,
        use_film: bool = False,
        kernel_sizes: Tuple[int, ...] = (3, 3, 3, 3),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        features: Tuple[int, ...] = (32, 96, 192, 384),
        padding: Tuple[int, ...] = (1, 1, 1, 1),
        num_features: int = 512,
        img_norm_type: str = "default",
    ):
        super().__init__(
            use_film=use_film,
            patch_size=16,  # Fixed to 16
            kernel_sizes=kernel_sizes,
            strides=strides,
            features=features,
            padding=padding,
            num_features=num_features,
            img_norm_type=img_norm_type,
        )
