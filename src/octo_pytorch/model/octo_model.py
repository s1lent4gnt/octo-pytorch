import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Config, T5EncoderModel

from octo_pytorch.model.components.image_tokenizer import (
    SmallStem,
    generate_proper_pad_mask,
    regex_filter,
)
from octo_pytorch.model.components.transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
    TokenGroup,
)


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
        # Tokenize the strings
        return self.tokenizer(strings, **self.tokenizer_kwargs)


class OctoModel(nn.Module):
    """Implementation of the Octo model."""

    def __init__(self, model_name: str = "octo-base", repeat_task_tokens: bool = False):
        super().__init__()

        # Core transformer configuration
        if model_name == "octo-base":
            self.token_embedding_size = 768
            self.num_layers = 12
            self.num_heads = 12
            self.mlp_dim = 3072
        elif model_name == "octo-small":
            self.token_embedding_size = 384
            self.num_layers = 12
            self.num_heads = 6
            self.mlp_dim = 1536
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.max_horizon = 10
        self.repeat_task_tokens = repeat_task_tokens

        # Initialize text processor
        self.text_processor = TextProcessor(
            tokenizer_name="t5-base",
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            },
        )

        # Create positional embeddings
        # Primary observation tokens: 256 tokens with d_model dimensions
        self.obs_primary_pos_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 256, self.token_embedding_size) * 0.02
        )

        # Wrist observation tokens: 64 tokens with d_model dimensions
        self.obs_wrist_pos_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 64, self.token_embedding_size) * 0.02
        )

        # Language task tokens: 16 tokens with d_model dimensions
        self.task_language_pos_embedding = nn.Parameter(torch.randn(1, 16, self.token_embedding_size) * 0.02)

        # Readout token embeddings - now separate from the transformer
        self.readout_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 1, self.token_embedding_size) * 0.02
        )

        # Initialize components
        self.observation_tokenizers = nn.ModuleDict()
        self.task_tokenizers = nn.ModuleDict()
        self.transformer = OctoTransformer(
            d_model=self.token_embedding_size,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.mlp_dim,
            max_horizon=self.max_horizon,
            repeat_task_tokens=self.repeat_task_tokens,
        )
        self.action_head = DiffusionActionHead(
            readout_key="readout_action",
            use_map=False,
            input_dim=self.token_embedding_size,
            action_dim=7,
            action_horizon=4,
        )

        # Projections
        self.obs_primary_projection = nn.Linear(512, self.token_embedding_size)
        self.obs_wrist_projection = nn.Linear(512, self.token_embedding_size)
        self.task_language_projection = nn.Linear(768, self.token_embedding_size)

        # Initialize tokenizers based on config
        self._init_tokenizers()

    def _init_tokenizers(self):
        """Initialize observation and task tokenizers"""
        # Primary image tokenizer (256x256)
        primary_encoder = SmallStem(
            use_film=False,
            patch_size=16,
            kernel_sizes=(3, 3, 3, 3),
            strides=(2, 2, 2, 2),
            features=(32, 96, 192, 384),
            padding=(1, 1, 1, 1),
            num_features=512,
        )

        self.observation_tokenizers["image_primary"] = ImageTokenizer(
            encoder=primary_encoder,
            # use_token_learner=False,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            task_film_keys=[],
        )

        # Wrist image tokenizer (128x128)
        wrist_encoder = SmallStem(
            use_film=False,
            patch_size=16,
            kernel_sizes=(3, 3, 3, 3),
            strides=(2, 2, 2, 2),
            features=(32, 96, 192, 384),
            padding=(1, 1, 1, 1),
            num_features=512,
        )

        self.observation_tokenizers["image_wrist"] = ImageTokenizer(
            encoder=wrist_encoder,
            # use_token_learner=False,
            obs_stack_keys=["image_wrist"],
            task_stack_keys=["image_wrist"],
            task_film_keys=[],
        )

        # Language tokenizer
        self.task_tokenizers["language_instruction"] = LanguageTokenizer(finetune_encoder=False)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        tasks: Dict[str, torch.Tensor],
        timestep_pad_mask: torch.Tensor,
        embodiment_action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass through the model."""

        batch_size, horizon = timestep_pad_mask.shape

        # Define attention rules
        task_attention_rules = {"task_*": AttentionRule.CAUSAL}
        obs_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
        }

        # Create prefix groups for task tokens
        prefix_groups = []
        for name, tokenizer in self.task_tokenizers.items():
            if name in tasks:
                token_group = tokenizer(tasks[name])
                projected_tokens = self.task_language_projection(token_group.tokens)

                # Add positional embedding
                group_name = f"task_{name}"
                pos_embedding = self.task_language_pos_embedding[:, : projected_tokens.shape[1]]
                processed_tokens = projected_tokens + pos_embedding

                prefix_groups.append(
                    PrefixGroup(
                        tokens=processed_tokens,
                        mask=token_group.mask,
                        name=group_name,
                        attention_rules=task_attention_rules,
                    )
                )

        # Create timestep groups for observation tokens
        timestep_groups = []
        for name, tokenizer in self.observation_tokenizers.items():
            if name in observations:
                token_group = tokenizer(observations, tasks)

                # Project tokens
                if name == "image_primary":
                    projected_tokens = self.obs_primary_projection(token_group.tokens)
                    pos_embedding = self.obs_primary_pos_embedding
                elif name == "image_wrist":
                    projected_tokens = self.obs_wrist_projection(token_group.tokens)
                    pos_embedding = self.obs_wrist_pos_embedding
                else:
                    projected_tokens = token_group.tokens
                    pos_embedding = None

                # Add positional embedding
                if pos_embedding is not None:
                    processed_tokens = projected_tokens + pos_embedding[:, : projected_tokens.shape[1]]
                else:
                    processed_tokens = projected_tokens

                # Create mask
                mask = torch.logical_and(timestep_pad_mask.unsqueeze(-1), token_group.mask)

                timestep_groups.append(
                    TimestepGroup(
                        tokens=processed_tokens,
                        mask=mask,
                        name=f"obs_{name}",
                        attention_rules=obs_attention_rules,
                    )
                )

        # Add readout tokens
        readout_tokens = torch.zeros(
            (batch_size, horizon, 1, self.token_embedding_size), device=timestep_pad_mask.device
        )
        readout_tokens += self.readout_embedding[:, :horizon]
        readout_mask = torch.ones((batch_size, horizon, 1), dtype=torch.bool, device=timestep_pad_mask.device)
        readout_attention_rules = {
            "task_*": AttentionRule.CAUSAL,
            "obs_*": AttentionRule.CAUSAL,
            "readout_action": AttentionRule.CAUSAL,
        }
        timestep_groups.append(
            TimestepGroup(
                tokens=readout_tokens,
                mask=readout_mask,
                name="readout_action",
                attention_rules=readout_attention_rules,
            )
        )

        # Run transformer
        _, timestep_outputs = self.transformer(prefix_groups, timestep_groups)

        # Create transformer outputs dict for action head
        transformer_outputs = {}
        for group in timestep_outputs:
            transformer_outputs[group.name] = group

        # Generate actions using the corrected interface
        actions = self.action_head.predict_action(
            transformer_outputs=transformer_outputs, embodiment_action_dim=embodiment_action_dim
        )

        return actions

    def create_tasks(
        self, goals: Optional[Dict[str, torch.Tensor]] = None, texts: Optional[Sequence[str]] = None
    ):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update(
                {k: torch.ones(v.shape[:1], dtype=torch.bool) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            # This part is tricky because self.example_batch is not available here.
            # For now, I will assume that if goals are not provided, we are in language-conditioned mode
            # and don't need to create dummy image goal tensors.
            # A more robust solution might require passing example_batch during initialization.
            pass

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = self.text_processor.encode(texts)
            tasks["pad_mask_dict"]["language_instruction"] = torch.ones(len(texts), dtype=torch.bool)
        else:
            batch_size = next(iter(goals.values())).shape[0]
            # Create dummy language instructions if none are provided.
            # The text processor expects a list of strings.
            dummy_texts = [""] * batch_size
            tasks["language_instruction"] = self.text_processor.encode(dummy_texts)
            tasks["pad_mask_dict"]["language_instruction"] = torch.zeros(batch_size, dtype=torch.bool)

        return tasks


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
        # use_token_learner: bool = False,
        num_tokens: int = 8,
        conditioning_type: str = "none",
        obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*"),
        task_stack_keys: Sequence[str] = tuple(),
        task_film_keys: Sequence[str] = tuple(),
        proper_pad_mask: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        # self.use_token_learner = use_token_learner
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys
        self.proper_pad_mask = proper_pad_mask

        # # Initialize token learner if needed
        # if use_token_learner:
        #     self.token_learner = TokenLearner(num_tokens=num_tokens)

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
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found. Skipping tokenizer entirely."
            )
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
                    logging.info(f"No task inputs matching {k} were found. Replacing with zero padding.")
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
            enc_inputs = torch.cat([enc_inputs, task_inputs], dim=-1)

        # Get shape information
        b, t, h, w, c = enc_inputs.shape

        # Reshape for encoder
        enc_inputs = enc_inputs.reshape(b * t, h, w, c)

        # Extract non-spatial FiLM inputs
        encoder_input_kwargs = {}
        # if self.task_film_keys and tasks is not None:
        #     film_keys = regex_filter(self.task_film_keys, sorted(tasks.keys()))
        #     if len(film_keys) > 0:
        #         film_inputs = extract_inputs(film_keys, tasks)
        #         # Repeat film inputs for each timestep
        #         film_inputs = film_inputs.unsqueeze(1).repeat(1, t, 1)
        #         encoder_input_kwargs.update(
        #             {"cond_var": film_inputs.reshape(b * t, -1)}
        #         )

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

        # # Apply token learner if specified
        # if self.use_token_learner:
        #     # Reshape for token learner: (b, t, n_tokens, c) -> (b*t, n_tokens, c)
        #     orig_shape = image_tokens.shape
        #     image_tokens = image_tokens.reshape(-1, *image_tokens.shape[2:])

        #     # Apply token learner
        #     image_tokens = self.token_learner(image_tokens, train=train)

        #     # Reshape back: (b*t, num_tokens, c) -> (b, t, num_tokens, c)
        #     image_tokens = image_tokens.reshape(b, t, self.num_tokens, -1)

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

    def __init__(self, finetune_encoder: bool = False):
        super().__init__()
        # Use T5-base configuration
        self.t5_config = T5Config.from_pretrained("t5-base")
        self.t5_encoder = T5EncoderModel(self.t5_config)
        self.finetune_encoder = finetune_encoder

        # Freeze T5 encoder if finetune_encoder is False
        if not self.finetune_encoder:
            for param in self.t5_encoder.parameters():
                param.requires_grad = False
            print("❄️ T5 encoder frozen (finetune_encoder=False)")
        else:
            print("🔥 T5 encoder trainable (finetune_encoder=True)")

    def forward(self, language_input: Dict[str, torch.Tensor]) -> TokenGroup:
        """
        Args:
            language_input: Dict with 'input_ids' and 'attention_mask'
        Returns:
            TokenGroup containing tokens and mask
        """
        # Run T5 encoder
        outputs = self.t5_encoder(
            input_ids=language_input["input_ids"], attention_mask=language_input["attention_mask"]
        )

        # Return T5 output directly (no projection, no positional embeddings)
        tokens = outputs.last_hidden_state

        # The mask is the attention mask from the input
        mask = language_input["attention_mask"].bool()

        # Handle dimension expansion if needed
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
            if mask.ndim == 2:
                mask = mask[:, None, :]

        return TokenGroup(tokens, mask)


class OctoTransformer(nn.Module):
    """Implementation of the Octo transformer with block-wise attention using BlockTransformer"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_horizon: int,
        repeat_task_tokens: bool = False,
    ):
        super().__init__()
        self.repeat_task_tokens = repeat_task_tokens

        # Create BlockTransformer with appropriate transformer kwargs
        transformer_kwargs = {
            "num_layers": num_layers,
            "mlp_dim": dim_feedforward,
            "num_attention_heads": nhead,
            "dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
            "add_position_embedding": False,
            "d_model": d_model,
        }

        self.transformer = BlockTransformer(
            transformer_kwargs=transformer_kwargs,
            enforce_causal=True,
        )

    def forward(
        self, prefix_groups: List[PrefixGroup], timestep_groups: List[TimestepGroup]
    ) -> Tuple[List[PrefixGroup], List[TimestepGroup]]:
        """
        A simple wrapper around the BlockTransformer.

        Args:
            prefix_groups: List of PrefixGroup objects for the transformer.
            timestep_groups: List of TimestepGroup objects for the transformer.

        Returns:
            A tuple of (prefix_outputs, timestep_outputs).
        """
        # Apply repeat_task_tokens logic if enabled
        if self.repeat_task_tokens:
            logging.info("repeating task tokens at each timestep to perform cross-modal attention")
            # Get task tokens from prefix groups
            for task_group in prefix_groups:
                # task_group.tokens shape: (batch, n_tokens, token_embedding_size)
                task_tokens = task_group.tokens.unsqueeze(1)  # Add timestep dimension
                ws = timestep_groups[0].tokens.shape[1]  # Get horizon/window size
                task_tokens = task_tokens.repeat(1, ws, 1, 1)  # Repeat for each timestep

                task_pad_mask = task_group.mask.unsqueeze(1)  # Add timestep dimension
                task_pad_mask = task_pad_mask.repeat(1, ws, 1)  # Repeat for each timestep

                group_name = f"obs_{task_group.name}"
                timestep_groups.append(
                    TimestepGroup(
                        tokens=task_tokens,
                        mask=task_pad_mask,
                        name=group_name,
                        attention_rules=task_group.attention_rules,
                    )
                )

        # The BlockTransformer now directly takes the groups
        prefix_outputs, timestep_outputs = self.transformer(
            prefix_groups=prefix_groups, timestep_groups=timestep_groups
        )

        return prefix_outputs, timestep_outputs


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
            f = 2 * math.pi * torch.matmul(x, self.kernel.T)
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
    
    def __init__(self, features: int, activation, dropout_rate: Optional[float] = None, use_layer_norm: bool = False):
        super().__init__()
        self.features = features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
        
        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        
        # # Residual connection with projection if needed
        # if residual.shape != x.shape:
        #     residual_proj = nn.Linear(residual.shape[-1], self.features, device=x.device, dtype=x.dtype)
        #     residual = residual_proj(residual)
        
        return residual + x


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

        # Store all parameters to match JAX version exactly
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

        # Create diffusion model exactly like JAX create_diffusion_model
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

        # Create beta schedule exactly like JAX
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

        # Handle initialization case
        if time is None or noisy_actions is None:
            if not hasattr(self, "_initializing") or not self._initializing:
                raise ValueError("Must provide time and noisy_actions when calling diffusion action head")
            else:
                time = torch.zeros((*embeddings.shape[:2], 1), dtype=torch.float32, device=embeddings.device)
                noisy_actions = torch.zeros(
                    (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                    dtype=torch.float32,
                    device=embeddings.device,
                )

        # Run diffusion model
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time)
        return pred_eps

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        embodiment_action_dim: Optional[int] = None,
        sample_shape: tuple = (),
    ) -> torch.Tensor:
        """Convenience method for predicting actions for the final timestep."""

        if embodiment_action_dim is None:
            logging.warning(
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

        # DDPM sampling loop
        def scan_fn(current_x, time):
            input_time = torch.full((*current_x.shape[:-1], 1), time, device=device, dtype=torch.float32)

            eps_pred = self.forward(transformer_outputs, input_time, current_x)

            alpha_1 = 1 / torch.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / torch.sqrt(1 - self.alpha_hats[time])
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            z = torch.zeros(current_x.shape, device=device)
            current_x = current_x + (time > 0) * (torch.sqrt(self.betas[time]) * z)

            current_x = torch.clamp(current_x, -self.max_action, self.max_action)

            # Set non-eval actions to the noise that would have been seen during training
            current_x = torch.where(flat_action_mask, current_x, torch.sqrt(1 - self.alpha_hats[time]) * z)

            return current_x

        # Initialize with noise
        noise = torch.zeros(
            (*sample_shape, batch_size, window_size, self.action_horizon * self.action_dim),
            device=device,
        )

        # Run reverse diffusion
        actions_flat = noise
        for time in reversed(range(self.diffusion_steps)):
            actions_flat = scan_fn(actions_flat, time)

        # Reshape and return last timestep
        actions = actions_flat.view(
            *sample_shape, batch_size, window_size, self.action_horizon, self.action_dim
        )
        # Only get the last timestep in the window
        return actions[..., -1, :, :]


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
        """Forward pass."""

        # Time preprocessing
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff)

        # Broadcast obs_enc if needed
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            logging.debug("Broadcasting obs_enc from %s to %s", obs_enc.shape, new_shape)
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
        """Forward pass."""

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.activation(x)
        x = self.output_proj(x)

        return x




if __name__ == "__main__":
    # Example usage
    print("Creating Octo model...")

    # Create model
    model = OctoModel(model_name="octo-base", repeat_task_tokens=True)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass with dummy data
    batch_size, horizon = 2, 2

    observations = {
        "image_primary": torch.randint(0, 255, (batch_size, horizon, 256, 256, 3), dtype=torch.uint8),
        "image_wrist": torch.randint(0, 255, (batch_size, horizon, 128, 128, 3), dtype=torch.uint8),
    }

    # Use the model's text processor to encode sample instructions
    sample_instructions = ["Pick up the red block and place it on the blue block"] * batch_size
    tasks = {"language_instruction": model.text_processor.encode(sample_instructions)}

    timestep_pad_mask = torch.ones(batch_size, horizon, dtype=torch.bool)

    print("Running forward pass...")
    with torch.no_grad():
        try:
            actions = model(observations, tasks, timestep_pad_mask)
            print(f"Output actions shape: {actions.shape}")
            print("Forward pass successful!")
        except Exception as e:
            print(f"Forward pass failed: {e}")
