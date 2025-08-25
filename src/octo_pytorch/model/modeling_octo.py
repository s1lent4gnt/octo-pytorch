import os
from typing import Dict, List, Optional, Sequence

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from octo_pytorch.model.components.action_heads import DiffusionActionHead
from octo_pytorch.model.components.block_transformer import (
    AttentionRule,
    BlockTransformer,
    PrefixGroup,
    TimestepGroup,
)
from octo_pytorch.model.components.image_encoders import SmallStem16
from octo_pytorch.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo_pytorch.model.configuration_octo import OctoConfig
from octo_pytorch.utils.text_processing import TextProcessor


class OctoModel(PreTrainedModel):
    """Implementation of the Octo model."""
    config = OctoConfig

    def __init__(self, config: OctoConfig):
        super().__init__(config)
        self.text_processor = TextProcessor(
            tokenizer_name="t5-base",
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            },
        )
        self.octo_transformer = OctoTransformer(
            d_model=config.token_embedding_size,
            nhead=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.mlp_dim,
            max_horizon=config.action_max_horizon,
            repeat_task_tokens=config.repeat_task_tokens,
        )
        self.action_head = DiffusionActionHead(
            readout_key="readout_action",
            use_map=False,
            input_dim=config.token_embedding_size,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        self.transformer_outputs = None

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        tasks: Dict[str, torch.Tensor],
        timestep_pad_mask: torch.Tensor,
        embodiment_action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            observations: Dict of observation tensors
            tasks: Dict of task tensors
            timestep_pad_mask: Boolean mask for timesteps
            embodiment_action_dim: Optional action dimension for embodiment

        Returns:
            Predicted actions tensor
        """

        # Run transformer
        self.transformer_outputs = self.octo_transformer(observations, tasks, timestep_pad_mask)
        actions = self.action_head.predict_action(
            transformer_outputs=self.transformer_outputs, embodiment_action_dim=embodiment_action_dim
        )
        return actions

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return model

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
            # Create dummy goals if none are provided.
            tasks.update({"image_primary": torch.zeros((batch_size, 256, 256, 3), dtype=torch.uint8)})
            tasks.update({"image_primary": torch.zeros((batch_size, 256, 256, 3), dtype=torch.uint8)})
            tasks.update({"timestep": torch.zeros((batch_size), dtype=torch.int32)})

            tasks["pad_mask_dict"].update(
                {k: torch.zeros(batch_size, dtype=torch.bool) for k in tasks.keys() if k != "pad_mask_dict"}
            )

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

        self.max_horizon = max_horizon
        self.repeat_task_tokens = repeat_task_tokens
        self.token_embedding_size = d_model

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

        # Readout token embeddings
        self.readout_embedding = nn.Parameter(
            torch.randn(1, self.max_horizon, 1, self.token_embedding_size) * 0.02
        )

        # Initialize components
        self.observation_tokenizers = nn.ModuleDict()
        self.task_tokenizers = nn.ModuleDict()

        # Projections
        self.obs_primary_projection = nn.Linear(512, self.token_embedding_size)
        self.obs_wrist_projection = nn.Linear(512, self.token_embedding_size)
        self.task_language_projection = nn.Linear(768, self.token_embedding_size)

        # Initialize tokenizers based on config
        self._init_tokenizers()

        # tmp
        self.prefix_groups = None
        self.timestep_groups = None
        self.transformer_outputs = None

    def _init_tokenizers(self):
        """Initialize observation and task tokenizers"""
        # Primary image tokenizer (256x256)
        primary_encoder = SmallStem16()

        self.observation_tokenizers["image_primary"] = ImageTokenizer(
            encoder=primary_encoder,
            # use_token_learner=False,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            task_film_keys=[],
        )

        # Wrist image tokenizer (128x128)
        wrist_encoder = SmallStem16()

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
        timestep_pad_mask: torch.Tensor
    ) -> List[TimestepGroup]:
        """
        A simple wrapper around the BlockTransformer.

        Args:
            prefix_groups: List of PrefixGroup objects for the transformer.
            timestep_groups: List of TimestepGroup objects for the transformer.

        Returns:
            A tuple of (prefix_outputs, timestep_outputs).
        """
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
                token_group = tokenizer(tasks[name], tasks)
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

        # Apply repeat_task_tokens logic if enabled
        if self.repeat_task_tokens:
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

        prefix_outputs, timestep_outputs = self.transformer(
            prefix_groups=prefix_groups, timestep_groups=timestep_groups
        )

        self.prefix_groups = prefix_groups
        self.timestep_groups = timestep_groups

        # Create transformer outputs dict for action head
        transformer_outputs = {}
        for group in timestep_outputs:
            transformer_outputs[group.name] = group

        return transformer_outputs


if __name__ == "__main__":
    # Example usage
    print("Creating Octo model...")

    # Create configuration
    cfg = OctoConfig(model_name="octo-small")

    # Create model
    model = OctoModel(cfg)

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
