import json
import os
from typing import Dict, List, Optional, Sequence

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import HfApi
from safetensors.torch import save_file
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

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        dataset_statistics_path: Optional[str] = None,
        local_save_path: Optional[str] = None,
    ):
        """Push the model to HuggingFace Hub with model card.

        Args:
            repo_id: HuggingFace Hub repository ID (e.g., 'username/model-name')
            private: Whether to make the repository private
            dataset_statistics_path: Optional path to dataset statistics file
            local_save_path: Optional local directory to save files before uploading
        """
        model_name = (
            self.config.model_name
            if hasattr(self.config, "model_name")
            else "octo-model"
        )

        # Set default local save path if not provided
        if local_save_path is None:
            local_save_path = f"output/output_hub/{model_name}_hub"

        os.makedirs(local_save_path, exist_ok=True)
        model_path = os.path.join(local_save_path, "model.safetensors")

        # Get state dict and handle shared tensors
        state_dict = self.state_dict()

        # Remove T5 encoder weights since T5-base is already on HuggingFace Hub
        keys_to_remove = []
        for key in state_dict.keys():
            if "task_tokenizers.language_instruction.t5_encoder" in key:
                keys_to_remove.append(key)

        # Create a new state dict without the T5 weights
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k not in keys_to_remove
        }

        if keys_to_remove:
            print(
                f"Excluded T5 encoder weights ({len(keys_to_remove)} tensors) - T5-base will be loaded from HuggingFace Hub"
            )

        save_file(filtered_state_dict, model_path)

        if dataset_statistics_path and os.path.exists(dataset_statistics_path):
            octo_stats = np.load(dataset_statistics_path, allow_pickle=True).item()
            stats_save_path = os.path.join(local_save_path, "dataset_statistics.npy")
            np.save(stats_save_path, octo_stats)

        # Create config.json for model metadata
        config_dict = {
            "model_type": "octo",
            "model_name": model_name,
            "token_embedding_size": self.config.token_embedding_size,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "mlp_dim": self.config.mlp_dim,
            "max_horizon": self.config.action_max_horizon,
            "repeat_task_tokens": self.config.repeat_task_tokens,
            "action_horizon": self.action_head.action_horizon,
            "action_dim": self.action_head.action_dim,
            "diffusion_steps": self.action_head.diffusion_steps,
        }

        config_path = os.path.join(local_save_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create model card
        model_card_content = self._generate_model_card(model_name, repo_id)
        model_card_path = os.path.join(local_save_path, "README.md")
        with open(model_card_path, "w") as f:
            f.write(model_card_content)

        api = HfApi()

        try:
            api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
                repo_type="model",
            )
            print(f"Repository {repo_id} created/verified on HuggingFace Hub")
        except Exception as e:
            print(f"Error creating repository: {e}")
            raise

        # Upload the entire folder to the hub
        try:
            api.upload_folder(
                folder_path=local_save_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Error uploading to hub: {e}")
            raise

    def _generate_model_card(self, model_name: str, repo_id: str) -> str:
        """Generate a model card for the HuggingFace Hub.

        Args:
            model_name: Name of the model (e.g., 'octo-small', 'octo-base')
            repo_id: Repository ID on HuggingFace Hub

        Returns:
            Model card content as a string
        """
        # Determine model size description
        if model_name == "octo-base":
            architecture_desc = "12 layers, 768 dim, 12 heads"
        elif model_name == "octo-small":
            architecture_desc = "12 layers, 384 dim, 6 heads"

        model_card = f"""---
license: mit
tags:
- robotics
- imitation-learning
- octo
- pytorch
---

# {model_name.title()} PyTorch Model

This is the {model_name} model converted to PyTorch format.

## Model Description

Octo is a generalist robot policy trained on diverse robot manipulation tasks.

- **Paper**: [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/pdf/2405.12213)
- **Original JAX Implementation**: [octo-models/octo](https://github.com/octo-models/octo)
- **Original Pytorch Implementation**: [emb-ai/octo-pytorch](https://github.com/emb-ai/octo-pytorch)
- **lil'km Implementation**: [s1lent4gnt/octo-pytorch](https://github.com/s1lent4gnt/octo-pytorch)
- **Model Size**: {model_name}

## Usage

### Loading the pretrained model

```python
import torch
from safetensors.torch import load_file
import json
from octo_pytorch.model import OctoModel
from octo_pytorch.model.configuration_octo import OctoConfig

# Load config
with open('config.json', 'r') as f:
    config_dict = json.load(f)

# Initialize model configuration
config = OctoConfig(model_name=config_dict['model_name'])

# Initialize model
model = OctoModel(config)

# Load weights (T5 encoder weights will be loaded automatically from HuggingFace Hub)
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict, strict=False)  # strict=False because T5 weights are not in the file
```

### Alternative: Direct loading from HuggingFace Hub

```python
from octo_pytorch.model import OctoModel

# Load model directly from HuggingFace Hub
model = OctoModel.from_pretrained('{repo_id}')
```

**Note**: The T5-base language encoder weights are not included in this upload to save space. They will be automatically downloaded from HuggingFace Hub when you initialize the model.

### Model Architecture

- **Transformer**: {architecture_desc}
- **Vision Encoder**: Custom CNN (SmallStem16)
- **Language Encoder**: T5-Base
- **Action Head**: Diffusion policy with {self.action_head.action_horizon} action steps
- **Max Horizon**: {self.config.action_max_horizon} timesteps
- **Action Dimension**: {self.action_head.action_dim}

## Files

- `model.safetensors`: Model weights in safetensors format
- `config.json`: Model configuration
- `dataset_statistics.npy`: Dataset statistics used for normalization (if available)

## Citation

If you use this model, please cite:

```bibtex
@article{{octo_2023,
    title={{Octo: An Open-Source Generalist Robot Policy}},
    author={{Octo Model Team et al.}},
    journal={{arXiv preprint arXiv:2405.12213}},
    year={{2024}}
}}
```
"""
        return model_card

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
