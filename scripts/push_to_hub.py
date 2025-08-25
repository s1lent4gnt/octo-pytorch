import argparse
import json
import os

import numpy as np
import torch
from huggingface_hub import HfApi
from octo_pytorch.model.modeling_octo import OctoModel
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(
        description="Upload Octo models to HuggingFace Hub"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["octo-small", "octo-base"],
        help="Model name (octo-small or octo-base)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the original checkpoint file",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., lilkm/octo-small)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=True,
        help="Push to HuggingFace Hub (default: True)",
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    # Initialize the model
    print(f"Initializing {args.model_name} model...")
    model = OctoModel(model_name=args.model_name, repeat_task_tokens=True)

    # Load the state dict into the model
    model.load_state_dict(checkpoint, strict=True)
    print("Model weights loaded successfully!")

    # Load dataset statistics
    stats_path = f"output/dataset_statistics_{args.model_name}.npy"
    if os.path.exists(stats_path):
        print(f"Loading dataset statistics from {stats_path}...")
        octo_stats = np.load(stats_path, allow_pickle=True).item()
    else:
        print(f"Warning: Dataset statistics not found at {stats_path}")
        octo_stats = {}

    # Save locally first
    local_save_path = f"output/output_hub/{args.model_name}_hub"
    os.makedirs(local_save_path, exist_ok=True)

    # Save model weights in safetensors format
    model_path = os.path.join(local_save_path, "model.safetensors")
    print(f"Saving model weights to {model_path}...")

    # Get state dict and handle shared tensors
    state_dict = model.state_dict()

    # Remove T5 encoder weights since T5-base is already on HuggingFace Hub
    # Users will load T5 separately from the hub
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

    # Save dataset statistics
    if octo_stats:
        stats_save_path = os.path.join(local_save_path, "dataset_statistics.npy")
        np.save(stats_save_path, octo_stats)
        print(f"Saved dataset statistics to {stats_save_path}")

    # Create config.json for model metadata
    config = {
        "model_type": "octo",
        "model_name": args.model_name,
        "token_embedding_size": model.token_embedding_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "mlp_dim": model.mlp_dim,
        "max_horizon": model.max_horizon,
        "repeat_task_tokens": model.repeat_task_tokens,
        "action_horizon": model.action_head.action_horizon,
        "action_dim": model.action_head.action_dim,
        "diffusion_steps": model.action_head.diffusion_steps,
    }

    config_path = os.path.join(local_save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config to {config_path}")

    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.repo_id}...")

        # Create model card content
        model_card_content = f"""---
license: mit
tags:
- robotics
- imitation-learning
- octo
- pytorch
---

# {args.model_name.title()} PyTorch Model

This is the {args.model_name} model converted to PyTorch format.

## Model Description

Octo is a generalist robot policy trained on diverse robot manipulation tasks.

- **Paper**: [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/pdf/2405.12213)
- **Original Implementation**: [octo-models/octo](https://github.com/octo-models/octo)
- **Model Size**: {args.model_name}

## Usage

### Loading the pretrained model

```python
import torch
from safetensors.torch import load_file
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize model (assuming you have the model class)
from octo_pytorch.model.octo_model import OctoModel
model = OctoModel(model_name=config['model_name'], repeat_task_tokens=config['repeat_task_tokens'])

# Load weights (T5 encoder weights will be loaded automatically from HuggingFace Hub)
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict, strict=False)  # strict=False because T5 weights are not in the file
```

**Note**: The T5-base language encoder weights are not included in this upload to save space. They will be automatically downloaded from HuggingFace Hub when you initialize the model.

### Model Architecture

- **Transformer**: {"12 layers, 768 dim, 12 heads" if args.model_name == "octo-base" else "12 layers, 384 dim, 6 heads"}
- **Vision Encoder**: Custom CNN (SmallStem16)
- **Language Encoder**: T5-Base
- **Action Head**: Diffusion policy with {model.action_head.action_horizon} action steps
- **Max Horizon**: {model.max_horizon} timesteps
- **Action Dimension**: {model.action_head.action_dim}

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

        # Write model card
        model_card_path = os.path.join(local_save_path, "README.md")
        with open(model_card_path, "w") as f:
            f.write(model_card_content)

        # Initialize HuggingFace API
        api = HfApi()

        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=args.repo_id,
                private=args.private,
                exist_ok=True,
                repo_type="model",
            )
            print(f"Repository {args.repo_id} created/verified on HuggingFace Hub")
        except Exception as e:
            print(f"Error creating repository: {e}")
            raise

        # Upload the entire folder to the hub
        try:
            api.upload_folder(
                folder_path=local_save_path,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {args.model_name} pretrained weights",
            )
            print(f"Successfully uploaded to https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"Error uploading to hub: {e}")
            raise

    print("Done!")


if __name__ == "__main__":
    main()
