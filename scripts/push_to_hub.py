import argparse
import os
import sys
import torch
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Upload Octo models to HuggingFace Hub")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["octo-small", "octo-base"],
        help="Model name (octo-small or octo-base)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the original checkpoint file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., lilkm/octo-small)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=True,
        help="Push to HuggingFace Hub (default: True)"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")


    # Some model config params
    action_horizon = 4


    print(f"Loading checkpoint from {args.checkpoint_path}...")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")

    # Original Octo VLA stats
    octo_stats = {}
    octo_stats = np.load(f"output/dataset_statistics_{args.model_name}.npy", allow_pickle=True).item()
    
    # Add dataset stats to state dict
    state_dict["dataset_stats"] = octo_stats
    
    # Save locally first
    local_save_path = f"outputs/{args.model_name}_lerobot"
    os.makedirs(local_save_path, exist_ok=True)
    
    # I want to save the model locally using LeRobot Mixin class

    print(f"Saving model locally to {local_save_path}...")
    save_pretrained(local_save_path)
    
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
- lerobot
---

# {args.model_name.title()} for LeRobot

This is the {args.model_name} model converted to work with LeRobot.

## Model Description

Octo is a generalist robot policy trained on diverse robot manipulation tasks.

- **Paper**: [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/pdf/2405.12213)
- **Original Implementation**: [octo-models/octo](https://github.com/octo-models/octo)
- **Model Size**: {args.model_name}

## Usage

### Loading the pretrained model

```python
from lerobot.policies.octo.modeling_octo import OctoPolicy

# Load the pretrained model
policy = OctoPolicy.from_pretrained("{args.repo_id}")
```

### Finetuning on your dataset

```bash
# Finetune with frozen transformer (only train action head)
lerobot-train \\
    --policy.path={args.repo_id} \\
    --policy.train_action_head_only=true \\
    --dataset.repo_id=your_dataset_id

# Finetune with frozen vision encoder
lerobot-train \\
    --policy.path={args.repo_id} \\
    --policy.freeze_vision_encoder=true \\
    --dataset.repo_id=your_dataset_id

# Full finetuning
lerobot-train \\
    --policy.path={args.repo_id} \\
    --policy.freeze_vision_encoder=false \\
    --policy.freeze_transformer=false \\
    --dataset.repo_id=your_dataset_id
```

## Model Architecture

- **Transformer**: {"12 layers, 768 dim, 12 heads" if args.model_name == "octo-base" else "12 layers, 384 dim, 6 heads"}
- **Vision Encoder**: Custom CNN
- **Language Encoder**: T5-Base
- **Action Head**: Diffusion policy with {action_horizon} action steps

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
        
        # Push to hub
        push_to_hub(
            args.repo_id,
            private=args.private,
            commit_message=f"Upload {args.model_name} pretrained weights"
        )
        
        print(f"Successfully uploaded to {args.repo_id}")
    
    print("Done!")


if __name__ == "__main__":
    main()
