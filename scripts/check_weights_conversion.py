import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random

import jax
import numpy as np
import torch
from octo.model.octo_model import OctoModel as JaxOctoModel

from octo_pytorch.model.octo_model import OctoModel as PyTorchOctoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="octo-base",
        help="Model name to use (e.g., 'octo-base', 'octo-small')",
    )
    args = parser.parse_args()

    # for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Set random seed to {seed}")

    model_name = args.model_name
    if model_name == "octo-base":
        jax_model = JaxOctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    elif model_name == "octo-small":
        jax_model = JaxOctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Save dataset statistics for PyTorch comparison
    np.save(f"output/dataset_statistics_{model_name}.npy", jax_model.dataset_statistics)
    print("Dataset statistics saved to dataset_statistics.npy")

    jax_image_primary = np.zeros(shape=(256, 256, 3), dtype=np.float32)
    jax_image_wrist = np.zeros(shape=(128, 128, 3), dtype=np.float32)
    jax_task_text = ["pick up the fork"]
    jax_image_primary = jax_image_primary[np.newaxis, np.newaxis, ...]
    jax_image_wrist = jax_image_wrist[np.newaxis, np.newaxis, ...]

    jax_observation = {
        "image_primary": jax_image_primary,
        "image_wrist": jax_image_wrist,
        "timestep_pad_mask": np.array([[True]]),
    }
    jax_task = jax_model.create_tasks(texts=jax_task_text)

    jax_action = jax_model.sample_actions(
        jax_observation,
        jax_task,
        unnormalization_statistics=jax_model.dataset_statistics["bridge_dataset"]["action"],
        rng=jax.random.PRNGKey(0),
    )

    # ##############################################

    torch_model = PyTorchOctoModel(model_name=model_name, repeat_task_tokens=True)
    torch_model.load_state_dict(torch.load(f"output/pytorch_{model_name}_model.pth"))
    torch_model.eval()
    print("Model loaded successfully.")

    dunmmy_image_primary = np.zeros(shape=(256, 256, 3), dtype=np.float32)
    dunmmy_image_wrist = np.zeros(shape=(128, 128, 3), dtype=np.float32)
    task_text = ["pick up the fork"]

    torch_image_primary = torch.from_numpy(dunmmy_image_primary).to(torch.float32).unsqueeze(0).unsqueeze(0)
    torch_image_wrist = torch.from_numpy(dunmmy_image_wrist).to(torch.float32).unsqueeze(0).unsqueeze(0)

    torch_observations = {
        "image_primary": torch_image_primary,
        "image_wrist": torch_image_wrist,
    }

    torch_tasks = {"language_instruction": torch_model.text_processor.encode(task_text)}

    timestep_pad_mask = torch.ones(1, 1, dtype=torch.bool)

    with torch.no_grad():
        # Pass embodiment_action_dim=7 for bridge_dataset (7 action dimensions)
        torch_action = torch_model(
            torch_observations, torch_tasks, timestep_pad_mask, embodiment_action_dim=7
        )

    stats = np.load(f"output/dataset_statistics_{model_name}.npy", allow_pickle=True).item()
    action_stats = stats["bridge_dataset"]["action"]
    # Ensure stats are float32 to match model's precision
    mean = torch.from_numpy(action_stats["mean"]).to(torch_action.device).float()
    std = torch.from_numpy(action_stats["std"]).to(torch_action.device).float()
    torch_action = torch_action * std + mean

    print("Finished inference.")

    # Compare the outputs
    print("Jax action:", jax_action)
    print("PyTorch action:", torch_action.squeeze().detach().numpy())
    np.testing.assert_allclose(jax_action.squeeze(), torch_action.squeeze().detach().numpy(), rtol=1e-2)
    print("Outputs are the same!")


if __name__ == "__main__":
    main()
