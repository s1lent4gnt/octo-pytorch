import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse

import jax
import numpy as np
import torch
from octo.model.octo_model import OctoModel as JaxOctoModel
# from octo_pytorch.model.configuration_octo import OctoConfig as PytorchOctoConfig
from octo_pytorch.model.modeling_octo import OctoModel as PyTorchOctoModel
from octo_pytorch.policy.policy import OctoPolicy as PytorchOctoPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="octo-base",
        help="Model name to use (e.g., 'octo-base', 'octo-small')",
    )
    args = parser.parse_args()

    # # for reproducibility
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # print(f"Set random seed to {seed}")

    model_name = args.model_name
    if model_name == "octo-base":
        jax_model_name = "hf://rail-berkeley/octo-base-1.5"
        pytorch_model_name = "lilkm/octo-base-test"
    elif model_name == "octo-small":
        jax_model_name = "hf://rail-berkeley/octo-small-1.5"
        pytorch_model_name = "lilkm/octo-small-test"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    jax_model = JaxOctoModel.load_pretrained(jax_model_name)

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

    # Directly call apply to get intermediate activations
    variables = jax_model.module.apply(
        {"params": jax_model.params},
        jax_observation,
        jax_task,
        jax_observation["timestep_pad_mask"],
        train=False,
        method="octo_transformer",
        mutable=["intermediates"],
    )

    # variables[1]["intermediates"]["octo_transformer"]["tasks"]
    # variables[1]["intermediates"]["octo_transformer"]["observations"]
    jax_prefix_groups = variables[1]["intermediates"]["octo_transformer"][
        "all_prefix_groups"
    ][0][0].tokens
    jax_timestep_groups_obs_primary = variables[1]["intermediates"]["octo_transformer"][
        "all_timestep_groups"
    ][0][
        0
    ].tokens  # obs_primary
    jax_timestep_groups_obs_wrist = variables[1]["intermediates"]["octo_transformer"][
        "all_timestep_groups"
    ][0][
        1
    ].tokens  # obs_wrist
    jax_timestep_groups_obs_task_language = variables[1]["intermediates"][
        "octo_transformer"
    ]["all_timestep_groups"][0][
        2
    ].tokens  # obs_task_language
    jax_timestep_groups_readout = variables[1]["intermediates"]["octo_transformer"][
        "all_timestep_groups"
    ][0][
        3
    ].tokens  # readout_action
    jax_transformer_outputs = variables[0][
        "readout_action"
    ].tokens  # readout_action output

    jax_action = jax_model.sample_actions(
        jax_observation,
        jax_task,
        unnormalization_statistics=jax_model.dataset_statistics["bridge_dataset"][
            "action"
        ],
        rng=jax.random.PRNGKey(0),
    )

    # ##############################################

    # cfg = PytorchOctoConfig(model_name=model_name)
    policy = PytorchOctoPolicy.from_pretrained(pytorch_model_name, device="cpu")
    
    dunmmy_image_primary = np.zeros(shape=(256, 256, 3), dtype=np.float32)
    dunmmy_image_wrist = np.zeros(shape=(128, 128, 3), dtype=np.float32)
    task_text = ["pick up the fork"]

    torch_observations = {
        "image_primary": dunmmy_image_primary[np.newaxis, np.newaxis, ...],
        "image_wrist": dunmmy_image_wrist[np.newaxis, np.newaxis, ...],
        "task": task_text,
        "timestep_pad_mask": np.array([[True]]),
    }

    # Use PyTorch model's dataset statistics from HuggingFace Hub
    if policy.dataset_statistics is not None:
        torch_action = policy.get_action(torch_observations, unnormalization_statistics=policy.dataset_statistics["bridge_dataset"]["action"])["action"]
        print("Using PyTorch model's dataset statistics from HuggingFace Hub")
    else:
        # Fallback to JAX model's statistics
        torch_action = policy.get_action(torch_observations, unnormalization_statistics=jax_model.dataset_statistics["bridge_dataset"]["action"])["action"]
        print("Fallback: Using JAX model's dataset statistics")

    torch_prefix_groups = policy.model.octo_transformer.prefix_groups[0].tokens.detach().cpu()
    torch_timestep_groups_obs_primary = policy.model.octo_transformer.timestep_groups[0].tokens.detach().cpu()
    torch_timestep_groups_obs_wrist = policy.model.octo_transformer.timestep_groups[1].tokens.detach().cpu()
    torch_timestep_groups_obs_task_language = policy.model.octo_transformer.timestep_groups[2].tokens.detach().cpu()
    torch_timestep_groups_readout = policy.model.octo_transformer.timestep_groups[3].tokens.detach().cpu()
    torch_transformer_outputs = policy.model.transformer_outputs["readout_action"].tokens.detach().cpu()

    print("Finished inference.")

    print("Prefix groups")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_prefix_groups) - torch_prefix_groups.numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_prefix_groups) - torch_prefix_groups.numpy()))}"
    )

    print("Timestep group obs primary")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_timestep_groups_obs_primary) - torch_timestep_groups_obs_primary.numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_timestep_groups_obs_primary) - torch_timestep_groups_obs_primary.numpy()))}"
    )

    print("Timestep group obs wrist")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_timestep_groups_obs_wrist) - torch_timestep_groups_obs_wrist.numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_timestep_groups_obs_wrist) - torch_timestep_groups_obs_wrist.numpy()))}"
    )

    print("Timestep group obs task language")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_timestep_groups_obs_task_language) - torch_timestep_groups_obs_task_language.numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_timestep_groups_obs_task_language) - torch_timestep_groups_obs_task_language.numpy()))}"
    )

    print("TImestep group readout")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_timestep_groups_readout) - torch_timestep_groups_readout.numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_timestep_groups_readout) - torch_timestep_groups_readout.numpy()))}"
    )

    print("Transformer output readout")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_transformer_outputs) - torch_transformer_outputs.numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_transformer_outputs) - torch_transformer_outputs.numpy()))}"
    )

    print("Output action")
    print(
        f"mean diff: {np.mean(np.abs(np.array(jax_action.squeeze()) - torch_action.squeeze().detach().cpu().numpy()))}"
    )
    print(
        f"max diff: {np.max(np.abs(np.array(jax_action.squeeze()) - torch_action.squeeze().detach().cpu().numpy()))}"
    )

    print("=" * 50)
    print("Jax action:", jax_action)
    print("PyTorch action:", torch_action.squeeze().detach().cpu().numpy())
    # np.testing.assert_allclose(jax_action.squeeze(), torch_action.squeeze().detach().numpy(), rtol=1e-5, atol=1e-5)
    # print("Outputs are the same!")


if __name__ == "__main__":
    main()
