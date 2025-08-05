import torch
import torch.optim as optim
from octo_pytorch.model.octo_model import OctoModel
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import format_big_number
from lerobot.datasets.utils import cycle
import wandb
import time
from datetime import datetime
import gymnasium as gym
import os
import sys
import torchvision.transforms.functional as F


# Add the lerobot path to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output/lerobot/src')))

from lerobot.scripts.rl.gym_manipulator import GymHilObservationProcessorWrapper, GymHilDeviceWrapper, BatchCompatibleWrapper, TorchActionWrapper

def transform_batch(batch, model, device):
    """
    Transforms a batch from the LeRobotDataset format to the format expected by the OctoModel.
    """
    image_primary = batch["observation.images.front"].to(device)
    image_wrist = batch["observation.images.wrist"].to(device)
    proprio = batch["observation.state"].to(device)
    raw_actions = batch["action"].to(device)

    batch_size = raw_actions.shape[0]
    raw_tasks = ["pick the pink cube"] * batch_size
    window_size = 1
    action_horizon = model.action_head.action_horizon
    action_dim = model.action_head.action_dim

    image_primary = image_primary.permute(0, 2, 3, 1).unsqueeze(1)
    image_wrist = image_wrist.permute(0, 2, 3, 1).unsqueeze(1)

    proprio = proprio.unsqueeze(1)  # (B, W, D)

    # For window_size=1, timestep will be 0 for all samples
    timestep = torch.zeros((batch_size, window_size), dtype=torch.int32, device=device)

    # Create timestep_pad_mask - all True since we have real data (no padding)
    timestep_pad_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=device)

    task_completed = torch.zeros((batch_size, window_size, action_horizon), dtype=torch.bool, device=device)

    # Create pad_mask_dict for observations
    obs_pad_mask_dict = {
        'image_primary': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        'image_wrist': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        'proprio': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        'timestep': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
    }

    observations = {
        'image_primary': image_primary,
        'image_wrist': image_wrist,
        'proprio': proprio,
        'timestep': timestep,
        'timestep_pad_mask': timestep_pad_mask,
        'task_completed': task_completed,
        'pad_mask_dict': obs_pad_mask_dict
    }

    language_instruction = model.text_processor.encode(raw_tasks)
    language_instruction = {k: v.to(device) for k, v in language_instruction.items()}

    task_pad_mask_dict = {
        'language_instruction': torch.ones(batch_size, dtype=torch.bool, device=device)
    }

    tasks = {
        'language_instruction': language_instruction,
        'pad_mask_dict': task_pad_mask_dict
    }

    x_y_z = raw_actions[:, :3]  # x, y, z
    gripper = raw_actions[:, 3:4]  # gripper
    rx_ry_rz = torch.zeros((batch_size, 3), dtype=raw_actions.dtype, device=device)  # rx, ry, rz as zeros
    raw_actions = torch.cat([x_y_z, rx_ry_rz, gripper], dim=1)  # x, y, z, rx, ry, rz, gripper

    actions = raw_actions.reshape(batch_size, window_size, 1, action_dim)
    actions = actions.repeat(1, 1, action_horizon, 1)

    action_pad_mask = torch.ones_like(actions, dtype=torch.bool, device=device)
    if action_dim >= 7:
        # Mask out rotation dimensions (indices 3, 4, 5)
        action_pad_mask[:, :, :, 3:6] = False

    return observations, tasks, actions, action_pad_mask, timestep_pad_mask


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch

def transform_observation_for_eval(obs, model, device):
    """
    Transforms a single observation from the gym-hil environment to the format expected by the OctoModel.
    """
    image_primary_from_env = obs["observation.images.front"]
    image_wrist_from_env = obs["observation.images.wrist"]
    proprio = obs["observation.state"]

    image_primary = F.resize(image_primary_from_env, (256, 256), antialias=True)
    # image_wrist = F.resize(image_wrist_from_env, (128, 128), antialias=True)
    image_wrist = image_wrist_from_env

    batch_size = 1
    window_size = 1
    action_horizon = model.action_head.action_horizon

    image_primary = image_primary.permute(0, 2, 3, 1).unsqueeze(1)
    image_wrist = image_wrist.permute(0, 2, 3, 1).unsqueeze(1)

    proprio = proprio.unsqueeze(1)

    timestep = torch.zeros((batch_size, window_size), dtype=torch.int32, device=device)
    timestep_pad_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=device)

    task_completed = torch.zeros((batch_size, window_size, action_horizon), dtype=torch.bool, device=device)

    obs_pad_mask_dict = {
        'image_primary': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        'image_wrist': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        'proprio': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
        'timestep': torch.ones((batch_size, window_size), dtype=torch.bool, device=device),
    }

    observations = {
        'image_primary': image_primary,
        'image_wrist': image_wrist,
        'proprio': proprio,
        'timestep': timestep,
        'timestep_pad_mask': timestep_pad_mask,
        'task_completed': task_completed,
        'pad_mask_dict': obs_pad_mask_dict
    }

    return observations, timestep_pad_mask


def evaluate_policy(model, env, num_episodes=3, episode_time_limit_s=15):
    model.eval()
    total_rewards = 0
    device = next(model.parameters()).device
    for i in range(num_episodes):
        print(f"Starting evaluation episode {i+1}/{num_episodes}")
        obs, _ = env.reset()
        episode_reward = 0
        start_episode_t = time.perf_counter()

        raw_tasks = ["pick the pink cube"]
        language_instruction = model.text_processor.encode(raw_tasks)
        language_instruction = {k: v.to(device) for k, v in language_instruction.items()}

        tasks = {
            'language_instruction': language_instruction,
            'pad_mask_dict': {'language_instruction': torch.ones(1, dtype=torch.bool, device=device)}
        }

        while time.perf_counter() - start_episode_t < episode_time_limit_s:
            start_time = time.perf_counter()
            observations, timestep_pad_mask = transform_observation_for_eval(obs, model, device)

            with torch.no_grad():
                actions = model(observations, tasks, timestep_pad_mask, embodiment_action_dim=7, training=False)

            action_tensor = actions.squeeze(0)[0]

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            episode_reward += reward
            if terminated or truncated:
                break

            dt_time = time.perf_counter() - start_time
            t = 1 / 10 - dt_time
            if t > 0:
                time.sleep(t)

        print(f"Episode {i+1} finished with reward: {episode_reward}")
        total_rewards += episode_reward

    avg_reward = total_rewards / num_episodes
    model.train()
    return avg_reward


def main():
    nb_epochs = 3 # int(1e5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "lilkm/panda_pick_octo_resized"
    model_name = "octo-small"
    batch_size = 128
    learning_rate = 3e-4
    weight_decay = 0.01
    clip_gradient = 1.0
    num_workers = 4
    log_freq = 50
    save_path = "output/models"

    os.makedirs(save_path, exist_ok=True)

    # WandB configuration
    use_wandb = True
    wandb_project = "octo-finetune"
    wandb_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb_config = {
        "model_name": model_name,
        "repo_id": repo_id,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "nb_epochs": nb_epochs,
        "device": device,
    }

    # Initialize WandB
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=wandb_config,
        )

    # Initialize model
    print(f"Loading model: {model_name}")
    model = OctoModel(model_name=model_name, repeat_task_tokens=True)

    checkpoint_path = f"output/pytorch_{model_name}_model.pth"
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    for param in model.transformer.parameters():
        param.requires_grad = False

    model = model.to(device)

    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    # Create a temporary dataloader to preprocess the entire dataset on CPU
    temp_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    preprocessed_batches = []
    for raw_batch in temp_dataloader:
        batch_data = transform_batch(raw_batch, model, "cpu")
        preprocessed_batches.append(batch_data)

    print("Dataset preprocessing complete.")

    dl_iter = cycle(preprocessed_batches)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*50}")
    print(f"Training Configuration:")
    print(f"{'='*50}")
    print(f"Device: {device}")
    print(f"Dataset frames: {dataset.num_frames} ({format_big_number(dataset.num_frames)})")
    print(f"Dataset episodes: {dataset.num_episodes}")
    print(f"Learnable params: {num_learnable_params} ({format_big_number(num_learnable_params)})")
    print(f"Total params: {num_total_params} ({format_big_number(num_total_params)})")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"{'='*50}\n")

    print("Starting training...")
    start_time = time.time()

    for step in range(nb_epochs):
        step_start_time = time.time()

        # Get preprocessed batch from CPU memory
        cpu_batch = next(dl_iter)

        # Move batch to device
        observations, tasks, actions, action_pad_mask, timestep_pad_mask = move_to_device(cpu_batch, device)
        transformer_outputs = model(observations, tasks, timestep_pad_mask, training=True)

        loss, metrics = model.action_head.loss(
            transformer_outputs,
            actions,
            timestep_pad_mask,
            action_pad_mask
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient)
        optimizer.step()
        step_time = time.time() - step_start_time

        if step % log_freq == 0:
            elapsed_time = time.time() - start_time
            steps_per_second = (step + 1) / elapsed_time

            log_dict = {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step_time": step_time,
                "steps_per_second": steps_per_second,
                "step": step,
            }

            # Add any additional metrics from the loss function
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        log_dict[key] = value.item()
                    else:
                        log_dict[key] = value

            print(f"Step {step:06d} | Loss: {loss.item():.4f} | "
                  f"Grad Norm: {grad_norm.item():.4f} | "
                  f"Steps/s: {steps_per_second:.2f}")

            if use_wandb:
                wandb.log(log_dict, step=step)

    # Save the final model
    final_model_path = os.path.join(save_path, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    import gym_hil
    eval_env = gym.make(
        "gym_hil/PandaPickCubeGamepad-v0",
        image_obs=True,
        render_mode="human",
        use_gripper=True,
        gripper_penalty=0.1,
    )
    eval_env = GymHilObservationProcessorWrapper(env=eval_env)
    eval_env = GymHilDeviceWrapper(env=eval_env, device=device)
    eval_env = BatchCompatibleWrapper(env=eval_env)
    eval_env = TorchActionWrapper(env=eval_env, device=device)

    print("Running final evaluation...")
    avg_reward = evaluate_policy(model, eval_env)
    print(f"Final evaluation average reward: {avg_reward:.4f}")

    if use_wandb:
        wandb.log({"final_eval_avg_reward": avg_reward})
        wandb.finish()

if __name__ == "__main__":
    main()
