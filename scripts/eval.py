import os
import sys

import gymnasium as gym
import torch
import torchvision.transforms.functional as F

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.octo_pytorch.model.octo_model import OctoModel
from src.octo_pytorch.utils.gym_wrappers import HistoryWrapper, RHCWrapper

import gym_hil
# Add the lerobot path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output/lerobot/src')))
from lerobot.scripts.rl.gym_manipulator import (
    GymHilObservationProcessorWrapper,
    GymHilDeviceWrapper,
    BatchCompatibleWrapper,
    TorchActionWrapper
)


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


def evaluate_policy(
    model: OctoModel,
    env: gym.Env,
    num_episodes: int = 3,
    max_steps_per_episode: int = 400,
    embodiment_action_dim: int = 7,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    
    total_rewards = 0
    
    for episode_idx in range(num_episodes):
        print(f"\nStarting evaluation episode {episode_idx + 1}/{num_episodes}")
        
        obs, info = env.reset()
        episode_reward = 0
        
        raw_tasks = ["pick the pink cube"]
        language_instruction = model.text_processor.encode(raw_tasks)
        language_instruction = {k: v.to(device) for k, v in language_instruction.items()}

        tasks = {
            'language_instruction': language_instruction,
            'pad_mask_dict': {'language_instruction': torch.ones(1, dtype=torch.bool, device=device)}
        }

        for _ in range(max_steps_per_episode):
            observations, timestep_pad_mask = transform_observation_for_eval(obs, model, device)

            with torch.no_grad():
                actions = model(observations, tasks, timestep_pad_mask, embodiment_action_dim=embodiment_action_dim, training=False)

            action_tensor = actions.squeeze(0)

            obs, reward, terminated, truncated, info = env.step(action_tensor)
            episode_reward += reward
            if terminated or truncated:
                break

        print(f"Episode {episode_idx+1} finished with reward: {episode_reward}")
        total_rewards += episode_reward
        
    avg_reward = total_rewards / num_episodes
    model.train()
    return avg_reward


def main():
    model_name = "octo-small"
    checkpoint_path = "output/models/octo-small_final_gold.pth"
    env_name = "gym_hil/PandaPickCubeGamepad-v0"
    horizon = 1
    exec_horizon = 2

    # Set device
    device = "cuda"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = OctoModel(model_name=model_name, repeat_task_tokens=True)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create environment
    print(f"Creating environment: {env_name}")
    
    env = gym.make(
        env_name,
        image_obs=True,
        render_mode="human",
        use_gripper=True,
        gripper_penalty=0.1,
    )
    env = GymHilObservationProcessorWrapper(env=env)
    env = GymHilDeviceWrapper(env=env, device=device)
    env = BatchCompatibleWrapper(env=env)
    env = TorchActionWrapper(env=env, device=device)
    
    # env = HistoryWrapper(env, horizon=horizon)
    env = RHCWrapper(env, exec_horizon=exec_horizon)
            
    print("\nStarting evaluation...")
    avg_return = evaluate_policy(
        model=model,
        env=env,
        num_episodes=3,
        max_steps_per_episode=10,
        embodiment_action_dim=7,
    )
    
    print(f"\nFinal average return: {avg_return:.2f}")


if __name__ == "__main__":
    main()
