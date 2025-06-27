import os
import warnings
import sys

import torch
import imageio
import cv2
import numpy as np
import miniworld

from src.storage.storage import Storage
from src.world.environments import make_vec_envs
from src.world.generation import generate_world, print_map_rooms, map_rooms_equals
from src.world.hidden_room_world import HiddenRoomWorld
from src.networks.gpt import GPT, GPTConfig
from src.load_and_save import load_model, get_name_without_path
from src.parameters import load_video_parameters, load_environment_parameters, load_hyperparameters

def main(config_file: str):
    # Setup
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA Device Found: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No CUDA device found: Using CPU")

    # Load the config
    video_parameters = load_video_parameters(config_file)

    video_model = video_parameters.video_model
    video_num_episodes = video_parameters.video_num_episodes
    video_fps = video_parameters.video_fps

    environment_parameters = load_environment_parameters(config_file)

    image_shape = 3, environment_parameters.obs_height, environment_parameters.obs_width
    large_map_shape = 3, environment_parameters.large_map_size, environment_parameters.large_map_size
    small_map_shape = 3, environment_parameters.small_map_size, environment_parameters.small_map_size
    action_space_size = 3

    hyperparameters = load_hyperparameters(config_file)

    # Initialize the neural network
    model_config = GPTConfig(
        3, # 0, 1, 2 [actions]
        hyperparameters.context_window_size * 3,
        n_layer=6,
        n_head=8,
        n_embd=128,
        model_type="reward_conditioned",
        max_timestep=1
    )
    network = GPT(model_config)
    network.to(device)

    # Initialize the optimizer
    optimizer = network.configure_optimizers(weight_decay=0.1, learning_rate=hyperparameters.learning_rate, betas=(0.9, 0.95))

    # Generate the map if the model is not specified
    if not video_model:
        map_rooms = generate_world(environment_parameters, set_prints=True)
    # Load the model if specified and use its map
    else:
        network, optimizer, steps_already_taken, \
        loaded_seed, loaded_map_rooms, _ = load_model(network, optimizer, video_model)
        print("Model loaded successfully: Seed =", loaded_seed)

        # Create the vectorized environments
        environment_parameters.seed = loaded_seed
        np.random.seed(environment_parameters.seed)

        map_rooms = generate_world(environment_parameters, set_prints=False)

        # Warning if the generation algorithm generates a different map with the same seed
        # Regardless the saved map is used
        print_map_rooms(loaded_map_rooms)
        if not map_rooms_equals(map_rooms, loaded_map_rooms):
            warnings.warn(
                "Loaded map rooms do not match seed generated map rooms. \n\
                Generation algorithm and/or parameters may have changed. \n\
                Using loaded map rooms..."
            )

    envs = make_vec_envs(HiddenRoomWorld, 1, environment_parameters, loaded_map_rooms, torch.device("cpu"))

    # Make videos folder if it does not exist
    if not os.path.exists("videos"):
        os.makedirs("videos")

    # Video writer
    video_writer = imageio.get_writer(f"videos/{get_name_without_path(video_model)}.mp4", fps=video_fps)

    # Initialize the rollouts storage
    rollouts = Storage(
        hyperparameters.rollout_size,
        1,
        [image_shape, large_map_shape, small_map_shape],
        hyperparameters.context_window_size
    )

    for _ in range(video_num_episodes):
        observation = envs.reset()
        dones = torch.zeros((1, 1)).to(device)

        hxs = torch.zeros((1, 512), dtype=torch.float32).to(device)
        episode_reward = 0
        step = 0
        step_count = 0

        while not dones[0]:
            if step >= 2048:
                step = 0
                rollouts.after_update()

            with torch.no_grad():
                observations, actions, returns_to_go, timesteps, attention_mask = rollouts.get_context_window(step)
                observations = [
                    observations[0].to(device),
                    observations[1].to(device),
                    observations[2].to(device)
                ]

                actions, action_log_probabilities, value_predictions = \
                    network.action(
                        observations,
                        actions.to(device),
                        returns_to_go.to(device),
                        timesteps.to(device)
                    )
                value_predictions = value_predictions.squeeze(1)

            observations, rewards, dones, infos = envs.step(actions)
            timesteps = torch.tensor([info["step_count"] for info in infos], dtype=torch.long)

            rollouts.insert(
                list(observations.values()),
                actions.view(1, 1),
                action_log_probabilities.view(1, 1), 
                value_predictions.view(1, 1), 
                rewards.view(1, 1), 
                dones.view(1, 1),
                timesteps.view(1, 1)
            )

            episode_reward += rewards[0]
            image_render = envs.env_method("render", indices=[0])[0]
            map_render = envs.env_method("get_top_map", indices=[0])[0]

            # Resize map to 2x size
            map_render = cv2.resize(
                map_render, 
                (map_render.shape[1] * 2, map_render.shape[0] * 2), 
                interpolation=cv2.INTER_NEAREST
            )
            # Transpose first and second axes
            map_render = np.transpose(map_render, (1, 0, 2))

            # Place map on top right of image
            image_render[0:map_render.shape[0]:, -map_render.shape[1]:, :] = map_render

            step += 1
            step_count += 1
            if step_count % 100 == 0:
                print(f"Step: {step_count}")

            video_writer.append_data(image_render)

        print(f"Episode Reward: {episode_reward}")
        print(f"Info: {infos[0]}")

    # Close 
    video_writer.close()
    envs.close()
    print("Video Saved.")

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python video.py <config_file>"
    config_file = sys.argv[1]
    main(config_file)