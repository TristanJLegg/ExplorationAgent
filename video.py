import os
import warnings
import sys

import torch
import imageio
import cv2
import numpy as np
import miniworld

from src.networks.actor_critic import ActorCritic
from src.world.environments import make_vec_envs
from src.world.generation import generate_world, print_map_rooms, map_rooms_equals
from src.world.hidden_room_world import HiddenRoomWorld
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
    network = ActorCritic(
        [image_shape, large_map_shape, small_map_shape],
        action_space_size,
        hyperparameters.hidden_state_size,
        hyperparameters.sequential_model
    )
    network.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0, eps=0)

    # Generate the map if the model is not specified
    if not video_model:
        map_rooms = generate_world(environment_parameters, set_prints=True)
    # Load the model if specified and use its map
    else:
        network, optimizer, steps_already_taken, loaded_seed, loaded_map_rooms, _ = load_model(network, optimizer, video_model)
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

    envs = make_vec_envs(HiddenRoomWorld, 1, environment_parameters, loaded_map_rooms, device)

    # Make videos folder if it does not exist
    if not os.path.exists("videos"):
        os.makedirs("videos")

    # Video writer
    video_writer = imageio.get_writer(f"videos/{get_name_without_path(video_model)}.mp4", fps=video_fps)

    for _ in range(video_num_episodes):
        observation = envs.reset()
        done = torch.zeros((1, 1)).to(device)

        hxs = torch.zeros((1, 512), dtype=torch.float32).to(device)
        episode_reward = 0
        step_count = 0

        while not done:
            network_observation = [observation["image"], observation["large_map"], observation["small_map"]]
            action, _, _, hxs = network.action(network_observation, hxs, done)
            action = action.detach()
            hxs = hxs.detach()

            observation, reward, done, info = envs.step(action)
            observation["image"].detach()
            observation["large_map"].detach()
            observation["small_map"].detach()
            done = done.detach()
            reward = reward.detach()

            episode_reward += reward
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

            step_count += 1
            if step_count % 1000 == 0:
                print(f"Step: {step_count}")

            video_writer.append_data(image_render)

        print(f"Episode Reward: {episode_reward}")
        print(f"Info: {info}")

    # Close 
    video_writer.close()
    envs.close()
    print("Video Saved.")

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python video.py <config_file>"
    config_file = sys.argv[1]  
    main(config_file)