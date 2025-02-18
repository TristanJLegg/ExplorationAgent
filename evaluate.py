import sys
import datetime
import warnings
from collections import deque

import pyglet
pyglet.options["headless"] = True

import torch
import numpy as np
import cv2

from src.load_and_save import load_model, save_model
from src.algorithms.ppo import PPO
from src.networks.actor_critic import ActorCritic
from src.networks.gpt import GPT, GPTConfig
from src.storage.storage import Storage
from src.world.environments import make_vec_envs
from src.world.generation import generate_world, print_map_rooms, map_rooms_equals
from src.logging_board import TrainingBoard
from src.world.hidden_room_world import HiddenRoomWorld
from src.parameters import load_parameters, load_hyperparameters, load_environment_parameters

def main(config_file: str, selected_device):
    # Setup torch and cuda device
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{selected_device}")
        pyglet.options["headless_device"] = selected_device
        print(f"CUDA is available... using GPU device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available... using CPU")

    # Load parameters
    parameters = load_parameters(config_file)
    model = parameters.model
    model_save_interval = parameters.model_save_interval
    console_update_interval = parameters.console_update_interval
    training_comment = parameters.training_comment

    # Load hyperparameters
    hyperparameters = load_hyperparameters(config_file)
    num_steps = hyperparameters.num_steps
    num_mini_batch = hyperparameters.num_mini_batch
    rollout_size = hyperparameters.rollout_size
    num_processes = hyperparameters.num_processes
    num_mini_batch = hyperparameters.num_mini_batch

    # Load environment parameters
    environment_parameters = load_environment_parameters(config_file)

    # Read or generate seed
    if environment_parameters.seed is not None:
        np.random.seed(environment_parameters.seed)
    else:
        environment_parameters.seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(environment_parameters.seed)

    # Initialize observation shapes and action space size
    image_shape = 3, environment_parameters.obs_height, environment_parameters.obs_width
    large_map_shape = 3, environment_parameters.large_map_size, environment_parameters.large_map_size
    small_map_shape = 3, environment_parameters.small_map_size, environment_parameters.small_map_size
    action_space_size = 3

    # Initialize the neural network
    if environment_parameters.observation_type == "RGBD":
        network_observation_shapes = [image_shape, large_map_shape, small_map_shape]
    elif environment_parameters.observation_type == "RGB":
        network_observation_shapes = [image_shape]

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

    # optimizer = torch.optim.Adam(network.parameters(), lr=hyperparameters.learning_rate, eps=hyperparameters.eps)
    optimizer = network.configure_optimizers(weight_decay=0.1, learning_rate=hyperparameters.learning_rate, betas=(0.9, 0.95))

    # Initial steps trained before this training session
    steps_already_taken = 0

    # Generate the map if the model is not specified
    if not model:
        map_rooms = generate_world(environment_parameters, set_prints=True)
    # Load the model if specified and use its map
    else:
        network, optimizer, steps_already_taken, \
        loaded_seed, loaded_map_rooms, _ = load_model(network, optimizer, model)
        map_rooms = generate_world(environment_parameters, set_prints=True)

    # Make vectorized environments
    print("Creating environments...")
    envs = make_vec_envs(HiddenRoomWorld, num_processes, environment_parameters, map_rooms, torch.device("cpu"))

    # Initializing training boards, rollouts, and algorithm
    print("Initializing training boards, rollouts, and algorithm...")

    # Initialize Date for Saving Board and Model
    training_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Logging and Plotting
    trainingboard_writer = TrainingBoard(
        parameters.training_tensorboard_directory,
        num_processes,
        f"{training_date}-{training_comment}",
        steps_already_taken
    )

    # Initialize the rollouts storage
    rollouts = Storage(
        rollout_size,
        num_processes,
        [image_shape, large_map_shape, small_map_shape],
        hyperparameters.context_window_size
    )

    # Initialize the PPO algorithm
    ppo = PPO(network, hyperparameters, optimizer)
    ppo.to(device)

    # Reset the environment
    envs.reset()

    # Episode rewards and coverage
    episode_rewards = torch.zeros(num_processes, 1)

    # Start training
    print("Starting evaluation...")
    episode_dones = torch.zeros(num_processes, 1)
    count = 0
    while not (episode_dones.flatten() == 1.0).all():
        for step in range(rollout_size):
            if count % 500 == 0:
                print(f"Step: {count}")
            count += 1
        
            # Determine actions using the current policy
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

            # Take actions in the environments
            # Using the first action from the policy TODO : Should I rather use top-k sampling?
            observations, rewards, dones, infos = envs.step(actions)
            
            # Update episode rewards and coverage
            episode_rewards += rewards

            # Logging
            trainingboard_writer.episode_update(
                episode_rewards, 
                infos,
                dones, 
                0
            )

            # Set completion
            for i in range(num_processes):
                if dones[i]:
                    episode_dones[i] = 1
                    
            # Reset environments that are done
            done_environments_indices = [i for i in range(num_processes) if dones[i]]
            envs.env_method("reset", indices=done_environments_indices)
            episode_rewards[done_environments_indices] = 0

            timesteps = torch.tensor([info["step_count"] for info in infos], dtype=torch.long)

            # Insert experiences into the rollouts storage
            rollouts.insert(
                list(observations.values()),
                actions.view(num_processes, 1),
                action_log_probabilities.view(num_processes, 1), 
                value_predictions.view(num_processes, 1), 
                rewards.view(num_processes, 1), 
                dones.view(num_processes, 1),
                timesteps.view(num_processes, 1)
            )
        rollouts.after_update()

    # Close the tensorboard writer and environments
    trainingboard_writer.close()
    envs.close()

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python evaluate.py <config_file> <selected_device>"
    config_file = sys.argv[1]

    selected_device = 0
    if len(sys.argv) == 3:
        selected_device = sys.argv[2]

    main(config_file, selected_device)