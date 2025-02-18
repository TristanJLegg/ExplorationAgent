import sys
import datetime
import warnings

import pyglet
pyglet.options["headless"] = True

import torch
import numpy as np

from src.load_and_save import load_model, save_model
from src.algorithms.ppo import PPO
from src.networks.actor_critic import ActorCritic
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

    network = ActorCritic(
        network_observation_shapes, 
        action_space_size,
        hyperparameters.hidden_state_size,
        hyperparameters.sequential_model
    )
    network.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=hyperparameters.learning_rate, eps=hyperparameters.eps)

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
        0
    )

    # Reset the environment
    observations = envs.reset()

    # Start evaluation
    print("Starting evaluation...")

    episode_rewards = torch.zeros(num_processes, 1)
    episode_dones = torch.zeros(num_processes, 1)
    recurrent_hidden_states = torch.zeros(num_processes, network.hidden_state_size)
    dones = torch.zeros(num_processes, 1)
    count = 0
    while not (episode_dones.flatten() == 1.0).all():
        count += 1
        if count % 1000 == 0:
            print(count)
        # Determine actions using the current policy
        with torch.no_grad():
            network_action_observations = [
                observations["image"].to(device),
                observations["large_map"].to(device),
                observations["small_map"].to(device)
            ]

            actions, _, _, recurrent_hidden_states = \
                network.action(
                    network_action_observations,
                    recurrent_hidden_states.to(device),
                    dones.to(device)
                )

        # Take actions in the environments
        observations, rewards, dones, infos = envs.step(actions)
        
        # Update episode rewards and coverage
        episode_rewards += rewards

        # Completion
        for i in range(num_processes):
            if dones[i]:
                episode_dones[i] = 1

        # Logging
        if dones[0]:
            trainingboard_writer.episode_update(
                episode_rewards, 
                infos,
                dones, 
                0
            )

    # Close the tensorboard writer and environments
    trainingboard_writer.close()
    envs.close()

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python train.py <config_file>"
    config_file = sys.argv[1]

    selected_device = 0
    if len(sys.argv) == 3:
        selected_device = sys.argv[2]

    main(config_file, selected_device)