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
        steps_already_taken
    )

    # Initialize the rollouts storage
    rollouts = Storage(rollout_size, num_processes, network.hidden_state_size, [image_shape, large_map_shape, small_map_shape])
    # rollouts.to(device)

    # Initialize the PPO algorithm
    ppo = PPO(network, hyperparameters, optimizer)

    # Reset the environment
    observation = envs.reset()

    # Begin storage observation with initial observation
    if environment_parameters.observation_type == "RGBD":
        rollouts.observations[0].copy_(observation['image'])
        rollouts.observations[1].copy_(observation['large_map'])
        rollouts.observations[2].copy_(observation['small_map'])
    elif environment_parameters.observation_type == "RGB":
        rollouts.observations[0].copy_(observation['image'])

    # Episode rewards and coverage
    episode_rewards = torch.zeros(num_processes, 1)

    # Start training
    print("Starting training...")
    num_updates = num_steps // rollout_size
    for update in range(num_updates):
        for step in range(0, rollout_size):
            # Determine actions using the current policy
            with torch.no_grad():
                if environment_parameters.observation_type == "RGBD":
                    network_action_observations = [
                        rollouts.observations[0][step].to(device),
                        rollouts.observations[1][step].to(device),
                        rollouts.observations[2][step].to(device)
                    ]
                elif environment_parameters.observation_type == "RGB":
                    network_action_observations = [rollouts.observations[0][step].to(device)]

                actions, action_log_probabilities, value_predictions, recurrent_hidden_states = \
                    network.action(
                        network_action_observations,
                        rollouts.recurrent_hidden_states[step].to(device),
                        rollouts.dones[step].to(device)
                    )
                value_predictions = value_predictions.squeeze(1)

            # Take actions in the environments
            observations, rewards, dones, infos = envs.step(actions)
            
            # Update episode rewards and coverage
            episode_rewards += rewards

            # Logging
            trainingboard_writer.episode_update(
                episode_rewards, 
                infos,
                dones, 
                steps_already_taken + (update * rollout_size) + step + 1
            )

            # Reset environments that are done
            done_environments_indices = [i for i in range(num_processes) if dones[i]]
            envs.env_method("reset", indices=done_environments_indices)
            episode_rewards[done_environments_indices] = 0

            # Insert experiences into the rollouts storage
            rollouts.insert(
                list(observations.values()),
                recurrent_hidden_states.view(num_processes, -1), 
                actions.view(num_processes, 1),
                action_log_probabilities.view(num_processes, 1), 
                value_predictions.view(num_processes, 1), 
                rewards.view(num_processes, 1), 
                dones.view(num_processes, 1)
            )

        # Calculate the value prediction for the last observation
        if environment_parameters.observation_type == "RGBD":
            next_value_prediction_observation = [
                rollouts.observations[0][-1].to(device),
                rollouts.observations[1][-1].to(device),
                rollouts.observations[2][-1].to(device)
            ]
        elif environment_parameters.observation_type == "RGB":
            next_value_prediction_observation = [rollouts.observations[0][-1].to(device)]
        with torch.no_grad():
            next_value_prediction = network.get_value(
                next_value_prediction_observation,
                rollouts.recurrent_hidden_states[-1].to(device), 
                rollouts.dones[-1].to(device)
            ).detach()
            
        # Calculate the returns
        rollouts.compute_returns(next_value_prediction, ppo.gamma)

        # Perform the PPO update
        value_loss_per_epoch, action_loss_per_epoch, dist_entropy_per_epoch = ppo.update(rollouts, num_mini_batch)
        rollouts.after_update()

        # Logging
        trainingboard_writer.model_update(
            value_loss_per_epoch, 
            action_loss_per_epoch, 
            dist_entropy_per_epoch, 
            steps_already_taken + (update + 1) * rollout_size
        )

        # Save the final model
        if (update + 1) == num_updates:
            save_model(
                network, 
                optimizer, 
                ((steps_already_taken) + ((update + 1) * rollout_size)),
                environment_parameters.seed, 
                map_rooms, 
                parameters,
                hyperparameters,
                environment_parameters,
                f"models/{training_date}-{training_comment}-steps({steps_already_taken + (update + 1) * rollout_size})-final.pth"
            )
        # Save the model at intervals   
        elif (update + 1) % model_save_interval == 0:
            save_model(
                network,
                optimizer,
                ((steps_already_taken) + ((update + 1) * rollout_size)),
                environment_parameters.seed, 
                map_rooms,
                parameters,
                hyperparameters,
                environment_parameters,
                f"models/{training_date}-{training_comment}-steps({steps_already_taken + (update + 1) * rollout_size}).pth"
            )

        if (update + 1) % console_update_interval == 0 or update == 0 or update == num_updates - 1:
            print(f"Update: {update + 1}/{num_updates} complete at {datetime.datetime.now().strftime('%H:%M:%S')}")

    # Close the tensorboard writer and environments
    trainingboard_writer.close()
    envs.close()

if __name__ == "__main__":
    assert len(sys.argv) > 1, "Usage: python train.py <config_file> <selected_device>"
    config_file = sys.argv[1]

    selected_device = 0
    if len(sys.argv) == 3:
        selected_device = sys.argv[2]
        
    main(config_file, selected_device)