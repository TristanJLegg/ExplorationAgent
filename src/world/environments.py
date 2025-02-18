import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecEnvWrapper)

from src.parameters import EnvironmentParameters
from src.world.generation import generate_world
from src.world.wrappers import DictionaryObservation, TransposeObservation, TorchEnvironment

def make_vec_envs(environment_type, num_processes, environment_parameters, map_rooms, device) -> TorchEnvironment:
    envs = [
        make_env(environment_type, environment_parameters, map_rooms)
        for _ in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = TorchEnvironment(envs, environment_parameters.observation_type, device)

    return envs

def make_env(environment_type, environment_parameters, map_rooms) -> gym.Env:
    def _thunk():
        env = environment_type(
            environment_parameters, 
            map_rooms,
            obs_height=environment_parameters.obs_height, 
            obs_width=environment_parameters.obs_width, 
            render_mode="rgb_array"
        )
        env = DictionaryObservation(
            env, 
            image_shape=(environment_parameters.obs_height, environment_parameters.obs_width, 3), 
            large_map_shape=(environment_parameters.large_map_size, environment_parameters.large_map_size, 3), 
            small_map_shape=(environment_parameters.small_map_size, environment_parameters.small_map_size, 3)
        )
        env = TransposeObservation(env)
        return env
    return _thunk