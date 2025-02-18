import torch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

class TorchEnvironment(VecEnvWrapper):
    def __init__(
            self, 
            venv, 
            observation_type, 
            device
        ) -> None:
        super(TorchEnvironment, self).__init__(venv)
        self.observation_type = observation_type
        self.device = device

    def reset(self):
        observations = self.venv.reset()
        if self.observation_type == "RGBD":
            observations = {
                "image": torch.from_numpy(observations["image"]).float().to(self.device),
                "large_map":  torch.from_numpy(observations["large_map"]).float().to(self.device),
                "small_map": torch.from_numpy(observations["small_map"]).float().to(self.device)
            }
        elif self.observation_type == "RGB":
            observations = {
                "image": torch.from_numpy(observations["image"]).float().to(self.device)
            }

        return observations

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()

        if self.observation_type == "RGBD":
            observations = {
                "image": torch.from_numpy(observations["image"]).float().to(self.device),
                "large_map":  torch.from_numpy(observations["large_map"]).float().to(self.device),
                "small_map": torch.from_numpy(observations["small_map"]).float().to(self.device)
            }
        elif self.observation_type == "RGB":
            observations = {
                "image": torch.from_numpy(observations["image"]).float().to(self.device)
            }

        rewards = torch.from_numpy(rewards).unsqueeze(1).float().to(self.device)
        dones = torch.from_numpy(dones).unsqueeze(1).float().to(self.device)

        return observations, rewards, dones, infos
    
class DictionaryObservation(gym.ObservationWrapper):
    def __init__(self, env, image_shape, large_map_shape, small_map_shape) -> None:
        super(DictionaryObservation, self).__init__(env)

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(image_shape[0], image_shape[1], image_shape[2]), 
                dtype=np.uint8
            ),
            "large_map": gym.spaces.Box(
                low=0,
                high=255,
                shape=(large_map_shape[0], large_map_shape[1], large_map_shape[2]),
                dtype=np.uint8
            ),
            "small_map": gym.spaces.Box(
                low=0,
                high=255,
                shape=(small_map_shape[0], small_map_shape[1], small_map_shape[2]),
                dtype=np.uint8
            )
        })

    def observation(self, observation: tuple) -> dict:
        image, large_map, small_map = observation

        observation = {
            "image": image,
            "large_map": large_map,
            "small_map": small_map
        }

        return observation

class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=[2, 0, 1]) -> None:
        super(TransposeObservation, self).__init__(env)

        self.shape = shape
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.observation_space["image"].shape[self.shape[0]], 
                    self.observation_space["image"].shape[self.shape[1]], 
                    self.observation_space["image"].shape[self.shape[2]]
                ),
                dtype=np.uint8
            ),
            "large_map": gym.spaces.Box(
                low=0,
                high=255,
                shape=
                (
                    self.observation_space["large_map"].shape[self.shape[0]], 
                    self.observation_space["large_map"].shape[self.shape[1]], 
                    self.observation_space["large_map"].shape[self.shape[2]]
                ),
                dtype=np.uint8
            ),
            "small_map": gym.spaces.Box(
                low=0,
                high=255,
                shape=
                (
                    self.observation_space["small_map"].shape[self.shape[0]],
                    self.observation_space["small_map"].shape[self.shape[1]], 
                    self.observation_space["small_map"].shape[self.shape[2]]
                ),
                dtype=np.uint8
            )
        })

    def observation(self, observation: tuple) -> np.ndarray:
        observation = {
            "image": observation["image"].transpose(self.shape),
            "large_map": observation["large_map"].transpose(self.shape),
            "small_map": observation["small_map"].transpose(self.shape)
        }

        return observation