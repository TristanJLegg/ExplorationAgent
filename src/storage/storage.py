from typing import Iterator

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from src.storage.tuples import BatchData

class Storage():
    def __init__(
            self,
            size: int,
            num_processes: int,
            observation_shapes: list[tuple],
            context_length = 1,
        ) -> None:
        self.size = size
        self.num_processes = num_processes
        self.context_length = context_length
        self.step = 0

        self.observations = [torch.zeros(size + context_length, num_processes, *shape) for shape in observation_shapes]
        self.rewards = torch.zeros(size + context_length, num_processes, 1)
        self.value_predictions = torch.zeros(size + 1, num_processes, 1)
        self.dones = torch.zeros(size + context_length, num_processes, 1)
        self.returns = torch.zeros(size + 1, num_processes, 1)
        self.actions = torch.zeros(size + context_length, num_processes, 1)
        self.action_log_probs = torch.zeros(size, num_processes, 1)
        self.timesteps = torch.zeros(size + context_length, num_processes, 1)
        self.episode_rewards = torch.zeros(size + context_length, num_processes, 1)

    def to(self, device) -> None:
        self.device = device

        for i in range(len(self.observations)):
            self.observations[i] = self.observations[i].to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.value_predictions = self.value_predictions.to(device)
        self.timesteps = self.timesteps.to(device)
        self.episode_rewards = self.episode_rewards.to(device)

    def insert(
            self, 
            observation: list[torch.Tensor],
            action: torch.Tensor, 
            action_log_prob: torch.Tensor, 
            value_prediction: torch.Tensor, 
            reward: torch.Tensor, 
            done: torch.Tensor,
            timestep: torch.Tensor,
        ) -> None:
        [self.observations[i][self.step + self.context_length].copy_(observation[i]) for i in range(len(observation))]
        self.actions[self.step + self.context_length].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_predictions[self.step].copy_(value_prediction)
        self.rewards[self.step + self.context_length].copy_(reward)
        self.dones[self.step + self.context_length].copy_(done)
        self.timesteps[self.step + self.context_length].copy_(timestep)

        self.episode_rewards[self.step + self.context_length] = (self.episode_rewards[self.step + self.context_length - 1] + reward) * (1 - done)
        self.step = (self.step + 1) % self.size

    def after_update(self):
        for i in range(self.context_length):
            for j in range(len(self.observations)):
                self.observations[j][i].copy_(self.observations[j][-self.context_length + i])
            self.actions[i].copy_(self.actions[-self.context_length + i])
            self.action_log_probs[i].copy_(self.action_log_probs[-self.context_length + i])
            self.value_predictions[i].copy_(self.value_predictions[-self.context_length + i])
            self.rewards[i].copy_(self.rewards[-self.context_length + i])
            self.timesteps[i].copy_(self.timesteps[-self.context_length + i])
            self.episode_rewards[i].copy_(self.episode_rewards[-self.context_length + i])

    def compute_returns(self, predicted_reward: torch.Tensor, gamma: float) -> None:
        self.returns[-1] = predicted_reward

        for step in reversed(range(self.size)):
            self.returns[step] = self.returns[step + 1] * gamma * (1 - self.dones[step + self.context_length]) + self.rewards[step + self.context_length]

    def get_context_window(self, step, max_reward=1) -> torch.Tensor:
        step = step + self.context_length

        observations = []
        for obs in self.observations:
            observations.append(torch.zeros(self.observations[0].size(1), self.context_length, *obs.shape[2:]))
        actions = torch.zeros(self.actions.size(1), self.context_length, self.actions.size(2))
        returns_to_go = torch.zeros(self.returns.size(1), self.context_length, self.returns.size(2))
        timesteps = self.timesteps[step - 1, :].unsqueeze(1)
        dones = torch.zeros(self.dones.size(1), self.context_length, self.dones.size(2))

        temp_start = max(0, step - self.context_length)
        temp_end = step 

        start = self.context_length - (temp_end - temp_start)
        end = self.context_length

        for i in range(len(self.observations)):
            obs = self.observations[i][temp_start:temp_end]
            obs = obs.permute(1, 0, 2, 3, 4)
            observations[i][:,start:end] = obs
            observations[i] = observations[i].clone().detach()

        returns_to_go = max_reward - self.episode_rewards[temp_start:temp_end].permute(1, 0, 2)

        actions[:,start:end] = self.actions[temp_start:temp_end].permute(1, 0, 2)
        dones[:,start:end] = self.dones[temp_start:temp_end].permute(1, 0, 2)
        attention_mask = torch.ones(self.observations[0].size(1), self.context_length)

        # Shorten the context window if there is a done signal and set the attention mask to 0s
        for i in range(self.observations[0].size(1)):
            last_done = (dones[i] == 1).nonzero()
            if last_done.nelement() != 0:
                mask_start = 0
                mask_end = torch.max(last_done) + 1
                attention_mask[i, mask_start:mask_end] = 0

        return observations, actions, returns_to_go, timesteps, attention_mask
    
    def get_batch(self, indices, max_reward=1) -> torch.Tensor:
        tuple_indices = [(index % self.num_processes, (index // self.num_processes) + self.context_length) for index in indices]

        length = len(tuple_indices)

        observations = []
        for obs in self.observations:
            observations.append(torch.zeros(length, self.context_length, *obs.shape[2:]))
        actions = torch.zeros(length, self.context_length, self.actions.size(2))
        returns_to_go = torch.zeros(length, self.context_length, self.returns.size(2))
        timesteps = torch.zeros(length, 1, self.timesteps.size(2), dtype=torch.int)
        attention_masks = torch.zeros(length, self.context_length)

        for i, (process, step) in enumerate(tuple_indices):
            temp_start = max(0, step - self.context_length)
            temp_end = step

            for j in range(len(self.observations)):
                observations[j][i] = self.observations[j][temp_start:temp_end, process]
            actions[i] = self.actions[temp_start:temp_end, process]
            returns_to_go[i] = max_reward - self.episode_rewards[temp_start:temp_end, process]
            timesteps[i] = self.timesteps[step - 1, process].unsqueeze(1)

        return observations, actions, returns_to_go, timesteps, attention_masks

    def feed_forward_generator(
            self,
            num_mini_batch=None,
            mini_batch_size=None
        ) -> Iterator[BatchData]:
        num_steps, num_processes = self.size, self.observations[0].size(1)
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            yield indices