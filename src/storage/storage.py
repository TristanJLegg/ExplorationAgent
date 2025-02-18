from typing import Iterator

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from src.storage.tuples import BatchData

class Storage():
    def __init__(
            self,
            size: int,
            num_processes: int,
            recurrent_hidden_state_size: int,
            observation_shapes: list[tuple]
        ) -> None:
        self.size = size
        self.step = 0
        self.observations = [torch.zeros(size + 1, num_processes, *shape) for shape in observation_shapes]
        self.recurrent_hidden_states = torch.zeros(size + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(size, num_processes, 1)
        self.value_predictions = torch.zeros(size + 1, num_processes, 1)
        self.dones = torch.zeros(size, num_processes, 1)
        self.returns = torch.zeros(size + 1, num_processes, 1)
        self.actions = torch.zeros(size, num_processes, 1)
        self.action_log_probs = torch.zeros(size, num_processes, 1)

    def to(self, device) -> None:
        for i in range(len(self.observations)):
            self.observations[i] = self.observations[i].to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.value_predictions = self.value_predictions.to(device)

    def insert(
            self, 
            observation: list[torch.Tensor],
            recurrent_hidden_states: torch.Tensor, 
            action: torch.Tensor, 
            action_log_prob: torch.Tensor, 
            value_prediction: torch.Tensor, 
            reward: torch.Tensor, 
            done: torch.Tensor
        ) -> None:
        [self.observations[i][self.step + 1].copy_(observation[i]) for i in range(len(observation))]
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_predictions[self.step].copy_(value_prediction)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)

        self.step = (self.step + 1) % self.size

    def after_update(self):
        [self.observations[i][0].copy_(self.observations[i][-1]) for i in range(len(self.observations))]
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])

    def compute_returns(self, predicted_reward: torch.Tensor, gamma: float) -> None:
        self.returns[-1] = predicted_reward

        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * (1 - self.dones[step]) + self.rewards[step]

    def feed_forward_generator(
            self,
            advantages: torch.Tensor,
            num_mini_batch=None,
            mini_batch_size=None
        ) -> Iterator[BatchData]:
        num_steps, num_processes = self.rewards.size()[0:2]
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
            observations_batch = [[] for _ in range(len(self.observations))]
            for i in range(len(self.observations)):
                observations_batch[i] = self.observations[i][:-1].view(-1, *self.observations[i].size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_predictions[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            dones_batch = self.dones.view(-1, 1)[indices]
            action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield BatchData(
                observations_batch, recurrent_hidden_states_batch, 
                actions_batch, action_log_probs_batch, value_preds_batch, 
                return_batch, dones_batch, adv_targ
            )

    def recurrent_data_generator(
            self, 
            advantages: torch.Tensor, 
            num_mini_batch: int
        ) -> Iterator[BatchData]:
        num_processes = self.rewards.size(1)

        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        
        num_envs_per_batch = num_processes // num_mini_batch

        # Randomly permute the indices
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = [[] for _ in range(self.observations)]
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            dones_batch = []
            action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                [observations_batch[i].append(self.observations[i][:, ind]) for i in range(len(self.observations))]
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_predictions[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                dones_batch.append(self.dones[:, ind])
                action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.size, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            for i in range(len(observations_batch)):
                observations_batch[i] = torch.stack(observations_batch[i], 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            dones_batch = torch.stack(dones_batch, 1)
            action_log_probs_batch = torch.stack(
                action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for i in range(len(observations_batch)):
                observations_batch[i] = observations_batch[i].view(T * N, *observations_batch[i].size()[2:])
            actions_batch = actions_batch.view(T * N, -1)
            value_preds_batch = value_preds_batch.view(T * N, -1)
            return_batch = return_batch.view(T * N, -1)
            dones_batch = dones_batch.view(T * N, -1)
            action_log_probs_batch = action_log_probs_batch.view(T * N, -1)
            adv_targ = adv_targ.view(T * N, -1)

            yield BatchData(
                observations_batch, recurrent_hidden_states_batch, 
                actions_batch, action_log_probs_batch, value_preds_batch, 
                return_batch, dones_batch, adv_targ
            )