import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
import cv2

from src.storage.storage import Storage
from src.parameters import Hyperparameters

class PPO():
    def __init__(
            self, 
            network: nn.Module, 
            hyperparameters: Hyperparameters,
            optimizer: optim.Optimizer
        ) -> None:
        self._initialize_hyperparameters(hyperparameters)

        self.network = network
        self.optimizer = optimizer

    def to(
            self,
            device: torch.device
    ) -> None:
        self.device = device
    
    def _initialize_hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        self.gamma = hyperparameters.gamma
        self.learning_rate = hyperparameters.learning_rate
        self.ppo_epochs = hyperparameters.ppo_epochs
        self.clip = hyperparameters.clip
        self.value_loss_coef = hyperparameters.value_loss_coef
        self.entropy_coef = hyperparameters.entropy_coef
        self.max_grad_norm = hyperparameters.max_grad_norm
        self.eps = hyperparameters.eps
        self.context_length = hyperparameters.context_window_size

    def update(self, rollouts: Storage, num_mini_batch: int) -> tuple[float, float, float]:
        value_loss_per_epoch = 0
        action_loss_per_epoch = 0
        dist_entropy_per_epoch = 0

        # Seemingly returns values get smaller and smaller
        advantages = rollouts.returns[:-1] - rollouts.value_predictions[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            # This should return indices as we need to access context windows during training
            data = rollouts.feed_forward_generator(num_mini_batch)
 
            for indices in data:
                observations_batch, actions_batch, returns_to_go_batch, timesteps_batch, attention_masks_batch = rollouts.get_batch(indices)
                # This needs to be truncated to fit the rollout size
                chosen_actions = rollouts.actions[rollouts.context_length:].view(-1, 1)[indices].to(self.device)

                # TODO : OH MAYBE IT NEEDS TO BE THE PREVIOUS OBS? to the return etc

                evaluate_actions_observations_batch = [
                    observations_batch[0].to(self.device),
                    observations_batch[1].to(self.device),
                    observations_batch[2].to(self.device),
                ]
                
                action_log_probabilities, values, dist_entropy = self.network.evaluate_actions(
                    evaluate_actions_observations_batch,
                    actions_batch.to(self.device),
                    returns_to_go_batch.to(self.device),
                    timesteps_batch.to(self.device),
                    chosen_actions,
                )
                action_log_probabilities = action_log_probabilities.view(-1, 1)
                
                action_log_probabilities_batch = rollouts.action_log_probs.view(-1, 1)[indices].to(self.device)
                advantage_targets = advantages.view(-1, 1)[indices].to(self.device)
                value_prediction_batch = rollouts.value_predictions[:-1].view(-1, 1)[indices].to(self.device)
                return_batch = rollouts.returns[:-1].view(-1, 1)[indices].to(self.device)
                
                ratio = torch.exp(action_log_probabilities - action_log_probabilities_batch)
                surr1 = ratio * advantage_targets
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage_targets
                action_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss
                value_prediction_clipped = value_prediction_batch + (values - value_prediction_batch).clamp(-self.clip, self.clip)
                value_losses = (return_batch - values).pow(2)
                value_loss_clipped = (value_prediction_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_loss_clipped).mean()

                self.optimizer.zero_grad()
                loss = (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_per_epoch += value_loss.item()
                action_loss_per_epoch += action_loss.item()
                dist_entropy_per_epoch += dist_entropy.item()

        num_updates = self.ppo_epochs * num_mini_batch

        value_loss_per_epoch = (value_loss_per_epoch / num_updates) * self.value_loss_coef
        action_loss_per_epoch = (action_loss_per_epoch / num_updates)
        dist_entropy_per_epoch = (dist_entropy_per_epoch / num_updates) * self.entropy_coef

        return value_loss_per_epoch, action_loss_per_epoch, dist_entropy_per_epoch