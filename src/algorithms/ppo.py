import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym

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
    
    def _initialize_hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        self.gamma = hyperparameters.gamma
        self.learning_rate = hyperparameters.learning_rate
        self.ppo_epochs = hyperparameters.ppo_epochs
        self.clip = hyperparameters.clip
        self.value_loss_coef = hyperparameters.value_loss_coef
        self.entropy_coef = hyperparameters.entropy_coef
        self.max_grad_norm = hyperparameters.max_grad_norm
        self.eps = hyperparameters.eps

    def update(self, rollouts: Storage, num_mini_batch: int) -> tuple[float, float, float]:
        value_loss_per_epoch = 0
        action_loss_per_epoch = 0
        dist_entropy_per_epoch = 0

        advantages = rollouts.returns[:-1] - rollouts.value_predictions[:-1]
        # TODO : Why does stable baselines3 use 1e-8 and not 1e-5?
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            data = rollouts.feed_forward_generator(advantages, num_mini_batch)
 
            for sample in data:
                # Move the sample to the device
                sample = tuple([
                    item.to(self.network.device) if isinstance(item, torch.Tensor) 
                    else [
                        subitem.to(self.network.device) if isinstance(subitem, torch.Tensor) 
                        else subitem
                        for subitem in item
                    ] if isinstance(item, (list, tuple)) 
                    else item
                    for item in sample
                ])

                observation_batch, recurrent_hidden_state_batch, action_batch, \
                action_log_probabilities_batch, value_prediction_batch, return_batch, \
                dones_batch, advantage_targets = sample

                action_log_probabilities, values, dist_entropy, _ = self.network.evaluate_actions(
                    observation_batch, 
                    recurrent_hidden_state_batch, 
                    dones_batch, 
                    action_batch
                )
                
                action_log_probabilities = action_log_probabilities.view(-1, 1)
                
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