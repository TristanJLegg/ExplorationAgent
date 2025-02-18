from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

class TrainingBoard:
    def __init__(self, directory: str, num_processes: int, board_name: str = None, steps_already_taken: int = 0) -> None:
        self.num_processes = num_processes

        # Tensorboard Writer
        if board_name:
            self.writer = SummaryWriter(
                log_dir=f"{directory}/{board_name}"
            )
        else: 
            self.writer = SummaryWriter()
        
        # Track episode lengths
        self.last_episode_step = [steps_already_taken for _ in range(num_processes)]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    def episode_update(
        self, 
        rewards: torch.Tensor, 
        infos: list[dict],
        dones: torch.Tensor,
        step: int
        ) -> None:

        # Should there be a combined board for all processes?
        for i in range(self.num_processes):
            if dones[i]:
                self.writer.add_scalar(f"Process({i})/EpisodeRewards", rewards[i], step)
                self.writer.add_scalar(f"Process({i})/EpisodeCoverage", infos[i]['coverage'], step)
                self.writer.add_scalar(f"Process({i})/EpisodeLengths", step - self.last_episode_step[i], step)
                self.writer.add_scalar(f"Process({i})/EpisodeVisitedRooms", infos[i]['visited_rooms'], step)
                self.writer.add_scalar(f"Process({i})/EpisodeVisitedHiddenRooms", infos[i]['visited_hidden_rooms'], step)
                self.last_episode_step[i] = step

    def model_update(
        self,
        value_loss_per_epoch: float,
        action_loss_per_epoch: float,
        dist_entropy_per_epoch: float,
        step: int
    ):
        total_loss = value_loss_per_epoch + action_loss_per_epoch + dist_entropy_per_epoch
        self.writer.add_scalar("Losses/Total_Loss_per_Epoch", total_loss, step)
        self.writer.add_scalar("Losses/Value_Loss_per_Epoch", value_loss_per_epoch, step)
        self.writer.add_scalar("Losses/Action_Loss_per_Epoch", action_loss_per_epoch, step)
        self.writer.add_scalar("Losses/Entrop_Loss_per_Epoch", dist_entropy_per_epoch, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()