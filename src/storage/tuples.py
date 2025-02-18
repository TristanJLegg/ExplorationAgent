from typing import NamedTuple

import torch

class BatchData(NamedTuple):  
    observations_batch: list[torch.Tensor]
    recurrent_hidden_states_batch: torch.Tensor
    actions_batch: torch.Tensor
    action_log_probs_batch: torch.Tensor
    value_preds_batch: torch.Tensor
    return_batch: torch.Tensor
    dones_batch: torch.Tensor
    adv_targ: torch.Tensor