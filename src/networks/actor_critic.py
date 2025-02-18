import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(
            self,
            observation_shapes: list[tuple],
            action_space: int,
            hidden_state_size: int = 512,
            sequential_model: str = 'gru'
        ) -> None:
        super(ActorCritic, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.sequential_model = sequential_model

        num_outputs = action_space

        self.models = nn.ModuleList([])
        convolutional_output_sizes = []
        for observation_shape in observation_shapes:
            cnn = nn.Sequential(
                nn.Conv2d(observation_shape[0], 32, 8, stride=4),   
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )

            self.models.append(cnn)
            convolutional_output_sizes.append(self._convolutional_output_size(cnn, observation_shape))

        convolutional_output_size = sum(convolutional_output_sizes)

        self.merge = nn.Sequential(
            nn.Linear(convolutional_output_size, hidden_state_size),
            nn.ReLU(),
        )

        # self.gru = nn.GRU(recurrent_hidden_state_size, recurrent_hidden_state_size)
        if sequential_model == 'gru':
            self.gru = nn.GRU(hidden_state_size, hidden_state_size)
        elif sequential_model == 'decision_transformer':
            self.decision_transformer_encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)

        self.critic = nn.Linear(hidden_state_size, 1)
        self.actor = nn.Linear(hidden_state_size, num_outputs)

        self.train()

    def to(self, device: torch.device) -> nn.Module:
        self = super(ActorCritic, self).to(device)
        self.device = device

        return self

    def forward(self, observations: list[torch.Tensor], hidden_state: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor]:
        # Should I be dividing by 255.0 here to set range to [0, 1]?
        cnn_out = torch.cat([model(observation) for model, observation in zip(self.models, observations)], dim=-1)
        sequential_in = self.merge(cnn_out)

        if self.sequential_model == 'gru':
            sequential_out, hxs = self._forward_gru(sequential_in, hidden_state, dones)
        elif self.sequential_model == 'decision_transformer':
            src_mask = torch.nn.Transformer.generate_square_subsequent_mask(sequential_in.size(1))
            sequential_out = self.decision_transformer_encoder(sequential_in, src_mask=src_mask, is_causal=True)
            hxs = hidden_state
        else:
            sequential_out = sequential_in
            hxs = hidden_state

        return self.critic(sequential_out), self.actor(sequential_out), hxs
    
    def action(self, observations: dict[str, torch.Tensor], hxs: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor]:
        values, pi, hxs = self.forward(observations, hxs, dones)

        distribution = torch.distributions.Categorical(logits=pi)

        actions = distribution.sample()
        action_log_probabilities = distribution.log_prob(actions)

        return actions, action_log_probabilities, values, hxs
    
    def evaluate_actions(self, observations: list[torch.Tensor], hxs: torch.Tensor, dones: torch.Tensor, action: torch.tensor) -> tuple[torch.Tensor]:
        values, pi, hxs = self.forward(observations, hxs, dones)

        distribution = torch.distributions.Categorical(logits=pi)

        actions = action.view(1, -1)
        action_log_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        return action_log_probabilities, values, entropy, hxs
    
    def get_value(self, observations: dict[str, torch.Tensor], hxs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        value, _, _ = self.forward(observations, hxs, dones)

        return value

    def _convolutional_output_size(
        self,
        convolutional_layer: nn.Sequential,
        observation_shape: tuple,
    ) -> int:
        return convolutional_layer(torch.zeros(1, *observation_shape)).view(1, -1).size(1)
    
    def _forward_gru(self, x: torch.Tensor, hxs: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor]:
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * (1 - dones)).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))
            dones = dones.view(T, N)

            has_dones= ((dones[1:] == 1).any(dim=-1).nonzero().squeeze().cpu())

            if has_dones.dim() == 0:
                has_dones = [has_dones.item() + 1]
            else:
                has_dones = (has_dones + 1).numpy().tolist()

            has_dones = [0] + has_dones + [T]

            outputs = []
            for i in range(len(has_dones) - 1):
                a, b = has_dones[i], has_dones[i + 1]
                hx = hxs.view(1, N, -1)
                x_ = x[a:b]
                rnn_scores, hxs = self.gru(x_, hx * (1 - dones[a - 1].view(1, -1, 1)))
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
    
    def _generate_causal_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()