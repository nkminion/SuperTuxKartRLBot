import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """
    3-action PPO actor: Steer [-1,1], Acceleration [0,1], Brake [0,1].

    Architecture:
    - 3 hidden layers of 256 neurons with ELU activations
    - Small weight init on the output layer so initial actions are controlled
      by the carefully chosen biases, not random noise
    - log_std initialized to -0.5 (std ≈ 0.6) so the policy is not too random
      at the start and not completely deterministic either
    """
    
    def __init__(self, state_dim=52, action_dim=3, hidden_dim=256):
        super(ActorNetwork, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)

        # Tiny weights — initial mean is dominated by the biases below
        nn.init.uniform_(self.mean_layer.weight, -0.001, 0.001)

        # Steering: start neutral (tanh(0) = 0)
        nn.init.constant_(self.mean_layer.bias[0], 0.0)

        # Acceleration: start ~0.73 forward (sigmoid(1.0) ≈ 0.73)
        nn.init.constant_(self.mean_layer.bias[1], 1.0)

        # Braking: start almost-never (sigmoid(-3.0) ≈ 0.047)
        nn.init.constant_(self.mean_layer.bias[2], -3.0)

        # log_std = -0.5  →  std ≈ 0.61  (moderate initial exploration)
        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))

    def forward(self, state):
        x        = self.backbone(state)
        mean_raw = self.mean_layer(x)

        # Bounded activations per action head
        steer = torch.tanh(mean_raw[:, 0:1])          # [-1, 1]
        accel = torch.sigmoid(mean_raw[:, 1:2])       # [0,  1]
        brake = torch.sigmoid(mean_raw[:, 2:3])       # [0,  1]  → treated as probability

        mean = torch.cat([steer, accel, brake], dim=1)

        log_std = torch.clamp(self.log_std, min=-4, max=0.5)
        std      = torch.exp(log_std)
        dist     = Normal(mean, std)
        return dist


if __name__ == '__main__':
    actor      = ActorNetwork(state_dim=52, action_dim=3)
    dummy      = torch.randn(1, 52)
    dist       = actor(dummy)
    action     = dist.sample()
    print(f'State shape:  {dummy.shape}')
    print(f'Action (Steer, Accel, Brake): {action}')
    print(f'Action shape: {action.shape}')
    print(f'Mean at init: {dist.mean}')
