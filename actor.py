import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
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

        nn.init.uniform_(self.mean_layer.weight, -0.001, 0.001)
        nn.init.constant_(self.mean_layer.bias[0], 0.0)
        nn.init.constant_(self.mean_layer.bias[1], 1.0)
        nn.init.constant_(self.mean_layer.bias[2], -3.0)

        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))

    def forward(self, state):
        x        = self.backbone(state)
        mean_raw = self.mean_layer(x)

        steer = torch.tanh(mean_raw[:, 0:1])
        accel = torch.sigmoid(mean_raw[:, 1:2])
        brake = torch.sigmoid(mean_raw[:, 2:3])

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
