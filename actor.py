import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, state_dim=60, action_dim=2, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize bias for acceleration to be positive (approx +1.0 after sigmoid is ~0.73)
        nn.init.constant_(self.mean_layer.bias[1], 1.0)
        
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, state):
        x = self.backbone(state)
        mean_raw = self.mean_layer(x)
        
        # Split and activate: Steering [-1, 1], Acceleration [0, 1]
        steer = torch.tanh(mean_raw[:, 0:1])
        accel = torch.sigmoid(mean_raw[:, 1:2])
        mean = torch.cat([steer, accel], dim=1)
        
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

if __name__ == "__main__":
    actor = ActorNetwork(state_dim=60, action_dim=2)
    dummy_state = torch.randn(1, 60)
    action_dist = actor(dummy_state)
    action = action_dist.sample()
    print(f"State shape: {dummy_state.shape}")
    print(f"Sampled action (Steering, Acceleration): {action}")
    print(f"Action shape: {action.shape}")
