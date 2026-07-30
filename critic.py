import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, state_dim=52, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        
        self.value_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = self.backbone(state)
        value = self.value_layer(x)
        return value

if __name__ == "__main__":
    critic = CriticNetwork(state_dim=52)
    dummy_state = torch.randn(1, 52)
    value = critic(dummy_state)
    
    print(f"State sHape: {dummy_state.shape}")
    print(f"Value: {value}")
    print(f"Value shape: {value.shape}")
