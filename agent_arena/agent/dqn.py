from __future__ import annotations

import torch
from torch import nn


class DQN(nn.Module):
    """Simple feed-forward Q-network for vector state inputs."""

    def __init__(self, state_size: int, action_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

