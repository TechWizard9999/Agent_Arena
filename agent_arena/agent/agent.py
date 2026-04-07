from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from agent_arena.agent.dqn import DQN
from agent_arena.agent.replay_buffer import ReplayBuffer
from agent_arena.config import ArenaConfig


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DQNAgent:
    """Deep Q-learning agent with target network and epsilon-greedy exploration."""

    def __init__(self, state_size: int, action_size: int, config: ArenaConfig) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = resolve_device()

        self.policy_net = DQN(state_size, action_size, config.hidden_dim).to(self.device)
        self.target_net = DQN(state_size, action_size, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity)

        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.training_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(dim=1, keepdim=True).values
            target_q = rewards_tensor + (1.0 - dones_tensor) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def pretrain_on_demonstrations(
        self,
        demonstrations: list[tuple[np.ndarray, int]],
        epochs: int,
        batch_size: int,
    ) -> list[float]:
        if not demonstrations:
            return []

        classification_loss = nn.CrossEntropyLoss()
        losses: list[float] = []
        effective_batch_size = max(1, min(batch_size, len(demonstrations)))

        for _ in range(epochs):
            random.shuffle(demonstrations)
            batch_losses: list[float] = []

            for start in range(0, len(demonstrations), effective_batch_size):
                batch = demonstrations[start : start + effective_batch_size]
                states = torch.as_tensor(
                    np.vstack([item[0] for item in batch]),
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    [item[1] for item in batch],
                    dtype=torch.int64,
                    device=self.device,
                )

                logits = self.policy_net(states)
                loss = classification_loss(logits, actions)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_losses.append(float(loss.item()))

            losses.append(float(np.mean(batch_losses)))

        self.update_target_network()
        return losses

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
                "config": self.config.to_dict(),
            },
            target,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["model_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_min))
