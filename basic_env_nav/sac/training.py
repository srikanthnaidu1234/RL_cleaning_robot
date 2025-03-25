"""SAC implementation for continuous vacuum cleaner environment."""
from __future__ import annotations

import random
from collections import deque
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from environment import ContinuousVacuumCleanerEnv


class Transition(NamedTuple):
    """Experience transition tuple."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    """Soft Actor-Critic agent implementation."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_range: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_range = action_range

        # Networks
        self.actor = self._create_actor(state_dim, action_dim).to(self.device)
        self.critic1 = self._create_critic(state_dim, action_dim).to(self.device)
        self.critic2 = self._create_critic(state_dim, action_dim).to(self.device)

        # Target networks
        self.target_critic1 = self._create_critic(state_dim, action_dim).to(
            self.device,
        )
        self.target_critic2 = self._create_critic(state_dim, action_dim).to(
            self.device,
        )
        self.hard_update(self.target_critic1, self.critic1)
        self.hard_update(self.target_critic2, self.critic2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # Temperature parameter
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def _create_actor(self, state_dim: int, action_dim: int) -> nn.Sequential:
        """Create actor network."""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2),  # Mean and log_std
        )

    def _create_critic(self, state_dim: int, action_dim: int) -> nn.Sequential:
        """Create critic network."""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def hard_update(self, target: nn.Module, source: nn.Module) -> None:
        """Hard update target networks."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(
        self,
        target: nn.Module,
        source: nn.Module,
        tau: float = 0.005,
    ) -> None:
        """Soft update target networks."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau,
            )

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get action from policy."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.actor(state_tensor).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
        else:
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)

        # Rescale to environment action range
        action_np = action.cpu().detach().numpy()
        action_range = self.action_range[1] - self.action_range[0]
        return (action_np + 1) * action_range / 2 + self.action_range[0]

    def update(self, batch: list[Transition], tau: float = 0.005) -> tuple[float, ...]:
        """Update agent parameters."""
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in batch])).to(
            self.device,
        )
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(-1).to(
            self.device,
        )
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(
            self.device,
        )
        dones = torch.FloatTensor([t.done for t in batch]).unsqueeze(-1).to(
            self.device,
        )

        # Critic update
        with torch.no_grad():
            next_actions = self.get_action(next_states.cpu().numpy())
            next_actions_tensor = torch.FloatTensor(next_actions).to(self.device)
            target_q1 = self.target_critic1(
                torch.cat([next_states, next_actions_tensor], 1),
            )
            target_q2 = self.target_critic2(
                torch.cat([next_states, next_actions_tensor], 1),
            )
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * 0.99 * target_q

        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        mean, log_std = self.actor(states).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        actions_pi = torch.tanh(z)

        q1_pi = self.critic1(torch.cat([states, actions_pi], 1))
        q2_pi = self.critic2(torch.cat([states, actions_pi], 1))
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * normal.log_prob(z).sum(-1, keepdim=True) - q_pi)
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature update (FIXED)
        alpha_loss = -(self.log_alpha *
                    (normal.log_prob(z).sum(-1).detach() +  # Critical fix here
                    self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        self.soft_update(self.target_critic1, self.critic1, tau)
        self.soft_update(self.target_critic2, self.critic2, tau)

        return (
            critic1_loss.item(),
            critic2_loss.item(),
            actor_loss.item(),
            alpha_loss.item(),
        )

def preprocess_observation(obs: dict[str, Any]) -> np.ndarray:
    """Flatten observation components."""
    return np.concatenate([obs["coverage"], obs["position"]])

def train() -> None:
    """Main training loop."""
    env = ContinuousVacuumCleanerEnv(size=5.0)
    state_dim = 50 * 50 + 3
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)
    agent = SACAgent(state_dim, action_dim, action_range)
    buffer = ReplayBuffer(100000)
    max_episodes = 1000
    batch_size = 256
    print_interval = 10
    render_enabled = True  # Add rendering flag
    try:
        for episode in range(max_episodes):
            obs = env.reset()
            state = preprocess_observation(obs)
            total_reward = 0.0
            done = False
            while not done:
                action = agent.get_action(state)
                next_obs, reward, done, _ = env.step(action)
                next_state = preprocess_observation(next_obs)
                buffer.push(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if render_enabled:  # Add rendering condition
                    env.render()
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    agent.update(batch)
            if episode % print_interval == 0:
                print(
                    f"Episode {episode} | "
                    f"Reward: {total_reward:.1f} | "
                    f"Alpha: {agent.alpha.item():.3f}",
                )
    finally:
        torch.save(agent.actor.state_dict(), "sac_cleaner_actor.pth")
        env.close()

if __name__ == "__main__":
    train()
