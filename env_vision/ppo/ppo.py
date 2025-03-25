import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, grid_size_x=25, grid_size_y=15):
        super().__init__()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        # Process coverage grid (binary grid of covered areas)
        self.coverage_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        # Process walls grid (binary grid of walls)
        self.walls_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        # Process position and orientation (3 values: x, y, theta)
        self.position_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Actor outputs mean for linear and angular velocity
        self.actor = nn.Linear(256, 2)
        # Critic outputs state value
        self.critic = nn.Linear(256, 1)
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, coverage, walls, position):
        # Reshape inputs if needed
        if len(coverage.shape) == 3:
            coverage = coverage.unsqueeze(1)  # Add channel dimension
        if len(walls.shape) == 3:
            walls = walls.unsqueeze(1)
        # Process coverage grid
        cov_features = self.coverage_net(coverage)
        # Process walls grid
        wall_features = self.walls_net(walls)
        # Process position
        pos_features = self.position_net(position)
        # Combine features
        combined = torch.cat([cov_features, wall_features, pos_features], dim=-1)
        shared_out = self.shared_net(combined)
        # Get action distribution
        mean = self.actor(shared_out)
        std = torch.exp(self.log_std).expand_as(mean)
        # Get state value
        value = self.critic(shared_out).squeeze()
        return mean, std, value

class PPO:
    def __init__(self, env):
        self.env = env
        obs_space = env.observation_space
        # Initialize policy network
        self.policy = ActorCritic(
            grid_size_x=env.size_x,
            grid_size_y=env.size_y
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.epsilon_clip = 0.2
        self.epochs = 4
        self.batch_size = 64
        self.max_grad_norm = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def act(self, obs):
        # Convert observations to tensors with correct shapes
        coverage = torch.FloatTensor(obs["coverage"]).view(1, self.env.size_y, self.env.size_x)
        walls = torch.FloatTensor(obs["walls"]).view(1, self.env.size_y, self.env.size_x)
        position = torch.FloatTensor(obs["position"]).view(1, -1)
        
        with torch.no_grad():
            coverage = coverage.to(self.device)
            walls = walls.to(self.device)
            position = position.to(self.device)
            
            mean, std, value = self.policy(coverage, walls, position)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()
            value = value.cpu().item()
            
            # Clip action to environment bounds
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
        return action, log_prob, value

    def update(self, rollout):
        # Convert rollout to tensors with correct shapes
        coverages = torch.FloatTensor(np.array([t["coverage"] for t in rollout])).view(-1, self.env.size_y, self.env.size_x)
        walls = torch.FloatTensor(np.array([t["walls"] for t in rollout])).view(-1, self.env.size_y, self.env.size_x)
        positions = torch.FloatTensor(np.array([t["position"] for t in rollout]))
        actions = torch.FloatTensor(np.array([t["action"] for t in rollout]))
        rewards = torch.FloatTensor(np.array([t["reward"] for t in rollout]))
        dones = torch.FloatTensor(np.array([t["done"] for t in rollout]))
        log_probs = torch.FloatTensor(np.array([t["log_prob"] for t in rollout]))
        values = torch.FloatTensor(np.array([t["value"] for t in rollout]))
        
        # Move to device
        coverages = coverages.to(self.device)
        walls = walls.to(self.device)
        positions = positions.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = log_probs.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        old_values = values.to(self.device)
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Shuffle indices for mini-batch updates
            indices = torch.randperm(len(coverages))
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                cov_batch = coverages[batch_idx]
                wall_batch = walls[batch_idx]
                pos_batch = positions[batch_idx]
                act_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                ret_batch = returns[batch_idx]
                adv_batch = advantages[batch_idx]
                
                # Get new policy outputs
                mean, std, values = self.policy(cov_batch, wall_batch, pos_batch)
                dist = Normal(mean, std)
                
                # Calculate log probs and entropy
                log_probs = dist.log_prob(act_batch).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                # Calculate policy loss
                ratios = torch.exp(log_probs - old_lp_batch)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values, ret_batch)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t] * next_non_terminal
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.lambda_gae * last_advantage
            last_advantage = advantages[t]
            next_value = values[t]
        
        return advantages

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()
