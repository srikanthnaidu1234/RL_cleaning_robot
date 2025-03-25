import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-8

class SquashedGaussianPolicy(nn.Module):
    def __init__(self, grid_size_x, grid_size_y, action_dim, hidden_dim=256):
        super().__init__()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        
        # Process coverage grid
        self.coverage_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        
        # Process walls grid
        self.walls_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        
        # Process position (x, y, theta)
        self.position_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 256 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, coverage, walls, position):
        # Process each input
        cov_features = self.coverage_net(coverage)
        wall_features = self.walls_net(walls)
        pos_features = self.position_net(position)
        
        # Combine features
        combined = torch.cat([cov_features, wall_features, pos_features], dim=-1)
        x = self.shared_net(combined)
        
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

class QNetwork(nn.Module):
    def __init__(self, grid_size_x, grid_size_y, action_dim, hidden_dim=256):
        super().__init__()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        
        # Process coverage grid
        self.coverage_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        
        # Process walls grid
        self.walls_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        
        # Process position (x, y, theta)
        self.position_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Process action
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU()
        )
        
        # Shared network
        self.q_net = nn.Sequential(
            nn.Linear(256 + 256 + 128 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, coverage, walls, position, action):
        # Process each input
        cov_features = self.coverage_net(coverage)
        wall_features = self.walls_net(walls)
        pos_features = self.position_net(position)
        act_features = self.action_net(action)
        
        # Combine features
        combined = torch.cat([cov_features, wall_features, pos_features, act_features], dim=-1)
        return self.q_net(combined)

class SAC:
    def __init__(self, env, hidden_dim=256, device='cpu'):
        self.env = env
        self.device = device
        self.action_range = [env.action_space.low, env.action_space.high]
        
        # Networks
        self.policy = SquashedGaussianPolicy(
            env.size_x, env.size_y, 
            env.action_space.shape[0], 
            hidden_dim
        ).to(device)
        
        self.q1 = QNetwork(
            env.size_x, env.size_y,
            env.action_space.shape[0],
            hidden_dim
        ).to(device)
        
        self.q2 = QNetwork(
            env.size_x, env.size_y,
            env.action_space.shape[0],
            hidden_dim
        ).to(device)
        
        # Target networks
        self.q1_target = QNetwork(
            env.size_x, env.size_y,
            env.action_space.shape[0],
            hidden_dim
        ).to(device)
        
        self.q2_target = QNetwork(
            env.size_x, env.size_y,
            env.action_space.shape[0],
            hidden_dim
        ).to(device)
        
        # Initialize targets
        self.hard_update(self.q1_target, self.q1)
        self.hard_update(self.q2_target, self.q2)

        # Optimizers
        self.policy_optim = Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optim = Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = Adam(self.q2.parameters(), lr=3e-4)

        # Temperature
        self.alpha = 0.2
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = Adam([self.log_alpha], lr=3e-4)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            # Convert observations to tensors
            coverage = torch.FloatTensor(obs["coverage"]).view(1, 1, self.env.size_y, self.env.size_x).to(self.device)
            walls = torch.FloatTensor(obs["walls"]).view(1, 1, self.env.size_y, self.env.size_x).to(self.device)
            position = torch.FloatTensor(obs["position"]).view(1, -1).to(self.device)
            
            # Get action from policy
            mean, log_std = self.policy(coverage, walls, position)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                z = normal.rsample()
                action = torch.tanh(z)
            
            # Rescale and return
            action = action.cpu().numpy()[0]
            return self.rescale_action(action)

    def rescale_action(self, action):
        return (action + 1) * (self.action_range[1] - self.action_range[0])/2 + self.action_range[0]

    def update_parameters(self, batch, tau=0.005):
        # Unpack batch
        coverages, walls, positions, actions, rewards, next_coverages, next_walls, next_positions, dones = batch
        
        # Convert to tensors
        coverages = torch.FloatTensor(coverages).view(-1, 1, self.env.size_y, self.env.size_x).to(self.device)
        walls = torch.FloatTensor(walls).view(-1, 1, self.env.size_y, self.env.size_x).to(self.device)
        positions = torch.FloatTensor(positions).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_coverages = torch.FloatTensor(next_coverages).view(-1, 1, self.env.size_y, self.env.size_x).to(self.device)
        next_walls = torch.FloatTensor(next_walls).view(-1, 1, self.env.size_y, self.env.size_x).to(self.device)
        next_positions = torch.FloatTensor(next_positions).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        # Q-function update
        with torch.no_grad():
            next_mean, next_log_std = self.policy(next_coverages, next_walls, next_positions)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_z = next_normal.rsample()
            next_actions = torch.tanh(next_z)

            q1_next = self.q1_target(next_coverages, next_walls, next_positions, next_actions)
            q2_next = self.q2_target(next_coverages, next_walls, next_positions, next_actions)
            q_next = torch.min(q1_next, q2_next)

            target_q = rewards + (1 - dones) * 0.99 * (q_next - self.alpha * next_normal.log_prob(next_z).sum(-1, keepdim=True))

        # Current Q estimates
        q1 = self.q1(coverages, walls, positions, actions)
        q2 = self.q2(coverages, walls, positions, actions)

        # Q losses
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        # Update Q networks
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Policy update
        mean, log_std = self.policy(coverages, walls, positions)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        actions_pi = torch.tanh(z)

        q1_pi = self.q1(coverages, walls, positions, actions_pi)
        q2_pi = self.q2(coverages, walls, positions, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * normal.log_prob(z).sum(-1, keepdim=True) - q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (normal.log_prob(z).sum(-1) + self.target_entropy).mean())

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        self.soft_update(self.q1_target, self.q1, tau)
        self.soft_update(self.q2_target, self.q2, tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()
