"""SAC implementation for GridMazeVacuumCleaner environment."""
from __future__ import annotations

import random
from collections import deque
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from environment import GridMazeVacuumCleanerEnv
from sac import SAC  # Import the modified SAC implementation

class Transition(NamedTuple):
    """Experience transition tuple for grid environment."""
    coverage: np.ndarray
    walls: np.ndarray
    position: np.ndarray
    action: np.ndarray
    reward: float
    next_coverage: np.ndarray
    next_walls: np.ndarray
    next_position: np.ndarray
    done: bool

class ReplayBuffer:
    """Experience replay buffer for grid environment."""
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        coverage: np.ndarray,
        walls: np.ndarray,
        position: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_coverage: np.ndarray,
        next_walls: np.ndarray,
        next_position: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append(Transition(
            coverage, walls, position,
            action, reward,
            next_coverage, next_walls, next_position,
            done
        ))

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

def train() -> None:
    """Main training loop for GridMazeVacuumCleanerEnv."""
    # Initialize environment
    env = GridMazeVacuumCleanerEnv(size_x=25, size_y=15, max_steps=1000)
    
    # Initialize SAC agent
    agent = SAC(env)
    
    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=100000)
    
    # Training parameters
    max_episodes = 1000
    batch_size = 256
    update_interval = 1  # Update every episode
    print_interval = 10
    render_every = 20  # Render every N episodes
    
    # Training statistics
    ep_rewards = []
    ep_coverages = []
    ep_lengths = []

    try:
        for episode in range(1, max_episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            start_time = time.time()
            
            # Determine if we should render this episode
            render_episode = (episode % render_every == 0) or (episode == 1)
            
            while not done:
                # Get action from policy
                action = agent.get_action(obs)
                
                # Take step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store experience in buffer
                buffer.push(
                    coverage=obs['coverage'].copy(),
                    walls=obs['walls'].copy(),
                    position=obs['position'].copy(),
                    action=action,
                    reward=reward,
                    next_coverage=next_obs['coverage'].copy(),
                    next_walls=next_obs['walls'].copy(),
                    next_position=next_obs['position'].copy(),
                    done=done
                )
                
                # Render if enabled
                if render_episode:
                    env.render()
                    time.sleep(0.02)  # Slow down for visualization
                
                total_reward += reward
                steps += 1
                obs = next_obs
                
                # Update if we've collected enough experience
                if len(buffer) >= batch_size and steps % update_interval == 0:
                    batch = buffer.sample(batch_size)
                    agent.update_parameters(batch)
            
            # Calculate episode statistics
            ep_time = time.time() - start_time
            coverage = info['coverage_percentage']
            
            # Save statistics
            ep_rewards.append(total_reward)
            ep_coverages.append(coverage)
            ep_lengths.append(steps)
            
            # Print episode summary
            if episode % print_interval == 0:
                print(f"Episode {episode:4d} | "
                      f"Coverage: {coverage:.2%} | "
                      f"Steps: {steps:4d}/{env.max_steps} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Time: {ep_time:.2f}s")
            
            # Print moving averages every 50 episodes
            if episode % 50 == 0:
                avg_reward = np.mean(ep_rewards[-50:])
                avg_coverage = np.mean(ep_coverages[-50:])
                avg_length = np.mean(ep_lengths[-50:])
                print(f"\nLast 50 episodes average:")
                print(f"Coverage: {avg_coverage:.2%} | "
                      f"Steps: {avg_length:.1f} | "
                      f"Reward: {avg_reward:.2f}\n")
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    
    finally:
        # Save final model and close environment
        torch.save(agent.policy.state_dict(), "sac_final_model.pth")
        env.close()
        
        # Print final statistics
        if len(ep_rewards) > 0:
            print("\nTraining completed.")
            print(f"Final coverage: {ep_coverages[-1]:.2%}")
            print(f"Average reward: {np.mean(ep_rewards):.2f}")
            print(f"Average coverage: {np.mean(ep_coverages):.2%}")
            print(f"Average episode length: {np.mean(ep_lengths):.1f} steps")

if __name__ == "__main__":
    import time
    train()