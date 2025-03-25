import gym
import torch
import numpy as np
from environment import ContinuousVacuumCleanerEnv
from sac import SAC

def preprocess_state(obs):
    """Flatten observation components (same as in training code)."""
    return np.concatenate([obs['coverage'], obs['position']])

def evaluate():
    env = ContinuousVacuumCleanerEnv(size=10.0, coverage_radius=0.5)
    state_dim = 50*50 + 3
    action_dim = env.action_space.shape[0]
    action_range = [env.action_space.low, env.action_space.high]

    agent = SAC(state_dim, action_dim, action_range)
    agent.actor.load_state_dict(torch.load("sac_cleaner_actor.pth"))  # Changed from policy to actor

    episodes = 3
    for ep in range(episodes):
        obs = env.reset()
        state = preprocess_state(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, deterministic=True)
            obs, reward, done, _ = env.step(action)
            state = preprocess_state(obs)
            total_reward += reward
            env.render()

        print(f"Episode {ep} | Total Reward: {total_reward:.1f} | Coverage: {env.coverage_percentage:.2%}")

    env.close()

if __name__ == "__main__":
    evaluate()
