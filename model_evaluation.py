from environment import ContinuousVacuumCleanerEnv
from ppo import PPO
import torch
import numpy as np

def evaluate(env, agent, episodes=3):
    # Load the trained model
    agent.policy.load_state_dict(torch.load("ppo_trained_model.pth"))
    agent.policy.eval()  # Set to evaluation mode

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Preprocess observations
            coverage = obs['coverage'].reshape(1, 1, 50, 50)
            position = obs['position'].reshape(1, -1)

            # Take deterministic action (use mean instead of sampling)
            with torch.no_grad():
                mean, _, _ = agent.policy(
                    torch.FloatTensor(coverage),
                    torch.FloatTensor(position)
                )
            action = mean.numpy().squeeze()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step the environment
            next_obs, reward, done, info = env.step(action)

            # Render
            env.render(mode="human")

            total_reward += reward
            obs = next_obs

        print(f"Evaluation Episode {episode}, Total Reward: {total_reward:.2f}")

    env.close()

# Run evaluation
env = ContinuousVacuumCleanerEnv(size=10.0, resolution=50, coverage_radius=0.5)
agent = PPO(env)
evaluate(env, agent, episodes=3)
