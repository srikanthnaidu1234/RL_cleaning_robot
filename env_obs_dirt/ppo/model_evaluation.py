from environment import GridMazeVacuumCleanerEnv
from ppo import PPO
import torch
import numpy as np
import time

def evaluate(env, agent, episodes=3, model_path="final_model.pth", render=True):
    """
    Evaluate the trained PPO agent on the environment.

    Args:
        env: The environment to evaluate in
        agent: The PPO agent
        episodes: Number of evaluation episodes
        model_path: Path to the saved model weights
        render: Whether to render the environment
    """
    # Load the trained model
    agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()  # Set to evaluation mode

    # Evaluation statistics
    ep_rewards = []
    ep_coverages = []
    ep_lengths = []

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()

        while not done:
            if render:
                env.render()
                time.sleep(0.02)  # Slow down for visualization

            # Get deterministic action (use mean instead of sampling)
            with torch.no_grad():
                coverage = torch.FloatTensor(obs["coverage"]).view(1, env.size_y, env.size_x).to(agent.device)
                walls = torch.FloatTensor(obs["walls"]).view(1, env.size_y, env.size_x).to(agent.device)
                position = torch.FloatTensor(obs["position"]).view(1, -1).to(agent.device)

                mean, _, _ = agent.policy(coverage, walls, position)
                action = mean.cpu().numpy()[0]
                action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step the environment
            next_obs, reward, done, info = env.step(action)

            total_reward += reward
            obs = next_obs

        # Calculate episode statistics
        ep_time = time.time() - start_time
        coverage = info['coverage_percentage']
        steps = info['steps']

        # Save statistics
        ep_rewards.append(total_reward)
        ep_coverages.append(coverage)
        ep_lengths.append(steps)

        # Print episode summary
        print(f"Evaluation Episode {episode} | "
            f"Coverage: {coverage:.2%} | "
            f"Steps: {steps:4d}/{env.max_steps} | "
            f"Reward: {total_reward:7.2f} | "
            f"Time: {ep_time:.2f}s")

    # Print evaluation summary
    print("\nEvaluation Complete:")
    print(f"Average Coverage: {np.mean(ep_coverages):.2%}")
    print(f"Average Reward: {np.mean(ep_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(ep_lengths):.1f} steps")

    env.close()

if __name__ == "__main__":
    # Initialize environment and agent
    env = GridMazeVacuumCleanerEnv(size_x=25, size_y=15, max_steps=1000)
    agent = PPO(env)

    # Run evaluation
    evaluate(
        env,
        agent,
        episodes=3,
        model_path="final_model.pth",  # Change to your model path
        render=True
    )
