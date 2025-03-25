import torch
import numpy as np
import time
from environment import GridMazeVacuumCleanerEnv
from sac import SAC  # Make sure this imports your modified SAC implementation

def evaluate(model_path="sac_final_model.pth", episodes=3, render=True):
    """
    Evaluate the trained SAC agent on the GridMazeVacuumCleanerEnv.
    
    Args:
        model_path: Path to the saved model weights
        episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    # Initialize environment
    env = GridMazeVacuumCleanerEnv(size_x=25, size_y=15, max_steps=1000)
    
    # Initialize SAC agent
    agent = SAC(env)
    
    # Load trained model
    agent.policy.load_state_dict(torch.load(model_path))
    agent.policy.eval()  # Set to evaluation mode
    
    # Evaluation statistics
    ep_rewards = []
    ep_coverages = []
    ep_lengths = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        while not done:
            # Get action from policy (deterministic for evaluation)
            action = agent.get_action(obs, deterministic=True)
            
            # Step the environment
            next_obs, reward, done, info = env.step(action)
            
            # Render if enabled
            if render:
                env.render()
                time.sleep(0.02)  # Slow down for visualization
            
            total_reward += reward
            steps += 1
            obs = next_obs
        
        # Calculate episode statistics
        ep_time = time.time() - start_time
        coverage = info['coverage_percentage']
        
        # Save statistics
        ep_rewards.append(total_reward)
        ep_coverages.append(coverage)
        ep_lengths.append(steps)
        
        # Print episode summary
        print(f"Evaluation Episode {ep} | "
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
    evaluate(
        model_path="sac_final_model.pth",  # Change to your model path
        episodes=3,
        render=True
    )