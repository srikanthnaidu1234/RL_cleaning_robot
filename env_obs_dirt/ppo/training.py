from environment import GridMazeVacuumCleanerEnv
from ppo import PPO
import torch
import numpy as np
import time

def main():
    # Initialize environment
    env = GridMazeVacuumCleanerEnv(size_x=25, size_y=15, max_steps=1000)
    agent = PPO(env)

    # Training parameters
    max_episodes = 1000
    update_interval = 200  # Update policy every N steps
    best_coverage = 0
    render_every = 20  # Render every N episodes
    save_interval = 50  # Save model every N episodes
    
    # Training statistics
    ep_rewards = []
    ep_coverages = []
    ep_lengths = []

    try:
        for episode in range(1, max_episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0
            rollout = []
            start_time = time.time()
            
            # Determine if we should render this episode
            render_episode = (episode % render_every == 0) or (episode == 1)
            
            while not done:
                if render_episode:
                    env.render()
                    time.sleep(0.02)  # Slow down for visualization
                
                # Get action from policy
                action, log_prob, value = agent.act(obs)
                
                # Take step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store experience with proper shapes
                rollout.append({
                    'coverage': obs['coverage'].reshape(env.size_y, env.size_x),
                    'walls': obs['walls'].reshape(env.size_y, env.size_x),
                    'position': obs['position'],
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value
                })
                
                total_reward += reward
                obs = next_obs
                
                # Update if we've collected enough experience
                if len(rollout) >= update_interval:
                    agent.update(rollout)
                    rollout = []
            
            # Final update with remaining experiences
            if len(rollout) > 0:
                agent.update(rollout)
            
            # Calculate episode statistics
            ep_time = time.time() - start_time
            coverage = info['coverage_percentage']
            steps = info['steps']
            
            # Save statistics
            ep_rewards.append(total_reward)
            ep_coverages.append(coverage)
            ep_lengths.append(steps)
            
            # Save best model
            if coverage > best_coverage:
                best_coverage = coverage
                torch.save(agent.policy.state_dict(), f"best_model_{best_coverage:.2f}.pth")
                print(f"New best model saved with coverage: {best_coverage:.2%}")
            
            # Periodic model saving
            if episode % save_interval == 0:
                torch.save(agent.policy.state_dict(), f"model_ep_{episode}.pth")
            
            # Print episode summary
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
        torch.save(agent.policy.state_dict(), "final_model.pth")
        env.close()
        
        # Print final statistics
        if len(ep_rewards) > 0:
            print("\nTraining completed.")
            print(f"Final coverage: {ep_coverages[-1]:.2%}")
            print(f"Average reward: {np.mean(ep_rewards):.2f}")
            print(f"Average coverage: {np.mean(ep_coverages):.2%}")
            print(f"Average episode length: {np.mean(ep_lengths):.1f} steps")

if __name__ == "__main__":
    main()
