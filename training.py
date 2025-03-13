from environment import ContinuousVacuumCleanerEnv
from ppo import PPO
import torch

def main():
    env = ContinuousVacuumCleanerEnv(size=5.0, max_steps=2000)
    agent = PPO(env)

    max_episodes = 1000
    update_interval = 500
    best_coverage = 0
    size_increments = [(200, 7.0), (400, 8.0), (600, 9.0), (800, 10.0)]
    render_enabled = True  # Flag to control rendering

    try:
        for episode in range(max_episodes):
            current_size = 5.0
            for threshold, size in size_increments:
                if episode >= threshold:
                    current_size = size
            env.size = current_size
            env.cell_size = env.size / env.resolution
            env.max_steps = int(2000 * (current_size/5.0))

            obs = env.reset()
            done = False
            total_reward = 0
            rollout = []

            while not done:
                # Render the environment
                if render_enabled:
                    env.render()

                # Preprocess observations
                coverage = obs['coverage'].reshape(1, 1, 50, 50)
                position = obs['position'].reshape(1, -1)

                # Get action
                action, log_prob, value = agent.act(coverage, position)
                next_obs, reward, done, info = env.step(action)

                # Store experience
                rollout.append({
                    'coverage': coverage.squeeze(),
                    'position': position.squeeze(),
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value
                })

                total_reward += reward
                obs = next_obs

                # Update if needed
                if len(rollout) >= update_interval:
                    agent.update(rollout)
                    rollout = []

            # Final update
            if len(rollout) > 0:
                agent.update(rollout)

            # Save best model
            current_cov = info['coverage_percentage']
            if current_cov > best_coverage:
                best_coverage = current_cov
                torch.save(agent.policy.state_dict(), f"best_model_{best_coverage:.2f}.pth")

            print(f"Ep {episode} | Coverage: {current_cov:.2%} | "
                f"Size: {current_size:.1f} | Reward: {total_reward:.1f}")

    except KeyboardInterrupt:
        print("Training stopped by user")

    # Save final model and close
    torch.save(agent.policy.state_dict(), "final_model.pth")
    env.close()

if __name__ == "__main__":
    main()
