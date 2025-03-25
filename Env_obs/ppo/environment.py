import math
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from matplotlib.collections import PatchCollection

class GridMazeVacuumCleanerEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self, size_x=25, size_y=15, coverage_radius=0.5, max_steps=1000):
        super().__init__()
        self.size_x = size_x
        self.size_y = size_y
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        # Define grid-based maze layout
        self._define_maze_layout()
        self.action_space = spaces.Box(
            low=np.array([-0.5, -np.pi/2]),
            high=np.array([0.5, np.pi/2]),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([0, 0, -np.pi]),
                high=np.array([size_x, size_y, np.pi]),
                dtype=np.float32
            ),
            "coverage": spaces.Box(
                low=0, high=1, shape=(size_x * size_y,), dtype=np.float32
            ),
            "walls": spaces.Box(
                low=0, high=1,
                shape=(size_x * size_y,),
                dtype=np.float32
            )
        })
        self.fig = None
        self.ax = None
        self.coverage_path = []  # To store coverage circles
        self.coverage_patches = []  # For rendering coverage


    def _define_maze_layout(self):
        # Create wall layout with minimal, simple walls
        self.wall_grid = np.zeros((self.size_y, self.size_x), dtype=np.int32)
        # Add vertical walls with wider gaps
        wall_positions = [8, 16]
        for x in wall_positions:
            # Create a wall with a wider gap in the middle
            gap_start = self.size_y // 2 - 3
            gap_height = 6
            for y in range(self.size_y):
                if y < gap_start or y >= gap_start + gap_height:
                    self.wall_grid[y, x] = 1
        # Add a horizontal wall touching the boundaries with a wider gap
        horizontal_wall_y = self.size_y // 2
        gap_start = self.size_x // 2 - 3
        gap_width = 6
        for x in range(self.size_x):
            # Create wider gap in the middle of the wall
            if x < gap_start or x >= gap_start + gap_width:
                self.wall_grid[horizontal_wall_y, x] = 1
        # Start and exit positions
        self.start_pos = np.array([1.0, 1.0])
        self.exit_pos = np.array([self.size_x-2.0, self.size_y-2.0])

    def reset(self):
        self.agent_position = self.start_pos.copy()
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.coverage_grid = np.zeros((self.size_x, self.size_y), dtype=np.float32)
        self.coverage_path = []
        self.coverage_patches = []
        self.steps = 0
        self.coverage_percentage = 0.0
        return self._get_observation()


    def _is_valid_move(self, position):
        """Check if the proposed position is within valid grid and not in a wall"""
        x, y = int(position[0]), int(position[1])
        # Check boundary conditions
        if (x < 0 or x >= self.size_x or
            y < 0 or y >= self.size_y):
            return False
        # Check wall collision
        if self.wall_grid[y, x] == 1:
            return False
        return True


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        lin_vel, ang_vel = action
        # Update orientation
        self.agent_orientation = ((self.agent_orientation + ang_vel * 0.1 + np.pi) 
                                 % (2 * np.pi)) - np.pi
        # Calculate potential new position
        delta_x = lin_vel * math.cos(self.agent_orientation) * 0.1
        delta_y = lin_vel * math.sin(self.agent_orientation) * 0.1
        new_pos = self.agent_position + np.array([delta_x, delta_y])
        # Validate move
        if self._is_valid_move(new_pos):
            self.agent_position = new_pos
            reward = 0.1  # Small positive reward for movement
        else:
            reward = -0.5  # Penalty for invalid move
        # Update coverage
        self._update_coverage()
        self.steps += 1
        done = (self.coverage_percentage >= 0.95 or
                self.steps >= self.max_steps or
                np.linalg.norm(self.agent_position - self.exit_pos) < 1.0)
        return self._get_observation(), reward, done, {
            "coverage_percentage": self.coverage_percentage,
            "steps": self.steps
        }


    def _update_coverage(self):
        # Add current position to coverage path
        self.coverage_path.append((self.agent_position[0],
                                self.agent_position[1],
                                self.coverage_radius))

        # Create a new coverage patch for this position
        coverage_circle = patches.Circle(
            (self.agent_position[0], self.agent_position[1]),
            radius=self.coverage_radius,
            color='blue',
            alpha=0.3
        )
        self.coverage_patches.append(coverage_circle)
        # Update coverage grid (for observation)
        x, y = int(self.agent_position[0]), int(self.agent_position[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size_x and
                    0 <= ny < self.size_y and
                    self.wall_grid[ny, nx] == 0):
                    self.coverage_grid[nx, ny] = 1
        self.coverage_percentage = np.sum(self.coverage_grid) / (self.size_x * self.size_y)


    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.ion()
        self.ax.clear()
        # Set background to white
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        # Draw walls as gray rectangles
        for y in range(self.size_y):
            for x in range(self.size_x):
                if self.wall_grid[y, x] == 1:
                    wall = patches.Rectangle(
                        (x, y), 1, 1,
                        facecolor=[0.5, 0.5, 0.5],
                        edgecolor='none'
                    )
                    self.ax.add_patch(wall)
        # Draw all coverage circles
        for patch in self.coverage_patches:
            self.ax.add_patch(patch)
        # Draw agent as a circle with orientation line
        agent_circle = patches.Circle(
            self.agent_position,
            radius=self.coverage_radius,
            facecolor="red",
            alpha=0.5,
            edgecolor="black"
        )
        self.ax.add_patch(agent_circle)
        # Draw orientation line
        end_x = self.agent_position[0] + self.coverage_radius * 2 * math.cos(self.agent_orientation)
        end_y = self.agent_position[1] + self.coverage_radius * 2 * math.sin(self.agent_orientation)
        self.ax.plot([self.agent_position[0], end_x], [self.agent_position[1], end_y], 
                    color='black',
                    linewidth=2)
        # Info text
        self.ax.text(0.05, 0.95,
                    f"Coverage: {self.coverage_percentage:.2%}\n"
                    f"Steps: {self.steps}/{self.max_steps}",
                    transform=self.ax.transAxes,
                    fontsize=10,
                    color='black',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7))
        self.ax.set_xlim(0, self.size_x)
        self.ax.set_ylim(0, self.size_y)
        self.ax.set_aspect('equal')
        plt.pause(0.01)


    def _get_observation(self):
        return {
            "position": np.array([
                self.agent_position[0],
                self.agent_position[1],
                self.agent_orientation
            ], dtype=np.float32),
            "coverage": self.coverage_grid.flatten(),
            "walls": self.wall_grid.flatten()
        }


def main():
    env = GridMazeVacuumCleanerEnv()
    obs = env.reset()
    for _ in range(1000):  # Run for 1000 steps max
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)  # Add delay to visualize better
        if done:
            print(f"Episode finished! Coverage: {info['coverage_percentage']:.2%}, Steps: {info['steps']}")
            break
    plt.show()  # Keep the final render visible
    env.close()


if __name__ == "__main__":
    main()
