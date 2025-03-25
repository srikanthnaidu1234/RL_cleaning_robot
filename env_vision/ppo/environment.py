import math
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class CleaningRobot:
    """Autonomous cleaning robot with vision capabilities"""
    def __init__(self, vision_range=5):
        self.vision_range = vision_range
        self.position = np.array([1.0, 1.0], dtype=np.float32)
        self.orientation = 0.0
        self.coverage_radius = 0.5
        self.path_history = []

    def update_pose(self, new_position, new_orientation):
        """Update robot's position and orientation"""
        self.position = np.clip(new_position, 0, [25, 15])  # Environment bounds
        self.orientation = (new_orientation + np.pi) % (2 * np.pi) - np.pi
        self.path_history.append(self.position.copy())

    def get_vision(self, wall_grid, coverage_grid):
        """Generate RGB vision observation of surroundings"""
        vision_size = 2 * self.vision_range + 1
        vision = np.zeros((vision_size, vision_size, 3), dtype=np.float32)
        x, y = int(round(self.position[0])), int(round(self.position[1]))
        for dy in range(-self.vision_range, self.vision_range+1):
            for dx in range(-self.vision_range, self.vision_range+1):
                nx, ny = x + dx, y + dy
                vision_y = dy + self.vision_range
                vision_x = dx + self.vision_range
                if 0 <= nx < wall_grid.shape[1] and 0 <= ny < wall_grid.shape[0]:
                    if wall_grid[ny, nx] == 1:
                        vision[vision_y, vision_x] = [0.4, 0.4, 0.4]  # Walls
                    elif coverage_grid[nx, ny] == 1:
                        vision[vision_y, vision_x] = [0.2, 0.4, 1.0]  # Cleaned
                    else:
                        vision[vision_y, vision_x] = [1.0, 1.0, 1.0]  # Dirty
                else:
                    vision[vision_y, vision_x] = [0.0, 0.0, 0.0]  # Out-of-bounds
        # Mark robot position (center)
        vision[self.vision_range, self.vision_range] = [1.0, 0.0, 0.0]
        return vision

class GridMazeVacuumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self, size_x=25, size_y=15, max_steps=1000):
        super().__init__()
        self.size_x = size_x
        self.size_y = size_y
        self.max_steps = max_steps
        # Initialize robot and maze
        self.robot = CleaningRobot()
        self._create_maze()
        # Action space: [linear velocity, angular velocity]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -np.pi/2]),
            high=np.array([0.5, np.pi/2]),
            dtype=np.float32
        )
        # Observation space: robot's vision (RGB grid)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(2*self.robot.vision_range+1,
                2*self.robot.vision_range+1, 3),
            dtype=np.float32
        )
        # Visualization
        self.fig = None
        self.ax = None
        self.steps = 0
        self.coverage_percentage = 0.0

    def _create_maze(self):
        """Generate maze walls and initialize coverage grid"""
        self.wall_grid = np.zeros((self.size_y, self.size_x), dtype=int)  # (y, x)
        self.coverage_grid = np.zeros((self.size_x, self.size_y), dtype=float)  # (x, y)
        # Vertical walls with gaps
        for x in [8, 16]:
            gap_start = self.size_y//2 - 3
            for y in range(self.size_y):
                if not (gap_start <= y < gap_start+6):
                    self.wall_grid[y, x] = 1
        # Horizontal wall with gap
        y_wall = self.size_y//2
        gap_start = self.size_x//2 - 3
        for x in range(self.size_x):
            if not (gap_start <= x < gap_start+6):
                self.wall_grid[y_wall, x] = 1
        self.exit_pos = np.array([self.size_x-2, self.size_y-2])

    def reset(self):
        """Reset environment to initial state"""
        self.robot.position = np.array([1.0, 1.0])
        self.robot.orientation = np.random.uniform(-np.pi, np.pi)
        self.robot.path_history = [self.robot.position.copy()]
        self.coverage_grid.fill(0)
        self.steps = 0
        self.coverage_percentage = 0.0
        return self.robot.get_vision(self.wall_grid, self.coverage_grid)

    def step(self, action):
        """Execute one environment step"""
        lin_vel, ang_vel = np.clip(action, self.action_space.low, self.action_space.high)
        new_orientation = (self.robot.orientation + ang_vel * 0.1) % (2 * np.pi)
        dx = lin_vel * math.cos(new_orientation) * 0.1
        dy = lin_vel * math.sin(new_orientation) * 0.1
        new_pos = self.robot.position + np.array([dx, dy])
        # Collision check and position update
        if self._valid_position(new_pos):
            self.robot.update_pose(new_pos, new_orientation)
            reward = 0.1
        else:
            reward = -0.5

        # Update coverage and steps
        self._update_coverage()
        self.steps += 1

        # Termination conditions
        done = (self.coverage_percentage >= 0.95 or
                self.steps >= self.max_steps or
                np.linalg.norm(self.robot.position - self.exit_pos) < 1.0)

        return (self.robot.get_vision(self.wall_grid, self.coverage_grid),
                reward, done, {"coverage": self.coverage_percentage})

    def _valid_position(self, position):
        """Check if position is within bounds and not a wall"""
        x, y = int(position[0]), int(position[1])
        return (0 <= x < self.size_x and
                0 <= y < self.size_y and
                self.wall_grid[y, x] == 0)

    def _update_coverage(self):
        """Update cleaned areas around current position"""
        x, y = int(self.robot.position[0]), int(self.robot.position[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size_x and
                    0 <= ny < self.size_y and
                    self.wall_grid[ny, nx] == 0):
                    self.coverage_grid[nx, ny] = 1

        self.coverage_percentage = np.sum(self.coverage_grid) / (self.size_x * self.size_y)

    def render(self, mode='human'):
        """Render environment state with proper alignment"""
        if self.fig is None:
            self.fig, (self.env_ax, self.vision_ax) = plt.subplots(1, 2, figsize=(12, 6))
            plt.ion()

        # Clear previous frame
        self.env_ax.clear()
        self.vision_ax.clear()

        # Environment setup
        self.env_ax.set_facecolor('white')
        self.env_ax.set_xlim(0, self.size_x)
        self.env_ax.set_ylim(0, self.size_y)
        self.env_ax.set_aspect('equal')
        self.env_ax.invert_yaxis()  # Match matrix coordinates

        # Draw walls
        for y in range(self.size_y):
            for x in range(self.size_x):
                if self.wall_grid[y, x] == 1:
                    self.env_ax.add_patch(patches.Rectangle(
                        (x, y), 1, 1, facecolor='0.5', edgecolor='none'))

        # Draw cleaned areas
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.coverage_grid[x, y] == 1:
                    self.env_ax.add_patch(patches.Rectangle(
                        (x, y), 1, 1, facecolor='blue', alpha=0.3))

        # Draw robot
        robot_circle = patches.Circle(
            self.robot.position, self.robot.coverage_radius,
            facecolor='red', edgecolor='black')
        self.env_ax.add_patch(robot_circle)

        # Draw orientation arrow
        arrow_len = 1.0
        dx = arrow_len * math.cos(self.robot.orientation)
        dy = arrow_len * math.sin(self.robot.orientation)
        self.env_ax.arrow(self.robot.position[0], self.robot.position[1],
                        dx, dy, head_width=0.3, head_length=0.4, fc='black')

        # Draw path history
        if len(self.robot.path_history) > 1:
            path = np.array(self.robot.path_history)
            self.env_ax.plot(path[:,0], path[:,1], 'r-', alpha=0.3)

        # Draw vision
        vision = self.robot.get_vision(self.wall_grid, self.coverage_grid)
        self.vision_ax.imshow(vision, interpolation='nearest', origin='lower')
        self.vision_ax.set_title('Robot Vision')
        self.vision_ax.set_xticks([])
        self.vision_ax.set_yticks([])

        plt.tight_layout()
        plt.pause(0.01)

    def close(self):
        """Close rendering windows"""
        if self.fig is not None:
            plt.close(self.fig)
        super().close()

if __name__ == "__main__":
    env = GridMazeVacuumEnv()
    obs = env.reset()
    try:
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.05)
            if done:
                print(f"Coverage: {info['coverage']:.2%}")
                break
    finally:
        env.close()
