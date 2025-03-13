import math
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class ContinuousVacuumCleanerEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, size=5.0, resolution=50, coverage_radius=0.5, max_steps=2000):
        super().__init__()
        self.size = size
        self.resolution = resolution
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        self.cell_size = size / resolution
        self.max_linear_velocity = 0.5
        self.max_angular_velocity = np.pi/2
        self.dt = 0.1

        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity]),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([0, 0, -np.pi]),
                high=np.array([size, size, np.pi]),
                dtype=np.float32
            ),
            "coverage": spaces.Box(
                low=0, high=1, shape=(resolution * resolution,), dtype=np.float32
            )
        })

        self.fig = None
        self.ax = None
        self.reset()

    def reset(self):
        self.agent_position = np.array([self.size/2, self.size/2], dtype=np.float32)
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.coverage_grid = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.steps = 0
        self.coverage_percentage = 0.0
        self._update_coverage()
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        lin_vel, ang_vel = action
        
        self.agent_orientation = ((self.agent_orientation + ang_vel * self.dt + np.pi) 
                                 % (2 * np.pi)) - np.pi
        
        delta_x = lin_vel * math.cos(self.agent_orientation) * self.dt
        delta_y = lin_vel * math.sin(self.agent_orientation) * self.dt
        new_pos = self.agent_position + np.array([delta_x, delta_y])
        self.agent_position = np.clip(new_pos, [0, 0], [self.size, self.size])
        
        newly_covered, _ = self._update_coverage()
        reward = self._calculate_reward(newly_covered)
        
        self.steps += 1
        done = self.coverage_percentage >= 0.95 or self.steps >= self.max_steps
        
        return self._get_observation(), reward, done, {
            "coverage_percentage": self.coverage_percentage,
            "steps": self.steps
        }

    def _update_coverage(self):
        agent_x, agent_y = self.agent_position
        min_x = max(0, int((agent_x - self.coverage_radius) / self.cell_size))
        max_x = min(self.resolution-1, int((agent_x + self.coverage_radius) / self.cell_size))
        min_y = max(0, int((agent_y - self.coverage_radius) / self.cell_size))
        max_y = min(self.resolution-1, int((agent_y + self.coverage_radius) / self.cell_size))
        
        newly_covered = 0
        for i in range(min_x, max_x+1):
            for j in range(min_y, max_y+1):
                cell_x = (i + 0.5) * self.cell_size
                cell_y = (j + 0.5) * self.cell_size
                distance = (cell_x - agent_x)**2 + (cell_y - agent_y)**2
                
                if distance <= self.coverage_radius**2:
                    if self.coverage_grid[j, i] == 0:
                        newly_covered += 1
                    self.coverage_grid[j, i] = 1
        
        self.coverage_percentage = np.sum(self.coverage_grid) / (self.resolution**2)
        return newly_covered, (max_x - min_x + 1) * (max_y - min_y + 1)

    def _calculate_reward(self, newly_covered):
        center = np.array([self.size/2, self.size/2])
        dist_from_center = np.linalg.norm(self.agent_position - center)
        
        recent_coverage = np.mean(self.coverage_grid[
            max(0, int((self.agent_position[1]-1)/self.cell_size)) : 
            min(self.resolution, int((self.agent_position[1]+1)/self.cell_size)),
            max(0, int((self.agent_position[0]-1)/self.cell_size)) : 
            min(self.resolution, int((self.agent_position[0]+1)/self.cell_size))
        ])
        
        return (
            newly_covered * 2.0 +
            dist_from_center * 0.05 +
            (1 - recent_coverage) * 0.1 -
            0.01 * (newly_covered == 0) -
            0.005 +
            (100 if self.coverage_percentage >= 0.95 else 0)
        )

    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8,8))
            plt.ion()
            
        self.ax.clear()
        self.ax.imshow(
            self.coverage_grid,
            extent=[0, self.size, 0, self.size],
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=1
        )
        
        agent_circle = Circle(
            self.agent_position,
            radius=self.coverage_radius,
            facecolor="red",
            alpha=0.5,
            edgecolor="black"
        )
        self.ax.add_patch(agent_circle)
        
        end_x = self.agent_position[0] + self.coverage_radius * 1.5 * math.cos(self.agent_orientation)
        end_y = self.agent_position[1] + self.coverage_radius * 1.5 * math.sin(self.agent_orientation)
        self.ax.plot([self.agent_position[0], end_x], [self.agent_position[1], end_y], 'k-')
        
        self.ax.text(0.05, 0.95, 
                    f"Coverage: {self.coverage_percentage:.2%}\nSteps: {self.steps}/{self.max_steps}",
                    transform=self.ax.transAxes,
                    fontsize=12,
                    verticalalignment='top')
        
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        plt.pause(0.01)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            plt.ioff()

    def _get_observation(self):
        return {
            "position": np.array([
                self.agent_position[0],
                self.agent_position[1],
                self.agent_orientation
            ], dtype=np.float32),
            "coverage": self.coverage_grid.flatten()
        }
