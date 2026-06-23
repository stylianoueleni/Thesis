import numpy as np
import matplotlib.pyplot as plt

class UAVEnvTopology:
    def __init__(
        self,
        num_obstacles=4,
        world_bounds=((-50,50),(-50,50)),
        start_pos=(-40,-40),
        goal_pos=(40,40),
        min_radius=3.0,
        max_radius=10.0,
        seed=None,
        obstacle_clearance=1.0,
        start_goal_clearance=5.0,
        max_attempts_per_obstacle=500,
    ):
        """
        Initialize the UAV environment with obstacles and parameters.

        num_obstacles: Number of circular obstacles
        world_bounds: Tuple defining the (x_min, x_max) and (y_min, y_max)
        start_pos: Starting position of the UAV
        goal_pos: Goal position of the UAV
        min_radius: Minimum radius of obstacles
        max_radius: Maximum radius of obstacles
        seed: Random seed for reproducibility
        obstacle_clearance: Minimum distance between obstacles
        start_goal_clearance: Minimum distance from start/goal to obstacles
        max_attempts_per_obstacle: Max attempts to place each obstacle

        """

        if min_radius <= 0 or max_radius <= 0:
            raise ValueError("Obstacle radii must be positive.")
        
        if min_radius > max_radius:
            raise ValueError("min_radius cannot be greater than max_radius.")
        
        (x_min, x_max), (y_min, y_max) = world_bounds
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Invalid world bounds.")
        
        if not(x_min < start_pos[0] < x_max and y_min < start_pos[1] < y_max):
            raise ValueError("Start position out of world bounds.") 
        
        if not(x_min < goal_pos[0] < x_max and y_min < goal_pos[1] < y_max):
            raise ValueError("Goal position out of world bounds.")
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.world_bounds = world_bounds
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.seed = seed
        self.obstacle_clearance = obstacle_clearance
        self.start_goal_clearance = start_goal_clearance
        self.max_attempts_per_obstacle = max_attempts_per_obstacle

        # Generate obstacles
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = []
        x_min, x_max = self.world_bounds[0]
        y_min, y_max = self.world_bounds[1]

        for _ in range(self.num_obstacles):
            placed = False
            for _ in range(self.max_attempts_per_obstacle):
                radius = self.rng.uniform(self.min_radius, self.max_radius)
                x_center = self.rng.uniform(x_min + radius, x_max - radius)
                y_center = self.rng.uniform(y_min + radius, y_max - radius)
                center = np.array([x_center, y_center])

                # Check clearance with existing obstacles
                if all(
                    np.linalg.norm(center - obs_center) >= (radius + obs_radius + self.obstacle_clearance)
                    for (obs_center, obs_radius) in obstacles
                ):
                    # Check clearance with start and goal positions
                    if (
                        np.linalg.norm(center - self.start_pos) >= (radius + self.start_goal_clearance) and
                        np.linalg.norm(center - self.goal_pos) >= (radius + self.start_goal_clearance)
                    ):
                        obstacles.append((center, radius))
                        placed = True
                        break
            if not placed:
                raise RuntimeError("Failed to place all obstacles with given constraints.")

        return obstacles
    
    def as_dict(self):
        return {
            "world_bounds": self.world_bounds,
            "start_pos": self.start_pos.tolist(),
            "goal_pos": self.goal_pos.tolist(),
            "obstacles": [
                {"center": center.tolist(), "radius": radius}
                for (center, radius) in self.obstacles
            ],
            "seed": self.seed,
        }
    
    def plot_environment(self, title="UAV Environment Topology"):
        fig, ax = plt.subplots(figsize=(8,8))
        x_min, x_max = self.world_bounds[0]
        y_min, y_max = self.world_bounds[1]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Plot obstacles
        for (center, radius) in self.obstacles:
            circle = plt.Circle(center, radius, color='r', alpha=0.5)
            ax.add_artist(circle)

        # Plot start and goal positions
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'bo', markersize=10, label='Goal')

        # Plot world boundary
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor='black',
            linestyle='--'
        )

        ax.add_patch(rect)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        plt.grid()
        plt.show()
    
class UAVEnv(UAVEnvTopology):
    def __init__(
            self,
            num_obstacles=4,
            world_bounds=((-50,50),(-50,50)),
            start_pos=(-40,-40),
            goal_pos=(40,40),
            min_radius=3.0,
            max_radius=10.0,
            seed=None,
            obstacle_clearance=1.0,
            start_goal_clearance=5.0,
            max_attempts_per_obstacle=500,
            dt=0.5,
            max_steps=400,
            speed=3.0,
            turn_degs=(-30, -15, 0, 15, 30),
            range_degs=(-60, -30, 0, 30, 60),
            max_range=30.0,
            goal_radius=2.0,
        ):
        
            super().__init__(
                num_obstacles=num_obstacles,
                world_bounds=world_bounds,
                start_pos=start_pos,
                goal_pos=goal_pos,
                min_radius=min_radius,
                max_radius=max_radius,
                seed=seed,
                obstacle_clearance=obstacle_clearance,
                start_goal_clearance=start_goal_clearance,
                max_attempts_per_obstacle=max_attempts_per_obstacle,
            )

            self.dt = dt
            self.max_steps = max_steps
            self.speed = speed
            self.turn_angles = np.deg2rad(np.array(turn_degs, dtype=float))
            self.num_actions = len(turn_degs)
            self.range_angles = np.deg2rad(np.array(range_degs, dtype=float))
            self.max_range = max_range
            self.goal_radius = goal_radius

            self.obstacle_centers = np.array([c for (c,r) in self.obstacles])
            self.obstacle_radii = np.array([r for (c,r) in self.obstacles])

            self.pos = None
            self.heading = None
            self.step_count = 0
            self.prev_dist = None

    def _wrap_to_pi(self, angle):
        """ Wrap angle to [-pi, pi] """
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def _compute_ranges(self, pos, heading):
        """ Compute range measurements along each sensor angle """
        ranges = np.full(len(self.range_angles), self.max_range, dtype=float)

        (x_min, x_max), (y_min, y_max) = self.world_bounds
        step = 0.5

        for i, angle in enumerate(self.range_angles):
            ray_angle = self._wrap_to_pi(heading + angle)
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=float)

            found = self.max_range
            d = 0.0
            while d <= self.max_range:
                point = pos + d * ray_dir

                if not (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max):
                    found = d
                    break

                hit = False
                for obs_center, obs_radius in zip(self.obstacle_centers, self.obstacle_radii):
                    if np.linalg.norm(point - obs_center) <= obs_radius:
                        found = d
                        hit = True
                        break

                if hit:
                    break

                d += step
            ranges[i] = found

        return ranges

    def _check_collision(self, pos):
        # Check if position collides with any obstacle
        for obs_center, obs_radius in zip(self.obstacle_centers, self.obstacle_radii):
            if np.linalg.norm(pos - obs_center) <= (obs_radius + 0.5):
                return True
                    
        # Check if position is out of world bounds
        (x_min, x_max), (y_min, y_max) = self.world_bounds
        if not (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max):
            return True
                
        return False
        
    def build_observation(self):
        """ Build observation vector: range measurements + distance and angle to goal """
        ranges = self._compute_ranges(self.pos, self.heading)
        to_goal = self.goal_pos - self.pos
        distance_to_goal = np.linalg.norm(to_goal)
        
        obs = np.concatenate([
            self.pos,
            np.array([self.heading]),
            np.array([distance_to_goal]),
            ranges
        ])

        return obs
        
    def reset(self):
        """ Reset environment to start a new episode """
        noise = self.rng.normal(scale=2.0, size=2)
        self.pos = self.start_pos + noise

        vec_to_goal = self.goal_pos - self.pos
        heading_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        self.heading = heading_to_goal + self.rng.normal(scale=0.2)

        self.step_count = 0
        self.prev_dist = None
        return self.build_observation()
        
    def step(self, action_idx):
        """ Take a step in the environment given an action index """
        if not (0 <= action_idx < self.num_actions):
            raise ValueError("Invalid action index.")
                
        self.step_count += 1

        # Turn
        dpsi = self.turn_angles[action_idx]
        self.heading = self._wrap_to_pi(self.heading + dpsi)
        
        #Move
        dx = self.speed * self.dt * np.array([np.cos(self.heading), np.sin(self.heading)], dtype=float)
        self.pos += dx

        dist_to_goal = np.linalg.norm(self.goal_pos - self.pos)

        # Collision
        if self._check_collision(self.pos):
            obs = self.build_observation()
            reward = -300.0
            done = True
            info = {'reason': 'collision'}
            return obs, reward, done, info
                
        # Goal reached
        if dist_to_goal <= self.goal_radius:
            obs = self.build_observation()
            heading_to_goal = np.arctan2(
                self.goal_pos[1] - self.pos[1],
                self.goal_pos[0] - self.pos[0]
            )
            heading_error = self._wrap_to_pi(self.heading - heading_to_goal)
            reward = 1200.0 - 50.0 * abs(heading_error)
            done = True
            info = {'reason': 'goal_reached'}
            return obs, reward, done, info
                
        if self.prev_dist is None:
            self.prev_dist = np.linalg.norm(self.goal_pos - self.start_pos)

        progress = self.prev_dist - dist_to_goal

        heading_to_goal = np.arctan2(
            self.goal_pos[1] - self.pos[1],
            self.goal_pos[0] - self.pos[0]
        )
        heading_error = abs(self._wrap_to_pi(self.heading - heading_to_goal))

        energy_proxy = abs(dpsi) / np.max(np.abs(self.turn_angles))

        energy_proxy = min(energy_proxy, 1.0)

        reward = 10.0 * progress - 5.0 * abs(heading_error) - 1.0 * energy_proxy - 0.5

        done = self.step_count >= self.max_steps
        obs = self.build_observation()
        self.prev_dist = dist_to_goal
        info = {}

        return obs, reward, done, info
        
if __name__ == "__main__":
    # Example usage
    env = UAVEnv(
        num_obstacles=8,
        world_bounds=((-50, 50), (-50, 50)),
        start_pos=(-40, -40),
        goal_pos=(40, 40),
        min_radius=3.0,
        max_radius=8.0,
        seed=42,
        obstacle_clearance=1.0,
        start_goal_clearance=6.0,
    )

    env.plot_environment("Random UAV Topology")

    obs = env.reset()
    print("Initial obs shape:", obs.shape)

    done = False
    total_reward = 0.0
    while not done:
        action = env.rng.integers(0, env.num_actions)
        obs, r, done, info = env.step(action)
        total_reward += r

    print("Episode finished. Total reward:", total_reward, "| info:", info)