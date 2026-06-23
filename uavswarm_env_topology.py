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
    
class SwarmUAVEnv(UAVEnvTopology):
    """
    Multi-agent version of UAVEnv for a swarm of UAVs (default: 5).

    key differences from single-agent UAVEnv:
    -state is arrays: pos[N,2], heading[N]
    -step(actions) takes N discrete action indices
    -returns: obs[N,obs_dim], rewards[N], dones[N], infos[dict]
    -episode ends when all UAVs are done OR a global time limit hits
    """
    def __init__(
            self,
            num_uavs=5,
            num_obstacles=8,
            world_bounds=((-50,50),(-50,50)),
            start_pos=(-40,-40),
            goal_pos=(40,40),
            min_radius=3.0,
            max_radius=8.0,
            seed=None,
            obstacle_clearance=1.0,
            start_goal_clearance=6.0,
            max_attempts_per_obstacle=500,
            dt=1.0,
            max_steps=400,
            speed=30.0,
            goal_radius=5.0,
            turn_degs=(-30, -15, 0, 15, 30),
            range_degs=(-60, -30, 0, 30, 60),
            detection_radius=30.0,
            uav_body_radius=0.5,
            uav_separation=2.0,
            spawn_spread=3.0,
            include_other_uavs_in_obs=True,
            other_uavs_clip=50.0,
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

            self.num_uavs = int(num_uavs)
            if self.num_uavs < 2:
                raise ValueError("SwarmUAVEnv requires at least 2 UAVs.")
            
            self.dt = float(dt)
            self.max_steps = int(max_steps)
            self.speed = float(speed)
            self.turn_angles = np.deg2rad(np.array(turn_degs, dtype=float))
            self.num_actions = len(turn_degs)
            self.range_angles = np.deg2rad(np.array(range_degs, dtype=float))
            self.detection_radius = float(detection_radius)
            self.goal_radius = float(goal_radius)

            self.uav_body_radius = float(uav_body_radius)
            self.uav_separation = float(uav_separation)
            self.spawn_spread = float(spawn_spread)
            self.include_other_uavs_in_obs = bool(include_other_uavs_in_obs)
            self.other_uavs_clip = float(other_uavs_clip)

            self.obstacle_centers = np.array([c for (c,r) in self.obstacles], dtype=float)
            self.obstacle_radii = np.array([r for (c,r) in self.obstacles], dtype=float)

            self.pos = None
            self.heading = None
            self.step_count = 0
            self.prev_dist = None
            self.dones = None

    def _wrap_to_pi(self, angle):
        """ Wrap angle to [-pi, pi] """
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def _compute_ranges(self, pos, heading):
        """ Compute range measurements along each sensor angle """
        ranges = np.full(len(self.range_angles), self.detection_radius, dtype=float)

        (x_min, x_max), (y_min, y_max) = self.world_bounds
        step = 0.5

        for i, angle in enumerate(self.range_angles):
            ray_angle = self._wrap_to_pi(heading + angle)
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=float)

            found = self.detection_radius
            d = 0.0
            while d <= self.detection_radius:
                point = pos + d * ray_dir

                if not (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max):
                    found = d
                    break

                hit = False
                for obs_center, obs_radius in zip(self.obstacle_centers, self.obstacle_radii):
                    if np.linalg.norm(point - obs_center) <= (obs_radius + self.uav_body_radius):
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
            if np.linalg.norm(pos - obs_center) <= (obs_radius + self.uav_body_radius):
                return True
                    
        # Check if position is out of world bounds
        (x_min, x_max), (y_min, y_max) = self.world_bounds
        if not (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max):
            return True
                
        return False
        
    def _check_inter_agent_collision(self):
        """
        Returns a boolean array collided[N] if UAV i is too close to any other UAV.
        """
        N = self.num_uavs
        collided = np.zeros(N, dtype=bool)
        for i in range(N):
            if self.dones[i]:
                continue
            for j in range(i+1, N):
                if self.dones[j]:
                    continue
                d = np.linalg.norm(self.pos[i] - self.pos[j])
                if d <= (2 * self.uav_body_radius + self.uav_separation):
                    collided[i] = True
                    collided[j] = True
        return collided
    
    def build_observation(self):
        """ Build observation vector: range measurements + distance and angle to goal """
        N = self.num_uavs
        obs_list = []

        for i in range(N):
            ranges = self._compute_ranges(self.pos[i], self.heading[i])

            to_goal = self.goal_pos - self.pos[i]
            distance_to_goal = np.linalg.norm(to_goal)
           
            core = np.concatenate(
                [
                   self.pos[i],
                   np.array([self.heading[i]]),
                   np.array([distance_to_goal]),
                   ranges,
                ]
            )

            if self.include_other_uavs_in_obs:
                rel = []
                for j in range(N):
                    if i == j:
                        continue
                    v = self.pos[j] - self.pos[i]
                    v = np.clip(v, -self.other_uavs_clip, self.other_uavs_clip)
                    rel.append(v)
                rel = np.concatenate(rel) if rel else np.array([], dtype=float)
                core = np.concatenate([core, rel])
            
            obs_list.append(core)
        return np.stack(obs_list, axis=0)
        
    def reset(self):
        """ 
         Spawn UAVs near start_pos with random offsets and headings roughly toward goal.
        """
        N = self.num_uavs
        self.step_count = 0
        self.dones = np.zeros(N, dtype=bool)

        # Spawn with simple rejection to avoid immediate collisions
        positions = []
        attempts = 0
        max_attempts = 5000

        while len(positions) < N and attempts < max_attempts:
            attempts += 1
            candidate = self.start_pos + self.rng.normal(scale=self.spawn_spread, size=2)

            # avoid obstacles/bounds on spawn
            if self._check_collision(candidate):
                continue

            # enforce start-goal clearance
            ok = True
            for obs_center, obs_radius in zip(self.obstacle_centers, self.obstacle_radii):
                if np.linalg.norm(candidate - obs_center) <= (obs_radius + self.start_goal_clearance):
                    ok = False
                    break
            if not ok:
                continue

            # avoid too close spawns
            ok = True
            for pos in positions:
                if np.linalg.norm(candidate - pos) <= (2 * self.uav_body_radius + self.uav_separation):
                    ok = False
                    break
            if not ok:
                continue

            positions.append(candidate)
        
        if len(positions) < N:
            raise RuntimeError("Failed to spawn all UAVs without collisions.")
        
        self.pos = np.stack(positions, axis=0)

        # Headings roughly toward goal
        headings = []
        for i in range(N):
            vec_to_goal = self.goal_pos - self.pos[i]
            heading_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0])
            headings.append(self._wrap_to_pi(
                heading_to_goal + self.rng.normal(scale=0.2)))
        self.heading = np.array(headings, dtype=float)

        # Distance baseline for progress reward
        self.prev_dist = np.linalg.norm(self.goal_pos - self.pos, axis=1)

        return self.build_observation()
        
    def _move_with_substeps(self, i, dx, substeps=10):
        """ Move UAV i in smaller substeps to check for collisions """
        step_dx = dx / substeps
        for _ in range(substeps):
            self.pos[i] += step_dx
            if self._check_collision(self.pos[i]):
                return True  # Collision occurred
        return False  # No collision
    
    def step(self, actions):
        """ Take a step in the environment given an action index """
        actions = np.asarray(actions, dtype=int)
        if actions.shape != (self.num_uavs,):
            raise ValueError(f"Actions must be of shape ({self.num_uavs},) got {actions.shape}")
                
        self.step_count += 1
        N = self.num_uavs
        rewards = np.zeros(N, dtype=float)
        info = {"events": []}

        # Turn and move for each non-done UAV
        dpsi = np.zeros(N, dtype=float)
        for i in range(N):
            if self.dones[i]:
                continue

            a = actions[i]
            if not (0 <= a < self.num_actions):
                raise ValueError(f"Action index {a} out of bounds for UAV {i}.")
            
            dpsi[i] = self.turn_angles[a]
            self.heading[i] = self._wrap_to_pi(self.heading[i] + dpsi[i])

            dx = self.speed * self.dt * np.array(
                [np.cos(self.heading[i]), np.sin(self.heading[i])], dtype=float)
            
            collided = self._move_with_substeps(i, dx, substeps=10)
            if collided:
                self.dones[i] = True
                rewards[i] = -300.0
                info["events"].append(f"UAV {i} collided during movement.")

        # Inter-agent collisions
        inter = self._check_inter_agent_collision()
        for i in range(N):
            if self.dones[i]:
                continue
            if inter[i]:
                self.dones[i] = True
                rewards[i] = -250.0
                info["events"].append(f"UAV {i} collided with another UAV.")

        # Goal checks and shaping rewards for those still active
        for i in range(N):
            if self.dones[i]:
                continue
            
            dist_to_goal = np.linalg.norm(self.goal_pos - self.pos[i])

            # Goal reached
            if dist_to_goal <= self.goal_radius:
                heading_to_goal = np.arctan2(
                    self.goal_pos[1] - self.pos[i, 1],
                    self.goal_pos[0] - self.pos[i, 0]
                )
                heading_error = abs(self._wrap_to_pi(self.heading[i] - heading_to_goal))

                self.dones[i] = True
                rewards[i] = 1200.0 - (50.0 * heading_error)
                info["events"].append(f"UAV {i} reached the goal.")
                continue

            # Shaping reward: progress toward goal
            progress = self.prev_dist[i] - dist_to_goal

            heading_to_goal = np.arctan2(
                self.goal_pos[1] - self.pos[i, 1],
                self.goal_pos[0] - self.pos[i, 0]
            )
            heading_error = abs(self._wrap_to_pi(self.heading[i] - heading_to_goal))

            energy_proxy = abs(dpsi[i]) / np.max(np.abs(self.turn_angles))
            energy_proxy = min(float(energy_proxy), 1.0)

            rewards[i] = (10.0 * progress) - (5.0 * heading_error) - (1.0 * energy_proxy) - 0.5

            self.prev_dist[i] = dist_to_goal

        # Time limit check
        if self.step_count >= self.max_steps:
            for i in range(N):
                if not self.dones[i]:
                    self.dones[i] = True
                    info["events"].append(f"UAV {i} reached time limit.")

        obs = self.build_observation()
        dones = self.dones.copy()

        return obs, rewards, dones, info
    
    def plot_swarm(self, title="Swarm UAV Environment", show_headings=True):
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

        # Plot UAVs
        if self.pos is not None:
            for i in range(self.num_uavs):
                ax.plot(self.pos[i,0], self.pos[i,1], 'ko', markersize=6)
                ax.text(self.pos[i,0]+0.6, self.pos[i,1]+0.6, f'UAV {i}', fontsize=9)
                if show_headings:
                    ax.arrow(
                        self.pos[i,0],
                        self.pos[i,1],
                        3.0 * np.cos(self.heading[i]),
                        3.0 * np.sin(self.heading[i]),
                        head_width=1.2,
                        length_includes_head=True,
                    )
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        plt.grid()
        
        return fig
        
if __name__ == "__main__":
    # Example usage
    env = SwarmUAVEnv(
        num_uavs=5,
        num_obstacles=8,
        world_bounds=((-50, 50), (-50, 50)),
        start_pos=(-40, -40),
        goal_pos=(40, 40),
        min_radius=3.0,
        max_radius=8.0,
        seed=None,
        obstacle_clearance=1.0,
        start_goal_clearance=6.0,

        speed=30.0,
        goal_radius=5.0,
        detection_radius=30.0,

        dt=1.0,
        max_steps=200,
    )

    obs = env.reset()
    print("Initial obs shape:", obs.shape)
    env.plot_swarm(title="Initial Swarm UAV Environment")

    dones = np.array([False]*env.num_uavs)
    total_reward = np.zeros(env.num_uavs, dtype=float)
    while not np.all(dones):
        action = env.rng.integers(0, env.num_actions, size=env.num_uavs)
        obs, r, dones, info = env.step(action)
        print("step", env.step_count, "dones", dones, "r", r, "events", info["events"])
        total_reward += r

    print("Episode finished. Total reward:", total_reward)
    print("Final dones:", dones)
    print("Final positions:\n", env.pos)
    print("Events:", info.get("events", []))
    env.plot_swarm(title="Final Swarm UAV Environment")
    plt.show()

