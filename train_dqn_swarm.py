import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uavswarm_env_topology import SwarmUAVEnv
from dqn_agent import DQNAgent
from swarm_reward_shaper import SwarmRewardShaper

num_uavs = 5
max_episodes = 400
max_steps = 400
seed = 42

eps_start = 1.0
eps_min = 0.01
eps_decay = 0.005
LR = 0.001
L2 = 1e-4
GAMMA = 0.99
batch_size = 64
buffer_cap = 100000
target_update_freq = 4
hidden_size = 128

desired_spacing = 6.0
cohesion_weight = 0.3
separation_weight = 4.0
alignment_weight = 0.8

save_path = "agent_DQN_swarm.npy"

_world_half = 50.0
_max_dist = np.sqrt(2) * 2 * _world_half
_max_heading = np.pi
_max_range = 30.0
_n_ranges = 5

def normalise_obs(obs: np.ndarray) -> np.ndarray:
    out = obs.copy().astype(np.float64)
    out[:, 0] /= _world_half
    out[:, 1] /= _world_half
    out[:, 2] /= _max_heading
    out[:, 3] /= _max_dist
    end_r = 4 + _n_ranges
    out[:, 4:end_r] /= _max_range
    if out.shape[1] > end_r:
        out[:, end_r:] /= _world_half
    return out


def build_env(seed=None) -> SwarmUAVEnv:
    return SwarmUAVEnv(num_uavs = num_uavs,
                       num_obstacles = 8,
                       world_bounds = ((-50, 50), (-50, 50)),
                       start_pos = (-40, -40),
                       goal_pos = (40, 40),
                       min_radius = 3.0,
                       max_radius = 8.0,
                       seed = seed,
                       obstacle_clearance = 1.0,
                       start_goal_clearance = 6.0,
                       speed = 3.0,
                       goal_radius = 3.0,
                       detection_radius = 30.0,
                       uav_separation = 2.0,
                       spawn_spread = 3.0,
                       dt = 0.5,
                       max_steps = max_steps,
    )

def plot_training_progress(ep_rewards: list, ep_steps: list, save_path: str = "training_progress.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eps = np.arange(1, len(ep_rewards) + 1)

    # ── Left: reward curve ──
    axes[0].plot(eps, ep_rewards, alpha=0.35, color="steelblue", lw=0.8)
    w = min(20, len(ep_rewards))
    if len(ep_rewards) >= w:
        smooth = np.convolve(ep_rewards, np.ones(w) / w, mode="valid")
        axes[0].plot(np.arange(w, len(ep_rewards) + 1), smooth,
                     color="navy", lw=2.0, label=f"{w}-ep avg")
        axes[0].legend()
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Mean swarm reward / step")
    axes[0].set_title("Training – Reward")
    axes[0].grid(alpha=0.3)

    # ── Right: episode length ──
    axes[1].plot(eps, ep_steps, color="darkorange", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Training – Episode length")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Training progress → {save_path}")

def plot_episode(env: SwarmUAVEnv, trajectories: list,
                 title: str, save_path: str):
    """
    trajectories : list of (T_i, 2) arrays, one per UAV.
    Circle markers = spawn positions, squares = final positions.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    (x0, x1), (y0, y1) = env.world_bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Obstacles
    for center, radius in env.obstacles:
        ax.add_patch(plt.Circle(center, radius, color="red", alpha=0.40))

    # World boundary
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor="black", linestyle="--"))

    # Start / goal
    ax.plot(*env.start_pos, "go", markersize=12, zorder=5, label="Start")
    ax.plot(*env.goal_pos,  "b*", markersize=14, zorder=5, label="Goal")

    # Per-UAV trajectories
    colors = cm.tab10(np.linspace(0, 1, len(trajectories)))
    for i, (traj, col) in enumerate(zip(trajectories, colors)):
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.6,
                label=f"UAV {i}")
        ax.plot(*traj[0],  "o", color=col, markersize=7)
        ax.plot(*traj[-1], "s", color=col, markersize=9)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Episode trajectory → {save_path}")     

def train():
    print("=" * 62)
    print(" UAV Swarm Double DQN Training ")
    print("=" * 62)
    env = build_env(seed=seed)
    o_dim = env.reset().shape[1]

    print(f"\n UAVs: {num_uavs}")
    print(f" obs_dim: {o_dim}")
    print(f"  num_actions : {env.num_actions}")
    print(f"  Episodes    : {max_episodes}   Steps/ep: {max_steps}")
    print(f"\n  ── DQN params (from agent_DQN.mat) ──")
    print(f"  ε start/decay/min : {eps_start} / {eps_decay} (additive/step) / {eps_min}")
    print(f"  LR / L2            : {LR} / {L2}")
    print(f"  γ                  : {GAMMA}")
    print(f"  Batch / Buffer     : {batch_size} / {buffer_cap:,}")
    print(f"  TargetUpdateFreq   : {target_update_freq}")
    print(f"  Hidden units       : {hidden_size} / {hidden_size}")
    print(f"  UseDoubleDQN       : True\n")

    agent = DQNAgent(
        obs_dim = o_dim,
        num_actions = env.num_actions,
        hidden = hidden_size,
        lr = LR,
        l2 = L2,
        gamma = GAMMA,
        batch_size = batch_size,
        buffer_capacity = buffer_cap,
        target_update_freq = target_update_freq,
        eps_start = eps_start,
        eps_decay = eps_decay,
        eps_min = eps_min,
        seed = seed,
    )

    shaper = SwarmRewardShaper(
        num_uavs = num_uavs,
        desired_spacing = desired_spacing,
        cohesion_weight = cohesion_weight,
        separation_weight = separation_weight,
        alignment_weight = alignment_weight,
        separation_min = env.uav_body_radius * 2.0 + env.uav_separation,
    )

    ep_rewards: list[float] = []
    ep_steps: list[int] = []
    best_mean_r = -np.inf
    t0 = time.time()

    for episode in range(1, max_episodes + 1):
        obs = normalise_obs(env.reset())
        dones = np.zeros(num_uavs, dtype=bool)
        ep_r = 0.0
        step = 0

        while not np.all(dones) and step < max_steps:
            actions = agent.select_actions(obs)
            next_obs_raw, base_rew, dones, info = env.step(actions)
            next_obs = normalise_obs(next_obs_raw)

            shape_r = shaper.shape(env.pos, env.heading, dones)
            rewards = base_rew + shape_r

            agent.store_swarm(obs, actions, rewards, next_obs,dones)
            agent.learn()
            agent.decay_epsilon()

            obs = next_obs
            ep_r += rewards.mean()
            step += 1
        
        ep_rewards.append(ep_r)
        ep_steps.append(step)

        if episode % 10 == 0 or episode == 1:
            mean20 = np.mean(ep_rewards[-20:])
            print(
                f"  Ep {episode:>4}/{max_episodes} | "
                f"steps={step:>4} | "
                f"mean_r={ep_r:>9.2f} | "
                f"avg20={mean20:>9.2f} | "
                f"ε={agent.eps:.4f} | "
                f"t={time.time() - t0:>6.1f}s"
            )

        if ep_r > best_mean_r:
            best_mean_r = ep_r
            agent.save(save_path)

    print(f"\n Training complete. Best episode reward: {best_mean_r:.2f}")
    plot_training_progress(ep_rewards, ep_steps)
    return agent, env

def evaluate(agent: DQNAgent, env: SwarmUAVEnv, num_eval: int = 3):
    """
    Run `num_eval` greedy episodes and plot each trajectory.
    
    """
    print("\n" + "=" * 62)
    print("  Post-training Evaluation  (greedy, ε=0)")
    print("=" * 62)

    agent.load(save_path)

    for ep in range(1, num_eval + 1):
        obs   = normalise_obs(env.reset())
        dones = np.zeros(num_uavs, dtype=bool)

        trajs = [[env.pos[i].copy()] for i in range(num_uavs)]
        step  = 0
        tot_r = np.zeros(num_uavs)

        while not np.all(dones) and step < env.max_steps:
            actions = agent.select_actions(obs, greedy=True)
            obs_raw, rew, dones, info = env.step(actions)
            obs = normalise_obs(obs_raw)
            tot_r += rew
            step  += 1
            for i in range(num_uavs):
                trajs[i].append(env.pos[i].copy())

        trajs_np = [np.array(t) for t in trajs]
        reached  = sum(
            1 for ev in info.get("events", [])
            if "reached the goal" in ev
        )

        print(f"\n  Eval {ep} | steps={step} | "
              f"reached_goal={reached}/{num_uavs}")
        for i in range(num_uavs):
            print(f"    UAV {i}:  total_r={tot_r[i]:>8.1f}  "
                  f"pos={env.pos[i].round(1)}")
        for ev in info.get("events", []):
            print(f"    {ev}")

        plot_episode(
            env, trajs_np,
            title=f"DQN Swarm Eval #{ep}  (reached={reached}/{num_uavs})",
            save_path=f"eval_episode_{ep}.png",
        )

if __name__ == "__main__":
    trained_agent, trained_env = train()
    evaluate(trained_agent, trained_env, num_eval=3)

    print("\n Output files:")
    for f in [save_path, "training_progress.png","eval_episode_1.png","eval_episode_2.png","eval_episode_3.png"]:
        status = "✓" if os.path.exists(f) else "✗"
        print(f"  {status} {f}")
