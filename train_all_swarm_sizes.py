import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uavswarm_env_topology import SwarmUAVEnv
from dqn_agent import DQNAgent
from swarm_reward_shaper import SwarmRewardShaper

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

SWARM_CONFIGS = [
    {"num_uavs": 3, "spawn_spread": 3.0, "save_path": "agent_DQN_three.npy"},
    {"num_uavs": 4, "spawn_spread": 3.0, "save_path": "agent_DQN_four.npy"},
]

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


def build_env(num_uavs: int, spawn_spread: float, seed=None) -> SwarmUAVEnv:
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
                       spawn_spread = spawn_spread,
                       dt = 0.5,
                       max_steps = max_steps,
    )

def plot_training_progress(ep_rewards, ep_steps, num_uavs, save_png):
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
    axes[0].set_title(f"Training - Reward ({num_uavs} UAVs)")
    axes[0].grid(alpha=0.3)

    # ── Right: episode length ──
    axes[1].plot(eps, ep_steps, color="darkorange", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title(f"Training - Episode length ({num_uavs} UAVs)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_png, dpi=120)
    plt.close(fig)
    print(f"  [plot] Training progress → {save_png}")

def train_one(num_uavs: int, spawn_spread: float, save_path: str):
    print()
    print("=" * 62)
    print(f" Training DQN - {num_uavs} UAVs ")
    print("=" * 62)

    env = build_env(num_uavs, spawn_spread, seed=seed)
    o_dim = env.reset().shape[1]

    print(f" obs_dim: {o_dim}")
    print(f"  num_actions : {env.num_actions}")
    print(f" save_path   : {save_path}\n")
    
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

    ep_rewards = []
    ep_steps = []
    best_r = -np.inf
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

        if ep_r > best_r:
            best_r = ep_r
            agent.save(save_path)

    print(f"\n Training complete. Best episode reward: {best_r:.2f}")
    png = save_path.replace(".npy", "_training.png")
    plot_training_progress(ep_rewards, ep_steps, num_uavs, png)
    return best_r

def main():
    """
    Trains each swarm size in a fresh subprocess. This ensures each agent
    gets the same OS-level random state it would get in isolation,
    producing the same convergence quality as standalone training.

    """
    import subprocess
    import sys

    t_start = time.time()

    print("=" * 62)
    print(" UAV Swarm DQN — Training all swarm sizes")
    print(f" Sizes: {[c['num_uavs'] for c in SWARM_CONFIGS]} UAVs")
    print(f" Episodes per size: {max_episodes}")
    print(" Mode: subprocess per size (isolated random state)")
    print("=" * 62)

    for cfg in SWARM_CONFIGS:
        n   = cfg["num_uavs"]
        sp = cfg["spawn_spread"]
        pth = cfg["save_path"]
        print(f"\n{'='*62}")
        print(f" Launching {n}-UAV training as subprocess...")
        print(f"\n{'='*62}")

        env_vars = os.environ.copy()
        env_vars["TRAIN_NUM_UAVS"] = str(n)
        env_vars["TRAIN_SPAWN_SPREAD"] = str(sp)
        env_vars["TRAIN_SAVE_PATH"] = pth

        result = subprocess.run(
            [sys.executable, __file__, "--train-one"],
            env = env_vars,
            check=False
        )
        if result.returncode != 0:
            print(f"  WARNING {n}-UAV subprocess exited with code {result.returncode}")
 
    total = time.time() - t_start
 
    print()
    print("=" * 62)
    print(" All training complete — Summary")
    print("=" * 62)
    print(f"\n  Total time: {total/60:.1f} min")
    print("\n Output files:")
    for cfg in SWARM_CONFIGS:
        for f in [cfg["save_path"],
                  cfg["save_path"].replace(".npy", "_training.png")]:
            print(f"  {'✓' if os.path.exists(f) else '✗'}  {f}")
 
if __name__ == "__main__":
    import sys
    if "--train-one" in sys.argv:
        n   = int(os.environ["TRAIN_NUM_UAVS"])
        sp  = float(os.environ["TRAIN_SPAWN_SPREAD"])
        pth = os.environ["TRAIN_SAVE_PATH"]
        train_one(n, sp, pth)
    else:
        main()
 
