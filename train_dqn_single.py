import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uav_env_topology import UAVEnv
from dqn_agent import DQNAgent

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

save_path = "agent_DQN_single.npy"

_world_half = 50.0
_max_dist = np.sqrt(2) * 2 * _world_half
_max_heading = np.pi
_max_range = 30.0
_n_ranges = 5

def normalise_obs(obs: np.ndarray) -> np.ndarray:
    out = obs.copy().astype(np.float64)
    out[0] /= _world_half
    out[1] /= _world_half
    out[2] /= _max_heading
    out[3] /= _max_dist
    end_r = 4 + _n_ranges
    out[4:4 + _n_ranges] /= _max_range
    
    return out


def build_env(seed=None) -> UAVEnv:
    return UAVEnv(
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
                dt = 0.5,
                max_steps = max_steps,
                turn_degs = (-30, -15, 0, 15, 30),
                range_degs = (-60, -30, 0, 30, 60),
                max_range = 30.0,
                goal_radius = 3.0,
    )

def plot_training_progress(ep_rewards: list[float], ep_steps: list[int], save_png: str):
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
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Training – Reward (Single UAV)")
    axes[0].grid(alpha=0.3)

    # ── Right: episode length ──
    axes[1].plot(eps, ep_steps, color="darkorange", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Training – Episode length (Single UAV)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_png, dpi=120)
    plt.close(fig)
    print(f"  [plot] Training progress → {save_png}")

def train():
    print("=" * 62)
    print(" Single-UAV DQN Training ")
    print("=" * 62)
    env = build_env(seed=seed)
    o_dim = env.reset().shape[0]

    print(f"\n obs_dim       : {o_dim}")
    print(f" num_actions   : {env.num_actions}")
    print(f" episodes      : {max_episodes}")
    print(f" steps/episode : {max_steps}")
    print(f" save_path     : {save_path}\n")

    print(f"\n  ── DQN params ──")
    print(f"  eps start/decay/min : {eps_start} / {eps_decay} / {eps_min}")
    print(f"  LR / L2            : {LR} / {L2}")
    print(f"  γ                  : {GAMMA}")
    print(f"  Batch / Buffer     : {batch_size} / {buffer_cap:,}")
    print(f"  TargetUpdateFreq   : {target_update_freq}")
    print(f"  Hidden units       : {hidden_size} / {hidden_size}")
    print(f"  use double DQN     : implemented in agent logic\n")

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

    ep_rewards: list[float] = []
    ep_steps: list[int] = []
    best_r = -np.inf
    t0 = time.time()

    for episode in range(1, max_episodes + 1):
        obs = normalise_obs(env.reset())
        done = False 
        ep_r = 0.0
        step = 0

        while not done and step < max_steps:
            action = agent.select_action(obs)
            next_obs_raw, reward, done, info = env.step(action)
            next_obs = normalise_obs(next_obs_raw)

            agent.store(obs, action, reward, next_obs, done)
            agent.learn()
            agent.decay_epsilon()

            obs = next_obs
            ep_r += reward
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
    np.save('single_training_history.npy', {
    'ep_rewards': ep_rewards,
    'ep_steps': ep_steps
    })
    print(f"\nSaved training history → single_training_history.npy")
    
    plot_training_progress(ep_rewards, ep_steps, save_png=save_path.replace(".npy", "_training.png"))

if __name__ == "__main__":
    train()