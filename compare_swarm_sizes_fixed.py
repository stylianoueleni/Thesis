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

swarm_sizes = list(range(2, 11))  # 2 to 10 agents
max_steps = 400
seed = 42
num_eval_episodes = 100
num_obstacles = 8
spawn_spread = 3.0
out_dir = "swarm_comparison_fixed"

eps_start = 1.0
eps_min = 0.01
eps_decay = 0.000042
LR = 0.001
L2 = 1e-4
GAMMA = 0.99
batch_size = 64
buffer_cap = 100000
target_update_freq = 4
hidden_size = 128

SWARM_CFG = {
    #  N : (episodes, sep_w, coh_w, align_w)
    2  : (800,       4.0,   0.30,  0.8),
    3  : (800,       4.0,   0.30,  0.8),
    4  : (800,       4.0,   0.30,  0.8),
    5  : (800,       4.0,   0.30,  0.8),
    6  : (800,       4.0,   0.30,  0.8),
    7  : (800,       4.0,   0.30,  0.8),
    8  : (800,       4.0,   0.30,  0.8),
    9  : (800,       4.0,   0.30,  0.8),
    10 : (800,       4.0,   0.30,  0.8),
}

def _cfg(n):
    return SWARM_CFG[n]

desired_spacing = 6.0

_world_half = 50.0
_max_dist = np.sqrt(2) * 2 * _world_half
_max_heading = np.pi
_max_range = 30.0
_n_ranges = 5

def agent_path(n: int) -> str:
    return f"agent_DQN_{n}uav_fixed.npy"

def build_env(num_uavs: int, num_obstacles: int= num_obstacles, seed: int | None = None) -> SwarmUAVEnv:
    return SwarmUAVEnv(num_uavs = num_uavs,
                       num_obstacles = num_obstacles,
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

def path_length(centroid_traj: list) -> float:
    if len(centroid_traj) < 2:
        return 0.0
    arr = np.array(centroid_traj)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

def train_one(num_uavs: int) -> tuple[list, list]:
    pth = agent_path(num_uavs)
    max_episodes, sep_w, coh_w, align_w = _cfg(num_uavs)
    n_neighbours  = max(1, num_uavs - 1)
    sep_w_normed  = sep_w / n_neighbours

    print(f"\n{'='*62}")
    print(f"  Training  DQN  ·  {num_uavs} UAVs  ·  {max_episodes} episodes")
    print(f"  sep_w={sep_w_normed:.3f} (raw={sep_w}/N-1)  "
          f"coh_w={coh_w}  align_w={align_w}")
    print(f"  eps_decay={eps_decay}  eps_min={eps_min}")
    print(f"{'='*62}")

    env = build_env(num_uavs, seed=seed)
    o_dim = env.reset().shape[1]

    agent = DQNAgent(
        obs_dim = o_dim,
        num_actions = env.num_actions,
        hidden= hidden_size,
        lr = LR,
        l2 = L2,
        gamma = GAMMA,
        batch_size = batch_size,
        buffer_capacity = buffer_cap,
        target_update_freq = target_update_freq,
        eps_start = eps_start,
        eps_min = eps_min,
        eps_decay = eps_decay,
        seed = seed,
    )

    shaper = SwarmRewardShaper(
        num_uavs = num_uavs,
        desired_spacing = desired_spacing,
        cohesion_weight = coh_w,
        separation_weight = sep_w_normed,
        alignment_weight = align_w,
        separation_min = env.uav_body_radius * 2.0 + env.uav_separation,
    )

    ep_rewards: list[float] = []
    ep_steps: list[int] = []
    best_r = -np.inf
    t0 = time.time()

    for episode in range(1, max_episodes + 1):
        env = build_env(num_uavs, seed=seed)  
        obs = normalise_obs(env.reset())
        dones = np.zeros(num_uavs, dtype=bool)
        ep_r = 0.0
        step = 0
        while not np.all(dones) and step < max_steps:
            actions = agent.select_actions(obs)
            next_obs, base_rew, dones, _ = env.step(actions)
            next_obs = normalise_obs(next_obs)
            shape_r = shaper.shape(env.pos, env.heading, dones)
            rewards = base_rew + shape_r

            agent.store_swarm(obs, actions, rewards, next_obs, dones)
            agent.learn()
            agent.decay_epsilon()

            obs = next_obs
            ep_r += rewards.mean()
            step += 1

        ep_rewards.append(ep_r)
        ep_steps.append(step)

        if episode % 50 == 0 or episode == 1:
            avg20 = np.mean(ep_rewards[-20:])
            print(f"  Ep {episode:>4}/{max_episodes} | "
                  f"steps={step:>4} | r={ep_r:>9.2f} | "
                  f"avg20={avg20:>9.2f} | ε={agent.eps:.4f} | "
                  f"t={time.time()-t0:.1f}s")
            
        if ep_r > best_r:
            best_r = ep_r
            agent.save(pth)

    print(f"  Done. Best reward={best_r:.2f}  →  {pth}")
    return ep_rewards, ep_steps

def evaluate_agent(num_uavs: int, n_episodes: int = num_eval_episodes) -> dict:
    pth = agent_path(num_uavs)
    env = build_env(num_uavs, seed=seed)
    obs_dim = env.reset().shape[1]

    agent = DQNAgent(obs_dim=obs_dim, num_actions=env.num_actions, hidden=hidden_size)
    agent.load(pth)
 
    rewards, lengths, steps_arr, smoothness, energy, successes = (
        [], [], [], [], [], []
    )

    for ep in range(n_episodes):
        obs   = normalise_obs(env.reset())
        dones = np.zeros(num_uavs, dtype=bool)
 
        centroid_hist = [env.pos.mean(axis=0).copy()]
        heading_hist  = [env.heading.copy()]
        total_rew     = 0.0
        step          = 0
        while not np.all(dones) and step < max_steps:
            actions = agent.select_actions(obs, greedy=True)
            obs_raw, rew, dones, info = env.step(actions)
            obs       = normalise_obs(obs_raw)
            total_rew += rew.mean()
            step      += 1
            centroid_hist.append(env.pos.mean(axis=0).copy())
            heading_hist.append(env.heading.copy())

        rewards.append(total_rew)
        lengths.append(path_length(centroid_hist))
        steps_arr.append(step)
 
        h_arr = np.array([h.mean() for h in heading_hist])
        dh    = np.abs(np.diff(h_arr)) if len(h_arr) > 1 else np.array([0.0])
        smoothness.append(float(dh.mean()))
        energy.append(float(dh.sum()))
 
        succeeded = any(
            "reached the goal" in ev for ev in info.get("events", [])
        )
        successes.append(succeeded)

    s_idx = np.where(successes)[0]
    def ms(arr):
        a = np.array(arr)
        return float(a[s_idx].mean()) if len(s_idx) > 0 else float(a.mean())
 
    return {
        "num_uavs"       : num_uavs,
        "successRate"    : float(np.mean(successes)),
        "meanReward"     : float(np.mean(rewards)),
        "meanPathLength" : ms(lengths),
        "meanSteps"      : ms(steps_arr),
        "meanSmoothness" : ms(smoothness),
        "meanEnergy"     : ms(energy),
        "allRewards"     : np.array(rewards),
        "allSuccesses"   : np.array(successes),
    }

def episodes_to_converge(ep_rewards: list, threshold_pct: float = 0.80, window: int = 20) -> int:
    if len(ep_rewards) < window:
        return 0
    n_eps = len(ep_rewards)
    rewards = np.array(ep_rewards)
    smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    peak = smooth.max()
    if peak <= 0:
        return n_eps
    target = threshold_pct * peak
    hits = np.where(smooth >= target)[0]
    if len(hits) == 0:
        return n_eps
    return int(hits[0]) + window

PALETTE = cm.tab10(np.linspace(0, 1, len(swarm_sizes)))

def _save(fig, name: str):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")

def fig1_training_reward_curves(training_history: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 20
    for col, n in zip(PALETTE, swarm_sizes):
        if n not in training_history:
            continue
        ep_r = np.array(training_history[n]["ep_rewards"])
        eps = np.arange(1, len(ep_r) + 1)
        ax.plot(eps, ep_r, color=col, alpha=0.15, lw=0.8)
        if len(ep_r) >= w:
            smooth = np.convolve(ep_r, np.ones(w)/w, mode='valid')
            ax.plot(np.arange(w, len(ep_r) + 1), smooth, color=col, lw=2.0,
                    label=f"{n} UAVs")


    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean swarm reward / step", fontsize=12)
    ax.set_title("Training reward curves", fontsize=14, fontweight="bold")
    ax.legend(ncol=3, fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "fig1_training_reward_curves.png")

def fig2_training_steps_curves(training_history: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 20
    for col, n in zip(PALETTE, swarm_sizes):
        if n not in training_history:
            continue
        ep_steps = np.array(training_history[n]["ep_steps"])
        eps = np.arange(1, len(ep_steps) + 1)
        ax.plot(eps, ep_steps, color=col, alpha=0.25, lw=0.7)
        if len(ep_steps) >= w:
            smooth = np.convolve(ep_steps, np.ones(w)/w, mode='valid')
            ax.plot(np.arange(w, len(ep_steps) + 1), smooth, color=col, lw=2.0,
                    label=f"{n} UAVs")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps", fontsize=12)
    ax.set_title("Training episode lengths", fontsize=14, fontweight="bold")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, "fig2_training_steps_curves.png")

def fig3_success_vs_size(metrics_list: list):
    ns = [m["num_uavs"] for m in metrics_list]
    succs = [m["successRate"]*100 for m in metrics_list]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ns, succs, color=PALETTE[:len(ns)], edgecolor="white", width=0.65, zorder=4)
    ax.plot(ns, succs, "-o", color="black", lw=1.5, markersize=6, zorder=4)

    for bar, v in zip(bars, succs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Swarm size (number of UAVs)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Success rate vs. swarm size", fontsize=14, fontweight="bold")
    ax.set_xticks(ns)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig3_success_vs_size.png")
    
def fig4_path_length_vs_size(metrics_list: list):
    ns = [m["num_uavs"] for m in metrics_list]
    pls = [m["meanPathLength"] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ns, pls, color=PALETTE[:len(ns)], edgecolor="white", width=0.65, zorder=3)
    for bar, v in zip(bars, pls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pls)*0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Swarm size (number of UAVs)", fontsize=12)
    ax.set_ylabel("Mean path length (m)", fontsize=12)
    ax.set_title("Mean path length vs. swarm size", fontsize=14, fontweight="bold")
    ax.set_xticks(ns)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig4_path_length_vs_size.png")

def fig5_smoothness_vs_size(metrics_list: list):
    ns = [m["num_uavs"] for m in metrics_list]
    vals = [m["meanSmoothness"] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ns, vals, color=PALETTE[:len(ns)], edgecolor="white", width=0.65, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Swarm size (number of UAVs)", fontsize=12)
    ax.set_ylabel("Mean smoothness (rad)", fontsize=12)
    ax.set_title("Mean smoothness vs. swarm size", fontsize=14, fontweight="bold")
    ax.set_xticks(ns)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig5_smoothness_vs_size.png")

def fig6_energy_vs_size(metrics_list: list):
    ns = [m["num_uavs"] for m in metrics_list]
    vals = [m["meanEnergy"] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ns, vals, color=PALETTE[:len(ns)], edgecolor="white", width=0.65, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Swarm size (number of UAVs)", fontsize=12)
    ax.set_ylabel("Mean energy (rad)", fontsize=12)
    ax.set_title("Mean energy vs. swarm size", fontsize=14, fontweight="bold")
    ax.set_xticks(ns)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig6_energy_vs_size.png")

def fig7_convergence_vs_size(training_history: dict):
    ns = []
    eps_to_conv = []
    for n in swarm_sizes:
        if n not in training_history:
            continue
        ep_rewards = training_history[n]["ep_rewards"]
        if len(ep_rewards) < 20:
            continue
        ns.append(n)
        eps_to_conv.append(episodes_to_converge(ep_rewards))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ns, eps_to_conv, color=PALETTE[:len(ns)], edgecolor="white", width=0.65, zorder=3)
    for bar, v in zip(bars, eps_to_conv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Swarm size (number of UAVs)", fontsize=12)
    ax.set_ylabel("Episodes to converge", fontsize=12)
    ax.set_title("Episodes to converge vs. swarm size", fontsize=14, fontweight="bold")
    ax.set_xticks(ns)
    ax.set_ylim(0, max(eps_to_conv, default=800) + 20)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig7_convergence_vs_size.png")

def fig8_heatmap_all_metrics(metrics_list: list):
    metric_keys = ["successRate", "meanPathLength", "meanSmoothness", "meanEnergy", "meanSteps"]
    metric_labels = ["Success rate (%)", "Path length (m)", "Smoothness (rad)", "Energy (rad)", "Steps"]
    ivert = [False, True, True, True, True] 
    ns = [m["num_uavs"] for m in metrics_list]
    raw = np.array([[m[k] for m in metrics_list] for k in metric_keys], dtype=float)

    normed = np.zeros_like(raw)
    for i, inv in enumerate(ivert):
        lo, hi = raw[i].min(), raw[i].max()
        if hi == lo:
            normed[i] = 0.5
        else:
            n = (raw[i] - lo) / (hi - lo)
            normed[i] = (1.0 - n) if inv else n

    fig, ax = plt.subplots(figsize=(max(8, len(ns) * 0.9 + 3), 5))
    im = ax.imshow(normed, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised score")

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"N={n}" for n in ns], fontsize=10)
    ax.set_yticks(range(len(metric_keys)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    ax.set_title("Comparison of all metrics across swarm sizes", fontsize=13, fontweight="bold")

    for i in range(len(metric_keys)):
        for j in range(len(ns)):
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color="black")

    _save(fig, "fig8_heatmap_all_metrics.png")

def fig9_radar_overlay(metrics_list: list):
    metric_keys = ["successRate", "meanPathLength", "meanSmoothness", "meanEnergy", "meanSteps"]
    metric_labels = ["Success\nrate", "Path\nefficiency", "Smoothness", "Energy\neff.", "Speed"]
    ivert = [False, True, True, True, True] 
    raw = np.array([[m[k] for k in metric_keys] for m in metrics_list], dtype=float)

    normed = np.zeros_like(raw)
    for j, inv in enumerate(ivert):
        lo, hi = raw[:, j].min(), raw[:, j].max()
        if hi == lo:
            normed[:, j] = 0.5
        else:
            n = (raw[:, j] - lo) / (hi - lo)
            normed[:, j] = (1.0 - n) if inv else n

    num_vars = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for col, row_idx, m in zip(PALETTE[:len(metrics_list)], range(len(metrics_list)), metrics_list):
        values = normed[row_idx].tolist() + [normed[row_idx][0]]
        ax.plot(angles, values, color=col, linewidth=2, label=f"N={m['num_uavs']} UAVs")
        ax.fill(angles, values, color=col, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title("Radar plot of metrics across swarm sizes", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=13, ncol=2)
    _save(fig, "fig9_radar_overlay.png")

def fig_summary_panel(metrics_list: list):
    ns   = np.array([m["num_uavs"] for m in metrics_list])
    succ = np.array([m["successRate"] * 100 for m in metrics_list])
    path = np.array([m["meanPathLength"] for m in metrics_list])
    smth = np.array([m["meanSmoothness"] for m in metrics_list])
    enrg = np.array([m["meanEnergy"] for m in metrics_list])
    stp  = np.array([m["meanSteps"] for m in metrics_list])
    rewd = np.array([m["meanReward"] for m in metrics_list]) 

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("UAV Swarm Comparison (N= 2 to 10) · DQN Agent", fontsize=15, fontweight="bold", y=1.01)

    def _bar(ax, y, ylabel, title, colour="steelblue", line=True):
        ax.bar(ns, y, color=PALETTE[:len(ns)], edgecolor="white",
               width=0.65, zorder=3)
        if line:
            ax.plot(ns, y, "-o", color="black", lw=1.3, ms=5, zorder=4)
        ax.set_xticks(ns)
        ax.set_xlabel("Swarm size N", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    _bar(axes[0, 0], succ, "Success rate (%)", "Success rate")
    _bar(axes[0, 1], path, "Mean path length (m)", "Mean path length")
    _bar(axes[0, 2], smth, "|Δhdg| (rad/step)", "Path Smoothness")
    _bar(axes[1, 0], enrg, "Σ|Δhdg| (rad)", "Energy Proxy")
    _bar(axes[1, 1], stp, "Steps", "Episode length")
    _bar(axes[1, 2], rewd, "Reward / step", "Mean episode reward")

    plt.tight_layout()
    _save(fig, "fig_summary_panel.png")

def main():
    os.makedirs(out_dir, exist_ok=True)
    history_cache = os.path.join(out_dir, "training_history.npy")
    metrics_cache = os.path.join(out_dir, "evaluation_metrics.npy")

    print("\n" + "=" * 62)
    print("UAV Swarm DQN - Comparison (N = 2 to10)")
    print("=" * 62)

    if os.path.exists(history_cache):
        print(f"\nLoading training history from cache → {history_cache}")
        training_history = np.load(history_cache, allow_pickle=True).item()
    else:
        training_history = {}

    for n in swarm_sizes:
        pth = agent_path(n)
        if os.path.exists(pth) and n in training_history:
            print(f"\n  Found existing model for {n} UAVs → {pth}")
            continue
        elif os.path.exists(pth) and n not in training_history:
            print(f"\n  Found model file for {n} UAVs but no history entry")
            training_history[n] = {"ep_rewards": [], "ep_steps": []}
            continue 
        ep_r, ep_s = train_one(n)
        training_history[n] = {"ep_rewards": ep_r, "ep_steps": ep_s}

    np.save(history_cache, training_history)
    print(f"\nTraining history saved → {history_cache}")

    if os.path.exists(metrics_cache):
        print(f"\nLoading evaluation metrics from cache → {metrics_cache}")
        metrics_list = list(np.load(metrics_cache, allow_pickle=True))
    else:
        metrics_list = []
        evaluated_ns = {m["num_uavs"] for m in metrics_list}
        print(f"\n  Evaluating agents  ({num_eval_episodes} greedy eps each) …")
        for n in swarm_sizes:
            if n in evaluated_ns:
                print(f"  Already have metrics for {n} UAVs, skipping evaluation")
                continue
            if not os.path.exists(agent_path(n)):
                print(f"  No model found for {n} UAVs, skipping evaluation")
                continue
            print(f"[N={n:2d}] evaluating …", end=" ", flush=True)
            t0 = time.time()
            m = evaluate_agent(n, n_episodes=num_eval_episodes)
            metrics_list.append(m)
            print(f"  Done in {time.time() - t0:.1f}s | Success rate: {m['successRate']*100:.1f}% | Path length: {m['meanPathLength']:.0f}")
        metrics_list.sort(key=lambda x: x["num_uavs"])
        np.save(metrics_cache, metrics_list)
        print(f"\nEvaluation metrics saved → {metrics_cache}")
    
    if not metrics_list:
        print("\nNo evaluation metrics available, cannot generate figures.")
        return

    print("\n  Generating figures …")
    fig1_training_reward_curves(training_history)
    fig2_training_steps_curves(training_history)
    fig3_success_vs_size(metrics_list)
    fig4_path_length_vs_size(metrics_list)
    fig5_smoothness_vs_size(metrics_list)
    fig6_energy_vs_size(metrics_list)
    fig7_convergence_vs_size(training_history)
    fig8_heatmap_all_metrics(metrics_list)
    fig9_radar_overlay(metrics_list)
    fig_summary_panel(metrics_list)

    print("\n" + "=" * 75)
    print(f"{'N':>3} {'Success':>9} {'Path(m)':>9}  "
         f"{'Smooth':>9} {'Energy':>9} {'Steps':>7} {'Reward':>9}")
    print("  " + "-" * 71)
    for m in metrics_list:
        print(f"{m['num_uavs']:>3}"
              f"{m['successRate']*100:>8.1f}%"
              f"{m['meanPathLength']:>9.1f}"
              f"{m['meanSmoothness']:>9.5f}"
              f"{m['meanEnergy']:>9.3f}"
              f"{m['meanSteps']:>7.1f}"
              f"{m['meanReward']:>9.2f}")
    print("=" * 75)
 
    all_figs = [
        "fig1_training_reward_curves.png",
        "fig2_training_steps_curves.png",
        "fig3_success_vs_size.png",
        "fig4_path_length_vs_size.png",
        "fig5_smoothness_vs_size.png",
        "fig6_energy_vs_size.png",
        "fig7_convergence_vs_size.png",
        "fig8_heatmap_all_metrics.png",
        "fig9_radar_overlay.png",
        "fig_summary_panel.png",
    ]
    print(f"\n  Output folder: {out_dir}/")
    for f in all_figs:
        p = os.path.join(out_dir, f)
        print(f" {'✓' if os.path.exists(p) else '✗'}  {f}")

if __name__ == "__main__":
    main()