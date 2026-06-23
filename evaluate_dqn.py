import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uavswarm_env_topology import SwarmUAVEnv
from dqn_agent import DQNAgent
from uav_env_topology import UAVEnv

num_uavs = 5
num_episodes = 300
obstacle_scenarios = [4, 6, 8]
seed_base = 0
out_folder = "evaluation_results_dqn"
summary_path = os.path.join(out_folder, "results_summary.npy")

dqn_path = "agent_DQN_swarm.npy"

ENV_BASE = dict(
    num_uavs = num_uavs,
    world_bounds = ((-50, 50), (-50, 50)),
    start_pos = (-40, -40),
    goal_pos = (40, 40),
    min_radius = 3.0,
    max_radius = 8.0,
    obstacle_clearance = 1.0,
    start_goal_clearance = 6.0,
    speed = 3.0,
    goal_radius = 3.0,
    detection_radius = 30.0,
    uav_separation = 2.0,
    spawn_spread = 3.0,
    dt = 0.5,
    max_steps = 400,
)

_world_half = 50.0
_max_dist = np.sqrt(2) * 2 * _world_half
_max_heading = np.pi
_max_range = 30.0
_n_ranges = 5

def build_env(num_obstacles: int, seed: int = None) -> SwarmUAVEnv:
    return SwarmUAVEnv(num_obstacles=num_obstacles, seed=seed, **ENV_BASE)

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

def _path_length(positions: list) -> float:
    """Sum of Euclidean distances between consecutive centroid positions."""
    if len(positions) < 2:
        return 0.0
    arr = np.array(positions)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

def load_agent(path: str, obs_dim: int, num_actions: int) -> DQNAgent:
    agent = DQNAgent(obs_dim=obs_dim, num_actions=num_actions, hidden=128)
    agent.load(path)
    return agent

def evaluate_agent(
        agent: DQNAgent,
        num_obs: int,
        n_episodes: int,
        energy_tracker_cls = None,
) -> dict:
    rewards = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes)
    successes = np.zeros(n_episodes, dtype=bool)
    times = np.zeros(n_episodes)
    smoothness_vals = np.zeros(n_episodes)
    energy_vals = np.zeros(n_episodes)
    trajectories = []

    inference_time = []

    for ep in range(n_episodes):
        env = build_env(num_obs, seed = seed_base + ep)
        obs_raw = env.reset()
        obs = normalise_obs(obs_raw)
        dones = np.zeros(num_uavs, dtype=bool)

        centroid_traj = [env.pos.mean(axis=0).copy()]
        headings_hist = [env.heading.copy()]

        step = 0
        total_rew = 0.0

        while not np.all(dones) and step < env.max_steps:
            t0 = time.perf_counter()
            actions = agent.select_actions(obs, greedy=True)
            t1 = time.perf_counter()
            inference_time.append((t1 - t0) * 1000)  # ms
            obs_raw, rew, dones, info = env.step(actions)
            obs = normalise_obs(obs_raw)

            total_rew += rew.mean()
            step += 1
            centroid_traj.append(env.pos.mean(axis=0).copy())
            headings_hist.append(env.heading.copy())
        
        rewards[ep] = total_rew
        lengths[ep] = _path_length(centroid_traj)
        times[ep] = step * env.dt

        h_arr = np.array([h.mean() for h in headings_hist])
        if len(h_arr) > 1:
            dh = np.abs(np.diff(h_arr))
            smoothness_vals[ep] = float(dh.mean())
            energy_vals[ep] = float(dh.sum())

        successes[ep] = any(
            "reached the goal" in ev for ev in info.get("events", []))
        trajectories.append(np.array(centroid_traj))

    succ_idx = np.where(successes)[0]

    def _ms(arr):
        return float(arr[succ_idx].mean()) if len(succ_idx) > 0 else float(arr.mean())
    
    inf_arr = np.array(inference_time)

    return {
        "rewards": rewards,
        "successRate": float(successes.mean()),
        "meanReward": float(rewards.mean()),
        "meanPathLength": _ms(lengths),
        "meanTime": _ms(times),
        "meanSmoothness": _ms(smoothness_vals),
        "meanEnergy": _ms(energy_vals),
        "trajectories": trajectories,
        "meanInferenceTime": float(inf_arr.mean()),
        "medianInferenceTime": float(np.median(inf_arr)),
        "allInferenceTimes": inference_time,
    }

def evaluate_agent_custom(agent, env_factory, num_obs, n_episodes, n_uav):
    rewards = []; lengths = []; successes = []; smoothness_vals = []; energy_vals = []
    for ep in range(n_episodes):
        env =  env_factory(num_obs, seed = seed_base + ep)
        obs = normalise_obs(env.reset())
        dones = np.zeros(n_uav, dtype=bool)
        centroid_traj = [env.pos.mean(axis=0).copy()]
        headings_hist = [env.heading.copy()]
        step = 0; total_rew = 0.0
        while not np.all(dones) and step < env.max_steps:
            actions = agent.select_actions(obs, greedy=True)
            obs_raw, rew, dones, info = env.step(actions)
            obs = normalise_obs(obs_raw)
            total_rew += rew.mean()
            step += 1
            centroid_traj.append(env.pos.mean(axis=0).copy())
            headings_hist.append(env.heading.copy())
        rewards.append(total_rew)
        lengths.append(_path_length(centroid_traj))
        h = np.array([h.mean() for h in headings_hist])
        dh = np.abs(np.diff(h)) if len(h) > 1 else np.array([0.])
        smoothness_vals.append(float(dh.mean()))
        energy_vals.append(float(dh.sum()))
        successes.append(any("reached the goal" in ev for ev in info.get("events", [])))
    
    s = np.where(successes)[0]
    ms = lambda a: float(np.array(a)[s].mean()) if len(s) > 0 else float(np.array(a).mean())
    return {    "successRate": np.mean(successes),
                "meanPathLength": ms(lengths),
                "meanSmoothness": ms(smoothness_vals),
                "meanEnergy": ms(energy_vals) }

def build_single_env(num_obstacles: int, seed: int = None) -> UAVEnv:
    return UAVEnv(
        num_obstacles=num_obstacles,
        world_bounds=ENV_BASE["world_bounds"],
        start_pos=ENV_BASE["start_pos"],
        goal_pos=ENV_BASE["goal_pos"],
        min_radius=ENV_BASE["min_radius"],
        max_radius=ENV_BASE["max_radius"],
        obstacle_clearance=ENV_BASE["obstacle_clearance"],
        start_goal_clearance=ENV_BASE["start_goal_clearance"],
        speed=ENV_BASE["speed"],
        seed=seed,
        goal_radius=ENV_BASE["goal_radius"],
        max_range=ENV_BASE["detection_radius"],
        dt=ENV_BASE["dt"],
        max_steps=ENV_BASE["max_steps"],
        )

def evaluate_single_agent(agent: DQNAgent, num_obs: int, n_episodes: int)-> dict:
    rewards = []
    lengths = []
    successes = []
    smoothness_vals = []
    energy_vals = []
    for ep in range(n_episodes):
        env = build_single_env(num_obs, seed=seed_base + ep)
        obs = normalise_obs(env.reset()[np.newaxis, :])[0]
        done = False
        traj = [env.pos.copy()]
        headings = [env.heading.copy()]
        total_rew = 0.0
        while (not done) and env.step_count < env.max_steps:
            action = agent.select_action(obs, greedy=True)
            obs_raw, rew, done, info = env.step(action)
            obs = normalise_obs(obs_raw[np.newaxis, :])[0]
            total_rew += rew
            traj.append(env.pos.copy())
            headings.append(env.heading)
        rewards.append(total_rew)
        lengths.append(_path_length(traj))
        h = np.array(headings, dtype=np.float64)
        dh = np.abs(np.diff(h)) if len(h) > 1 else np.array([0.])
        smoothness_vals.append(float(dh.mean()))
        energy_vals.append(float(dh.sum()))
        successes.append(info.get("reason") == "goal_reached")
    
    succ_idx = np.where(successes)[0]
    def _ms(arr):
        return float(np.array(arr)[succ_idx].mean()) if len(succ_idx) > 0 else float(np.array(arr).mean())
    
    return {
        "successRate": float(np.mean(successes)),
        "meanReward": float(np.mean(rewards)),
        "meanPathLength": _ms(lengths),
        "meanSmoothness": _ms(smoothness_vals),
        "meanEnergy": _ms(energy_vals),
    }

def fig_single_vs_swarm_bars(single_metrics, swarm_metrics, out):
    labels = ["Success Rate (%)", "Path Length (m)", "Smoothness (rad)", "Energy"]
    single_vals = [
        single_metrics["successRate"] * 100,
        single_metrics["meanPathLength"],
        single_metrics["meanSmoothness"],
        single_metrics["meanEnergy"],
    ]
    swarm_vals = [
        swarm_metrics["successRate"] * 100,
        swarm_metrics["meanPathLength"],
        swarm_metrics["meanSmoothness"],
        swarm_metrics["meanEnergy"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10,7))
    axes = axes.ravel()

    for i, (lab, s1, s5) in enumerate(zip(labels, single_vals, swarm_vals)):
        ax = axes[i]
        ax.bar(["Single UAV", "5 UAVs"], [s1, s5], color=["steelblue", "darkorange"])
        ax.set_title(lab)
        ax.grid(axis="y", alpha=0.3)
        for j, v in enumerate([s1, s5]):
            ax.text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    
    fig.suptitle("DQN Performance: Single UAV vs 5-UAV Swarm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_single_vs_swarm_bars.png"), dpi=120)
    plt.close(fig)

def fig_single_vs_swarm_tradeoff(single_metrics, swarm_metrics, out):
    fig, ax = plt.subplots(figsize=(6,5))

    x = [single_metrics["meanPathLength"], swarm_metrics["meanPathLength"]]
    y = [single_metrics["meanSmoothness"], swarm_metrics["meanSmoothness"]]

    ax.scatter(x, y, s=120)
    ax.annotate("Single UAV", (x[0] + 0.3, y[0]), fontsize=10)
    ax.annotate("5 UAVs", (x[1] + 0.3, y[1]), fontsize=10)

    ax.set_title("Single UAV vs 5-UAV Swarm: Path-Smoothness Profile")
    ax.set_xlabel("Path Length (m) - lower is more efficient")
    ax.set_ylabel("Mean Smoothness (rad) - lower is smoother")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_single_vs_swarm_tradeoff.png"), dpi=120)
    plt.close(fig)

def run_single_vs_swarm_comparison(out):
    single_path = "agent_DQN_single.npy"
    swarm_path = "agent_DQN_swarm.npy"

    if not os.path.exists(single_path) or not os.path.exists(swarm_path):
        print("SKIP single vs swarm comparison - missing model file(s)")
        print(f"  Expected: {single_path} and {swarm_path}")
        return
    
    print("\n=== Single UAV vs 5-UAV Swarm Comparison (fixed 4 obs, 200 eps each) ===")
    
    env_single = build_single_env(num_obstacles=4, seed=0)
    agent_single = load_agent(single_path, env_single.reset().shape[0], num_actions=env_single.num_actions)
    m_single = evaluate_single_agent(agent_single, num_obs=4, n_episodes=200)

    env_swarm = build_env(num_obstacles=4, seed=0)
    agent_swarm = load_agent(swarm_path, env_swarm.reset().shape[1], num_actions=env_swarm.num_actions)
    m_swarm = evaluate_agent(agent_swarm, num_obs=4, n_episodes=200)

    print(f"  1 UAV  | Success={m_single['successRate']*100:.1f}% "
          f"| Path={m_single['meanPathLength']:.2f}m "
          f"| Smooth={m_single['meanSmoothness']:.4f} "
          f"| Energy={m_single['meanEnergy']:.4f}")

    print(f"  5 UAVs | Success={m_swarm['successRate']*100:.1f}% "
          f"| Path={m_swarm['meanPathLength']:.2f}m "
          f"| Smooth={m_swarm['meanSmoothness']:.4f} "
          f"| Energy={m_swarm['meanEnergy']:.4f}")
    
    fig_single_vs_swarm_bars(m_single, m_swarm, out)
    fig_single_vs_swarm_tradeoff(m_single, m_swarm, out)

def fig_uav_comparison(uav_counts, succ, paths, smooth, energy, out):
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    colors = ["steelblue", "darkorange", "green"]
    labels = [str(n) for n in uav_counts]

    for ax, vals, title, ylabel in zip(axes, [succ, paths, smooth, energy],
                                 ["Success Rate", "Path Length", "Smoothness", "Energy"],
                                 ["%", "m", "rad", "rad"]):
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Number of UAVs")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01*max(vals), f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("DQN Performance vs Swarm Size (4 obstacles)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_uav_comparison.png"), dpi=120)
    plt.close(fig)

def _bar_grouped(scenarios, vals_dqn, xlabel, ylabel, title, fname, out):
        fig, ax = plt.subplots(figsize=(7,4))
        x = np.arange(len(scenarios))
        w = 0.35
        ax.bar(x, vals_dqn, w, label="DQN", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in scenarios])
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, fname), dpi=120)
        plt.close(fig)

def _radar_scenario(m_dqn, n_obs, fname, out):
    labels = ["Success", "1/Path", "1/Smooth", "1/Energy"]
    n = len(labels)

    def _safe_inv(x):
        return 1.0 / x if x > 1e-9 else 0.0

    dqn_raw = np.array([m_dqn["successRate"], _safe_inv(m_dqn["meanPathLength"]), _safe_inv(m_dqn["meanSmoothness"]), _safe_inv(m_dqn["meanEnergy"])])
    
    ref = np.array([1.0, 1/30.0, 1/0.03, 1/3.0])
    ref[ref < 1e-9] = 1.0
    dqn_n = np.clip(dqn_raw / ref, 0.0, 1.0).tolist()

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    dqn_n += dqn_n[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    ax.plot(angles, dqn_n,   "-ob", linewidth=1.5, label="DQN", markersize=5)
    ax.fill(angles, dqn_n,   alpha=0.10, color="blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Scenario (obs={n_obs})", fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out, fname), dpi=120)
    plt.close(fig)

def fig_success_line(scenarios, succ_dqn, out):
    """line chart - success rate vs obstacle count"""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(scenarios, succ_dqn, "-ob", linewidth=1.8, markerfacecolor="b", label="DQN")
    ax.set_xlabel("Number of Obstacles")
    ax.set_ylabel("Success Rate(%)")
    ax.set_title("Goal-Reaching Success Across Scenarios")
    ax.set_ylim(0, 100)
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    ax.set_xticks(scenarios)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig1_success_line.png"), dpi=120)
    plt.close(fig)

def fig_heatmap(scenarios, succ_dqn, paths_dqn, smooth_dqn, energy_dqn, out):
    """heatmap - performance overview"""
    def _safe_inv(arr):
        return np.where(arr > 1e-9, 1.0 / arr, 0.0)
    
    raw = np.array([
        succ_dqn,
        _safe_inv(np.array(paths_dqn)),
        _safe_inv(np.array(smooth_dqn)),
        _safe_inv(np.array(energy_dqn)),
    ])

    row_min = raw.min(axis=1, keepdims=True)
    row_max = raw.max(axis=1, keepdims=True)
    row_range = np.where((row_max - row_min) > 1e-9, row_max - row_min, 1.0)
    matrix_norm = (raw - row_min) / row_range

    row_labels = ["Succ DQN", "1/Path DQN", "1/Smooth DQN", "1/Energy DQN"]
    col_labels = [f"{s} obs" for s in scenarios]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalized (per row)")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_title("DQN Performance Overview Heatmap (row-normalized)")

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax.text(j, i, f"{raw[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig6_heatmap.png"), dpi=120)
    plt.close(fig)

def fig_tradeoff_scatter(scenarios, paths_dqn, smooth_dqn, out):
    """scatter - path length vs smoothness tradeoff"""
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(paths_dqn, smooth_dqn, s=80, c="steelblue", zorder=5, label="DQN")
    for i, s in enumerate(scenarios):
        ax.annotate(str(s), (paths_dqn[i]   + 0.5, smooth_dqn[i] + 0.0002),   color="steelblue", fontsize=10)
    ax.set_xlabel("Path Length (m)")
    ax.set_ylabel("Mean Smoothness (rad)")
    ax.set_title("Path Length vs Smoothness Tradeoff")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out,"fig7_tradeoff_scatter.png"), dpi=120)
    plt.close(fig)

def fig_success_line_extended(scenarios_ext, succ_ext, out):
    """line chart - success rate vs obstacle count (100 episodes each])"""
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(scenarios_ext, succ_ext, "-ob", linewidth=2, markersize=7, markerfacecolor="b", label="DQN (100 eps each)")
    ax.set_xlabel("Number of Obstacles")
    ax.set_ylabel("Success Rate(%)")
    ax.set_title("DQN Success Rate vs Obstacle Count (1-10)")
    ax.set_ylim(0, 100)
    ax.set_xticks(scenarios_ext)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_success_1to10.png"), dpi=120)
    plt.close(fig)
    print(f" [fig] Success line 1-10 -> {out}/fig_success_1to10.png")

def run_uav_comparison(out):
    """
    UAV count comparison: 3 vs 4 vs 5 UAVs.
    Requires separately trained agents:
      agent_DQN_three.npy  (train with num_uavs=3)
      agent_DQN_four.npy  (train with num_uavs=4)
      agent_DQN_swarm.npy (already done, num_uavs=5)

    """
    uav_paths_map = { 3: "agent_DQN_three.npy", 4: "agent_DQN_four.npy", 5: "agent_DQN_swarm.npy" }
    uav_counts = [3, 4, 5]
    uav_succ = [];
    uav_paths = [];
    uav_smooth =[];
    uav_energy = []
    print("\n=== UAV Count Comparison (fixed 4 obs, 200 eps each) ===")
    for n_uav in uav_counts:
        agent_path = uav_paths_map[n_uav]
        if not os.path.exists(agent_path):
            print(f"SKIP {n_uav} UAVs -- {agent_path} not found")
            continue
        env_kwargs = {**ENV_BASE, "num_uavs": n_uav}
        def build_env_nuav(num_obstacles, seed=None, kw=env_kwargs):
            return SwarmUAVEnv(num_obstacles=num_obstacles, seed=seed, **kw)
        env_tmp = build_env_nuav(4, seed=0)
        obs_dim_n = env_tmp.reset().shape[1]
        agent_n = load_agent(agent_path, obs_dim_n, env_tmp.num_actions)
        m = evaluate_agent_custom(agent_n, build_env_nuav, 4, 200, n_uav)
        uav_succ.append(m["successRate"]*100)
        uav_paths.append(m["meanPathLength"])
        uav_smooth.append(m["meanSmoothness"])
        uav_energy.append(m["meanEnergy"])
        print(f" {n_uav} UAVs | Success={uav_succ[-1]:.1f}% "
              f"Path={uav_paths[-1]:.1f}m Smooth={uav_smooth[-1]:.4f}")
    
    if len(uav_succ) == len(uav_counts):
        fig_uav_comparison(uav_counts, uav_succ, uav_paths, uav_smooth, uav_energy, out)


def main():
    t_total_start = time.time()
    os.makedirs(out_folder, exist_ok=True)
 
    for p in [dqn_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p} — run train_dqn_swarm.py / "
            )
    
 
    env_probe = build_env(4, seed=0)
    obs_dim   = env_probe.reset().shape[1]
    n_actions = env_probe.num_actions
 
    print(f"  Loading {dqn_path} …")
    agent_dqn   = load_agent(dqn_path,   obs_dim, n_actions)
 
    all_results = []

    print(f"\nEvaluating agents for scenarios: {obstacle_scenarios}")
 
    for n_obs in obstacle_scenarios:
        print(f"\n=== Scenario: {n_obs} obstacles ===")
 
        print(f"  DQN …")
        t_scen_start = time.time()
        m_dqn = evaluate_agent(agent_dqn, n_obs, num_episodes, energy_tracker_cls=None)
        t_scen = time.time() - t_scen_start
 
        all_results.append({
            "nObs":        n_obs,
            "metricsDQN":  m_dqn,
        })
 
        print(f"  DQN      | Success: {m_dqn['successRate']*100:.1f}%"
              f" | Path: {m_dqn['meanPathLength']:.2f}m"
              f" | Smooth: {m_dqn['meanSmoothness']:.4f}"
              f" | Energy: {m_dqn['meanEnergy']:.4f}"
              f" | Eval runtime: {t_scen:.1f}s")
        
    t_total = time.time() - t_total_start
    print("\n" + "="*40)
    print(" Evaluation Timing Summary")
    print("="*40)
    print(f"  Total time: {t_total:.1f}s  ({t_total/60:.1f} min)")
    print("="*40)

    all_inf = []
    for r in all_results:
        all_inf.extend(r["metricsDQN"].get("allInferenceTimes", []))
    if all_inf:
        ia = np.array(all_inf)
        print(f"\nInference Time Summary (across all scenarios and episodes):")
        print(f" Mean: {ia.mean():.4f} ms")
        print(f" Median: {np.median(ia):.4f} ms")
        print(f" Min: {ia.min():.4f} ms")
        print(f" Max: {ia.max():.4f} ms")
        print(f" Calls: {len(ia)}")
    
    print("\n=== Extended: Success vs Obstacle Count (1-10) ===")
    ext_scenarios = list(range(1, 11))
    ext_succ_dqn = []
    for n_obs_ext in ext_scenarios:
        m_ext = evaluate_agent(agent_dqn, n_obs_ext, 100)
        ext_succ_dqn.append(m_ext["successRate"]*100)
        print(f" {n_obs_ext:2d} obstacles: {ext_succ_dqn[-1]:.1f}%")
    fig_success_line_extended(ext_scenarios, ext_succ_dqn, out_folder)

    run_uav_comparison(out_folder)
    run_single_vs_swarm_comparison(out_folder)

    np.save(summary_path, all_results, allow_pickle=True)
    print(f"\n  Results saved → {summary_path}")
 
    scenarios  = [r["nObs"] for r in all_results]
    paths_dqn   = [r["metricsDQN"]["meanPathLength"]  for r in all_results]
    smooth_dqn  = [r["metricsDQN"]["meanSmoothness"]  for r in all_results]
    energy_dqn  = [r["metricsDQN"]["meanEnergy"]      for r in all_results]
    succ_dqn    = [r["metricsDQN"]["successRate"]*100  for r in all_results]

    print("\n  Saving evaluateAgents_multi_obs figures …")
 
    print("  Saving more_figs figures …")
 
    fig_success_line(scenarios, succ_dqn, out_folder)
 
    _bar_grouped(scenarios, paths_dqn,
                 "Number of Obstacles", "Average Path Length (m)",
                 "Path Length Comparison",
                 "fig2_path_length.png", out_folder)
 
    _bar_grouped(scenarios, smooth_dqn,
                 "Number of Obstacles", "Mean Heading Change (rad)",
                 "Path Smoothness Comparison",
                 "fig3_smoothness.png", out_folder)
 
    _bar_grouped(scenarios, energy_dqn,
                 "Number of Obstacles", "Energy Proxy (Σ|Δψ|)",
                 "Energy Consumption Comparison",
                 "fig4_energy.png", out_folder)
 
    for r in all_results:
        _radar_scenario(r["metricsDQN"], r["nObs"],
                        f"fig5_radar_{r['nObs']}.png", out_folder)
 
    fig_heatmap(scenarios, succ_dqn, paths_dqn, smooth_dqn, energy_dqn, out_folder)
 
    fig_tradeoff_scatter(scenarios, paths_dqn, smooth_dqn, out_folder)
 
    all_figs = [
        "fig1_success_line.png",
        "fig2_path_length.png",
        "fig3_smoothness.png",
        "fig4_energy.png",
        "fig5_radar_4.png",
        "fig5_radar_6.png",
        "fig5_radar_8.png",
        "fig6_heatmap.png",
        "fig7_tradeoff_scatter.png",
        "fig_uav_comparison.png",
        "fig_success_1to10.png",
        "fig_single_vs_swarm_bars.png",
        "fig_single_vs_swarm_tradeoff.png",
    ]
    print(f"\n  All results saved to folder: {out_folder}/")
    for f in all_figs:
        p = os.path.join(out_folder, f)
        print(f"    {'✓' if os.path.exists(p) else '✗'}  {f}")
 
 
if __name__ == "__main__":
    main()