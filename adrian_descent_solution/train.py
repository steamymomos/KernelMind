"""Training loop for The Adrian Descent."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from agent import AgentConfig, ProbeAgent
from env import ProbeEnv


def moving_average(values: list[float] | np.ndarray, window: int = 250) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def train(episodes: int = 15_000, seed: int = 7, out_dir: str | Path = ".") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ProbeEnv(seed=seed)
    config = AgentConfig(epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.02)
    agent = ProbeAgent(n_actions=env.n_actions, config=config, seed=seed)

    rewards: list[float] = []
    successes: list[int] = []
    results: list[str] = []
    impact_velocities: list[float] = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        agent.decay_epsilon()
        rewards.append(total_reward)
        successes.append(1 if env.last_result == "success" else 0)
        results.append(env.last_result)
        impact_velocities.append(env.v)

    # Save metrics.
    metrics_path = out_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "success", "result", "impact_velocity"])
        for i, (r, s, result, v) in enumerate(zip(rewards, successes, results, impact_velocities), start=1):
            writer.writerow([i, r, s, result, v])

    # Plot moving averages.
    window = 250
    reward_ma = moving_average(rewards, window)
    success_ma = moving_average(successes, window) * 100.0
    x = np.arange(window, episodes + 1) if len(reward_ma) != episodes else np.arange(1, episodes + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, reward_ma)
    axes[0].set_title(f"Adrian Descent Q-Learning: Moving Average Reward (window={window})")
    axes[0].set_ylabel("Episode reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, success_ma)
    axes[1].set_title(f"Landing Success Rate Moving Average (window={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    curve_path = out_dir / "learning_curve.png"
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)

    agent_path = out_dir / "adrian_q_table.npz"
    agent.save(str(agent_path))

    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes[-1000:]) if len(successes) >= 1000 else np.mean(successes)),
        "final_epsilon": agent.epsilon,
        "learning_curve": str(curve_path),
        "metrics_csv": str(metrics_path),
        "q_table": str(agent_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()
    summary = train(episodes=args.episodes, seed=args.seed, out_dir=args.out_dir)
    print(summary)
