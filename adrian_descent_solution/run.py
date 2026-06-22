"""Greedy evaluation / terminal rendering for the trained Adrian descent agent."""
from __future__ import annotations

import argparse
from pathlib import Path

from agent import AgentConfig, ProbeAgent
from env import ProbeEnv, render_probe_ascii


def run_greedy(q_table_path: str | None = "adrian_q_table.npz", render: bool = True, seed: int = 123) -> dict:
    env = ProbeEnv(seed=seed)
    agent = ProbeAgent(n_actions=env.n_actions, config=AgentConfig(epsilon=0.0), seed=seed)

    if q_table_path and Path(q_table_path).exists():
        agent.load(q_table_path)
    agent.epsilon = 0.0

    state = env.reset()
    total_reward = 0.0
    done = False
    last_action = 0

    while not done:
        action = agent.choose_action(state)
        last_action = action
        if render:
            render_probe_ascii(env.h, env.drop_altitude, env.v, action, env.wind_idx, env.steps)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    return {
        "result": env.last_result,
        "steps": env.steps,
        "final_altitude": env.h,
        "impact_velocity": env.v,
        "last_action": last_action,
        "total_reward": total_reward,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-table", type=str, default="adrian_q_table.npz")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(run_greedy(args.q_table, render=not args.no_render, seed=args.seed))
