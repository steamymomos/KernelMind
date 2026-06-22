"""Bonus challenge: variable spin-drive with OFF / 50% / 100% thrust."""
from __future__ import annotations

from agent import AgentConfig, ProbeAgent
from env import ProbeEnv
from train import moving_average


def make_bonus_env_and_agent(seed: int = 11) -> tuple[ProbeEnv, ProbeAgent]:
    env = ProbeEnv(thrust_levels=(0.0, 0.5, 1.0), seed=seed)
    agent = ProbeAgent(n_actions=env.n_actions, config=AgentConfig(epsilon=1.0), seed=seed)
    return env, agent


if __name__ == "__main__":
    env, agent = make_bonus_env_and_agent()
    print("Bonus action space:", env.thrust_levels)
    print("Q-table shape:", agent.q_table.shape)
    print("Q-table entries:", agent.q_table.size)
