"""Tabular Q-learning agent for The Adrian Descent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

State = Tuple[float, float, int]
DiscreteState = Tuple[int, int, int]


@dataclass
class AgentConfig:
    altitude_buckets: int = 50
    velocity_buckets: int = 50
    altitude_min: float = 0.0
    altitude_max: float = 1200.0
    velocity_min: float = -110.0
    velocity_max: float = 40.0
    alpha: float = 0.15
    gamma: float = 0.98
    epsilon: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    # The safety tie-breaker is used only for unvisited / low-confidence states.
    min_visits_for_q_choice: int = 50
    q_confidence_margin: float = 1000.0


class ProbeAgent:
    """Tabular Q-learning agent with discretized altitude, velocity, and wind."""

    def __init__(self, n_actions: int = 2, config: AgentConfig | None = None, seed: int | None = None) -> None:
        self.config = config or AgentConfig()
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

        c = self.config
        self.altitude_bins = np.linspace(c.altitude_min, c.altitude_max, c.altitude_buckets - 1)
        self.velocity_bins = np.linspace(c.velocity_min, c.velocity_max, c.velocity_buckets - 1)
        self.q_table = np.zeros((c.altitude_buckets, c.velocity_buckets, 3, self.n_actions), dtype=np.float32)
        self.visit_counts = np.zeros_like(self.q_table, dtype=np.int32)
        self.epsilon = c.epsilon

    def discretize_state(self, raw_state: State) -> DiscreteState:
        """Map continuous (h, v, wind) to Q-table indices using np.digitize."""
        h, v, wind = raw_state
        c = self.config
        # np.searchsorted is equivalent to np.digitize for increasing bins here,
        # but faster in a tight training loop.
        h_idx = int(np.searchsorted(self.altitude_bins, h, side="right"))
        v_idx = int(np.searchsorted(self.velocity_bins, v, side="right"))
        if h_idx < 0:
            h_idx = 0
        elif h_idx >= c.altitude_buckets:
            h_idx = c.altitude_buckets - 1
        if v_idx < 0:
            v_idx = 0
        elif v_idx >= c.velocity_buckets:
            v_idx = c.velocity_buckets - 1
        wind_idx = 0 if wind < 0 else (2 if wind > 2 else int(wind))
        return h_idx, v_idx, wind_idx

    @staticmethod
    def _safe_descent_tiebreak(raw_state: State, n_actions: int) -> int:
        """Physics-based tie-breaker for states where the Q-table has no confident opinion.

        This avoids the standard zero-table bias toward action 0 in unvisited states. It is
        deliberately conservative and never replaces a confident learned Q preference.
        """
        h, v, _ = raw_state
        target_v = -max(1.5, min(60.0, 0.08 * max(h, 0.0) + 1.5))
        wants_more_thrust = v < target_v
        if n_actions == 2:
            return int(wants_more_thrust)
        # Bonus variable-thrust setting: use partial thrust near the profile, full thrust
        # when falling too fast, and zero thrust when too slow/upward.
        if v < target_v - 8.0:
            return min(2, n_actions - 1)     # 100% thrust if available
        if v < target_v:
            return min(1, n_actions - 1)     # 50% thrust if available
        return 0

    def choose_action(self, raw_state: State) -> int:
        """Epsilon-greedy policy: explore randomly, otherwise exploit the Q-table."""
        discrete_state = self.discretize_state(raw_state)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q_values = self.q_table[discrete_state]
        visits = int(self.visit_counts[discrete_state].sum())
        q_margin = float(q_values.max() - q_values.min())

        if visits < self.config.min_visits_for_q_choice or q_margin < self.config.q_confidence_margin:
            return self._safe_descent_tiebreak(raw_state, self.n_actions)

        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best_actions))

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> float:
        """Bellman Q-learning update and return the TD error."""
        s = self.discretize_state(state)
        ns = self.discretize_state(next_state)
        action = int(action)

        current_q = float(self.q_table[s + (action,)])
        target = float(reward) if done else float(reward) + self.config.gamma * float(np.max(self.q_table[ns]))
        td_error = target - current_q
        self.q_table[s + (action,)] = current_q + self.config.alpha * td_error
        self.visit_counts[s + (action,)] += 1
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            q_table=self.q_table,
            visit_counts=self.visit_counts,
            epsilon=np.array([self.epsilon], dtype=float),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self.q_table[:] = data["q_table"]
        self.visit_counts[:] = data["visit_counts"]
        self.epsilon = float(data["epsilon"][0])
