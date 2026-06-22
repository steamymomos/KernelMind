"""Probe environment for SoC '26 KernelMind Assignment 2: The Adrian Descent."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

try:
    from IPython.display import clear_output
except Exception:  # pragma: no cover - only used outside notebooks
    clear_output = None


State = Tuple[float, float, int]


@dataclass(frozen=True)
class AdrianConstants:
    """Lore-accurate constants from the assignment."""

    mass: float = 1000.0                 # kg
    gravity: float = 13.7                # m/s^2
    radius: float = 10_700_000.0         # m
    drag_coefficient: float = 2.0        # high-density atmosphere
    max_thrust: float = 25_000.0         # N
    dt: float = 0.1                      # s
    co2_percent: float = 91.0
    methane_percent: float = 7.0
    argon_percent: float = 1.0


class ProbeEnv:
    """Physics-based stochastic MDP for the Adrian atmospheric descent.

    State tuple: (altitude_h_m, velocity_v_m_per_s, wind_index)
        * h > 0 means above the Taumoeba target.
        * v < 0 means descending.
        * wind_index in {0, 1, 2}: Calm, Gusty, Adrian Gale.

    Action space:
        By default: 0 = OFF, 1 = 100% thrust.
        Bonus mode: pass thrust_levels=(0.0, 0.5, 1.0) for OFF/50%/100%.
    """

    wind_names = ("Calm", "Gusty", "Adrian Gale")
    wind_multipliers = np.array([1.0, 1.5, 2.5], dtype=float)

    # Rows are current wind, columns are next wind. Rows sum to 1.0.
    wind_transition = np.array(
        [
            [0.86, 0.12, 0.02],
            [0.15, 0.70, 0.15],
            [0.04, 0.22, 0.74],
        ],
        dtype=float,
    )

    def __init__(
        self,
        drop_altitude: float = 1000.0,
        max_altitude: float = 1200.0,
        max_steps: int = 700,
        safe_velocity: float = -3.0,
        thrust_levels: Sequence[float] = (0.0, 1.0),
        seed: int | None = None,
    ) -> None:
        self.constants = AdrianConstants()
        self.drop_altitude = float(drop_altitude)
        self.max_altitude = float(max_altitude)
        self.max_steps = int(max_steps)
        self.safe_velocity = float(safe_velocity)
        self.thrust_levels = tuple(float(x) for x in thrust_levels)
        if min(self.thrust_levels) < 0.0 or max(self.thrust_levels) > 1.0:
            raise ValueError("All thrust levels must be fractions in [0.0, 1.0].")

        self.rng = np.random.default_rng(seed)
        self.h = self.drop_altitude
        self.v = 0.0
        self.wind_idx = 0
        self.steps = 0
        self.last_result = "not_started"
        self.reset()

    @property
    def n_actions(self) -> int:
        return len(self.thrust_levels)

    def reset(self) -> State:
        """Reset to the drop altitude, zero velocity, calm wind, and zero steps."""
        self.h = self.drop_altitude
        self.v = 0.0
        self.wind_idx = 0
        self.steps = 0
        self.last_result = "in_flight"
        return (self.h, self.v, self.wind_idx)

    def _transition_wind(self) -> int:
        """Sample the next wind state from the Markov transition matrix.

        This uses a tiny hand-written categorical sampler instead of np.random.choice
        because it is called millions of times during training.
        """
        probs = self.wind_transition[self.wind_idx]
        u = float(self.rng.random())
        if u < probs[0]:
            self.wind_idx = 0
        elif u < probs[0] + probs[1]:
            self.wind_idx = 1
        else:
            self.wind_idx = 2
        return self.wind_idx

    def _gravity_force(self, h: float) -> float:
        c = self.constants
        return -c.mass * c.gravity * (1.0 - h / c.radius) ** 2

    def _drag_force(self, v: float, wind_idx: int) -> float:
        if abs(v) < 1e-12:
            return 0.0
        c = self.constants
        return c.drag_coefficient * (v**2) * math.copysign(1.0, -v) * self.wind_multipliers[wind_idx]

    @staticmethod
    def target_velocity_profile(h: float) -> float:
        """A conservative descent-rate profile used only for reward shaping.

        At high altitude the probe is allowed to fall fast. Near the target the desired
        descent speed approaches roughly -1.5 m/s, safely above the -3 m/s catch limit.
        """
        h = max(float(h), 0.0)
        return -max(1.5, min(60.0, 0.08 * h + 1.5))

    def _reward(self, prev_h: float, action: int, done: bool) -> float:
        """Shaped reward that forces landing while avoiding the cowardly-agent loophole."""
        thrust_fraction = self.thrust_levels[action]
        descent_progress = max(prev_h - self.h, 0.0)
        desired_v = self.target_velocity_profile(self.h)
        velocity_error = abs(self.v - desired_v)

        # Dense guidance terms.
        reward = -0.02                                      # time cost
        reward += 0.010 * descent_progress                  # descend; do not hover forever
        reward -= 0.050 * thrust_fraction                   # Astrophage fuel cost
        reward -= 0.050 * velocity_error                    # track safe descent profile

        # Strong slow-down pressure near the catch zone.
        if self.h < 150.0 and self.v < self.safe_velocity:
            reward -= 0.150 * (abs(self.v) - abs(self.safe_velocity)) ** 2

        # Avoid flying upward as a degenerate anti-landing strategy.
        if self.v > 0.0:
            reward -= 0.30 * self.v

        # Terminal rewards / penalties.
        if done:
            if self.last_result == "success":
                reward += 2000.0 - 20.0 * abs(self.v)
            elif self.last_result == "crash":
                excess_speed = max(0.0, abs(self.v) - abs(self.safe_velocity))
                reward -= 2500.0 + 5.0 * excess_speed**2
            elif self.last_result == "runaway":
                reward -= 3000.0 + max(0.0, self.h - self.max_altitude)
            elif self.last_result == "timeout":
                reward -= 3500.0 + 2.0 * max(self.h, 0.0) + 10.0 * abs(self.v)
        return float(reward)

    def step(self, action: int) -> tuple[State, float, bool]:
        """Advance the physics by one Euler step and return (next_state, reward, done)."""
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"Invalid action {action}. Expected 0 to {self.n_actions - 1}.")
        action = int(action)
        prev_h = self.h

        self._transition_wind()

        c = self.constants
        gravity_force = self._gravity_force(self.h)
        thrust_force = self.thrust_levels[action] * c.max_thrust
        drag_force = self._drag_force(self.v, self.wind_idx)
        net_force = gravity_force + thrust_force + drag_force
        acceleration = net_force / c.mass

        # Euler integration using the updated velocity, as specified in the assignment.
        self.v = self.v + acceleration * c.dt
        self.h = self.h + self.v * c.dt
        self.steps += 1

        done = False
        self.last_result = "in_flight"
        if self.h <= 0.0:
            done = True
            self.last_result = "success" if self.v >= self.safe_velocity else "crash"
        elif self.h > self.max_altitude:
            done = True
            self.last_result = "runaway"
        elif self.steps >= self.max_steps:
            done = True
            self.last_result = "timeout"

        reward = self._reward(prev_h, action, done)
        return (self.h, self.v, self.wind_idx), reward, done


def render_probe_ascii(
    h: float,
    max_h: float,
    v: float,
    action: int,
    wind: int,
    step_count: int,
    is_jupyter: bool = False,
) -> None:
    """High-framerate pure ASCII renderer from the assignment, cleaned for execution."""
    if is_jupyter and clear_output is not None:
        clear_output(wait=True)
    elif not is_jupyter:
        os.system("clear" if os.name == "posix" else "cls")

    term_lines = 40
    if h > 150.0:
        display_max = max_h
        zoom_str = "[ CAMERA: WIDE ANGLE (1000 m) ]"
    else:
        display_max = 150.0
        zoom_str = "[ CAMERA: TARGET APPROACH (150 m) ]"

    pos = int((max(h, 0.0) / display_max) * term_lines)
    pos = max(0, min(term_lines, pos))

    wind_strs = ["~ Calm ~", "Gusty", "Adrian Gale"]
    thrust_str = "[####] ON" if action else "[    ] OFF"

    print(
        f"T+{step_count:03d} | ALT: {h:7.1f} m | VEL: {v:8.2f} m/s | "
        f"THRUST: {thrust_str:10s} | WIND: {wind_strs[wind]}"
    )
    print(zoom_str)
    print("-" * 75)

    for i in range(term_lines, -1, -1):
        if i == pos:
            if action:
                print("                 /\\")
                print("                 ||")
                print("                /WW\\")
                print("                 ||    <-- spin-drive")
            else:
                print("                 /\\")
                print("                 ||")
                print("                /--\\")
                print("                       ")
        else:
            if i % 10 == 0:
                level = int((i / term_lines) * display_max)
                print(f"{level:4d}m +-------------------------------------------")
            else:
                print("      |")

    print("======================= [ TAUMOEBA TARGET ] =======================")
    time.sleep(0.04)
