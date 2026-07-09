from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SchedulerConfig
from .process import Process
from .state import ready_queue_to_state
from .workload import clone_workload


@dataclass
class RewardTerms:
    tick_penalty: float = 0.0
    queue_penalty: float = 0.0
    completion_bonus: float = 0.0
    fairness_penalty: float = 0.0
    starvation_penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.tick_penalty + self.queue_penalty + self.completion_bonus + self.fairness_penalty + self.starvation_penalty


class DirectSchedulerEnvironment:
    """Single-core direct-control MDP.

    The action is an integer ready-queue slot. Invalid padded slots are rejected
    immediately; masking mistakes should fail loudly instead of silently training
    the agent on impossible actions.
    """

    def __init__(self, workload: list[Process], cfg: SchedulerConfig = SchedulerConfig()):
        self.original_workload = clone_workload(workload)
        self.cfg = cfg
        self.processes: list[Process] = []
        self.ready_queue: list[Process] = []
        self.completed: list[Process] = []
        self.clock = 0
        self.next_arrival_idx = 0
        self.last_reward_terms = RewardTerms()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.processes = sorted(clone_workload(self.original_workload), key=lambda p: (p.arrival_time, p.pid))
        self.ready_queue = []
        self.completed = []
        self.clock = 0
        self.next_arrival_idx = 0
        self.last_reward_terms = RewardTerms()
        self._admit_arrivals()
        self._fast_forward_if_idle()
        return self._state()

    def _state(self) -> tuple[np.ndarray, np.ndarray]:
        return ready_queue_to_state(self.ready_queue, self.cfg)

    def _admit_arrivals(self) -> None:
        while self.next_arrival_idx < len(self.processes) and self.processes[self.next_arrival_idx].arrival_time <= self.clock:
            self.ready_queue.append(self.processes[self.next_arrival_idx])
            self.next_arrival_idx += 1

    def _fast_forward_if_idle(self) -> None:
        if self.ready_queue or self.next_arrival_idx >= len(self.processes):
            return
        self.clock = max(self.clock, self.processes[self.next_arrival_idx].arrival_time)
        self._admit_arrivals()

    def valid_action_indices(self) -> list[int]:
        return list(range(min(len(self.ready_queue), self.cfg.max_queue_size)))

    def _compute_reward_terms(self, completed_this_tick: bool) -> RewardTerms:
        tick_penalty = -0.020
        queue_penalty = -0.040 * min(len(self.ready_queue), self.cfg.max_queue_size) / self.cfg.max_queue_size
        completion_bonus = 0.800 if completed_this_tick else 0.0

        if len(self.ready_queue) >= 2:
            vr = np.array([p.virtual_runtime for p in self.ready_queue], dtype=np.float32)
            normalized_var = float(np.var(vr) / (self.cfg.max_burst ** 2))
            fairness_penalty = -0.180 * min(normalized_var, 1.0)
        else:
            fairness_penalty = 0.0

        if self.ready_queue:
            max_wait = max(p.wait_time for p in self.ready_queue)
            excess = max(0, max_wait - self.cfg.starvation_threshold)
            bounded_fraction = min((excess / max(self.cfg.starvation_threshold, 1)) ** 2, 1.0)
            starvation_penalty = -0.250 * bounded_fraction
        else:
            starvation_penalty = 0.0

        return RewardTerms(tick_penalty, queue_penalty, completion_bonus, fairness_penalty, starvation_penalty)

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
        _, mask = self._state()
        if action < 0 or action >= self.cfg.max_queue_size or not bool(mask[action]):
            raise ValueError(f"Invalid action {action}; current valid mask is {mask.tolist()}")

        proc = self.ready_queue[action]
        proc.remaining_time -= 1
        proc.virtual_runtime += 1

        for i, other in enumerate(self.ready_queue):
            if i != action:
                other.wait_time += 1

        self.clock += 1
        completed_this_tick = proc.remaining_time == 0
        if completed_this_tick:
            proc.finish_time = self.clock
            self.completed.append(proc)
            self.ready_queue.pop(action)

        self._admit_arrivals()
        self._fast_forward_if_idle()
        done = len(self.completed) == len(self.processes)
        self.last_reward_terms = self._compute_reward_terms(completed_this_tick)
        next_state, next_mask = self._state()
        return next_state, next_mask, self.last_reward_terms.total, done
