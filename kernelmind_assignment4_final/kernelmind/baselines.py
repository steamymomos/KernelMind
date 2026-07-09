from __future__ import annotations

import random
from collections import deque
from typing import Callable

from .config import SchedulerConfig
from .metrics import aggregate_metrics, calculate_metrics
from .process import Process
from .workload import clone_workload


def _admit(processes: list[Process], ready: list[Process], idx: int, clock: int) -> int:
    while idx < len(processes) and processes[idx].arrival_time <= clock:
        ready.append(processes[idx])
        idx += 1
    return idx


def _fast_forward(processes: list[Process], ready: list[Process], idx: int, clock: int) -> tuple[int, int]:
    if ready or idx >= len(processes):
        return idx, clock
    clock = max(clock, processes[idx].arrival_time)
    idx = _admit(processes, ready, idx, clock)
    return idx, clock


def _simulate_tick_policy(
    workload: list[Process],
    select_action: Callable[[list[Process], random.Random], int],
    seed: int = 0,
    cfg: SchedulerConfig = SchedulerConfig(),
) -> list[Process]:
    rng = random.Random(seed)
    processes = sorted(clone_workload(workload), key=lambda p: (p.arrival_time, p.pid))
    ready: list[Process] = []
    completed: list[Process] = []
    clock = 0
    idx = _admit(processes, ready, 0, clock)

    while len(completed) < len(processes):
        idx, clock = _fast_forward(processes, ready, idx, clock)
        if not ready:
            continue
        action = select_action(ready, rng)
        proc = ready[action]
        proc.remaining_time -= 1
        proc.virtual_runtime += 1
        for j, other in enumerate(ready):
            if j != action:
                other.wait_time += 1
        clock += 1
        if proc.remaining_time == 0:
            proc.finish_time = clock
            completed.append(proc)
            ready.pop(action)
        idx = _admit(processes, ready, idx, clock)
    return completed


def simulate_fcfs(workload: list[Process], cfg: SchedulerConfig = SchedulerConfig()) -> list[Process]:
    return _simulate_tick_policy(workload, lambda ready, rng: 0, 0, cfg)


def simulate_sjf(workload: list[Process], cfg: SchedulerConfig = SchedulerConfig()) -> list[Process]:
    return _simulate_tick_policy(
        workload,
        lambda ready, rng: min(range(len(ready)), key=lambda i: (ready[i].remaining_time, ready[i].arrival_time, ready[i].pid)),
        0,
        cfg,
    )


def simulate_random(workload: list[Process], seed: int = 0, cfg: SchedulerConfig = SchedulerConfig()) -> list[Process]:
    return _simulate_tick_policy(workload, lambda ready, rng: rng.randrange(len(ready)), seed, cfg)


def simulate_rr(workload: list[Process], quantum: int = 3, cfg: SchedulerConfig = SchedulerConfig()) -> list[Process]:
    processes = sorted(clone_workload(workload), key=lambda p: (p.arrival_time, p.pid))
    ready: deque[Process] = deque()
    completed: list[Process] = []
    clock = 0
    idx = 0
    current: Process | None = None
    q_used = 0

    def admit() -> None:
        nonlocal idx
        while idx < len(processes) and processes[idx].arrival_time <= clock:
            ready.append(processes[idx])
            idx += 1

    admit()
    while len(completed) < len(processes):
        if current is None and not ready and idx < len(processes):
            clock = max(clock, processes[idx].arrival_time)
            admit()
        if current is None:
            if not ready:
                continue
            current = ready.popleft()
            q_used = 0

        current.remaining_time -= 1
        current.virtual_runtime += 1
        for other in ready:
            other.wait_time += 1
        clock += 1
        q_used += 1
        admit()

        if current.remaining_time == 0:
            current.finish_time = clock
            completed.append(current)
            current = None
            q_used = 0
        elif q_used >= quantum:
            ready.append(current)
            current = None
            q_used = 0
    return completed


BASELINE_SIMULATORS = {
    "FCFS": simulate_fcfs,
    "SJF": simulate_sjf,
    "RR": lambda workload, cfg=SchedulerConfig(): simulate_rr(workload, cfg.rr_quantum, cfg),
    "Random": simulate_random,
}


def evaluate_baselines(test_set: list[list[Process]], cfg: SchedulerConfig = SchedulerConfig()) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, simulator in BASELINE_SIMULATORS.items():
        rows: list[dict[str, float]] = []
        for i, workload in enumerate(test_set):
            if name == "Random":
                completed = simulate_random(workload, seed=10_000 + i, cfg=cfg)
            else:
                completed = simulator(workload, cfg)  # type: ignore[misc]
            rows.append(calculate_metrics(completed))
        results[name] = aggregate_metrics(rows)
    return results
