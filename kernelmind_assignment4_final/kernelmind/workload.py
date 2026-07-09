from __future__ import annotations

import random
from typing import Iterable

from .config import SchedulerConfig
from .process import Process


def clone_workload(workload: Iterable[Process]) -> list[Process]:
    return [p.clone_fresh() for p in workload]


def generate_workload(seed: int, cfg: SchedulerConfig = SchedulerConfig(), memory: bool = False) -> list[Process]:
    rng = random.Random(seed)
    workload: list[Process] = []
    for pid in range(cfg.n_processes):
        arrival = rng.randint(0, cfg.arrival_window)
        burst = rng.randint(cfg.burst_min, cfg.burst_max)
        priority = rng.randint(cfg.priority_min, cfg.priority_max)
        mem = rng.randint(cfg.memory_min, cfg.memory_max) if memory else 1
        workload.append(Process(pid, arrival, burst, priority, memory_required=mem))
    return sorted(workload, key=lambda p: (p.arrival_time, p.pid))


def generate_universal_test_set(count: int, seed_offset: int, cfg: SchedulerConfig = SchedulerConfig()) -> list[list[Process]]:
    return [generate_workload(seed_offset + i, cfg) for i in range(count)]


def generate_memory_test_set(count: int, seed_offset: int, cfg: SchedulerConfig = SchedulerConfig()) -> list[list[Process]]:
    return [generate_workload(seed_offset + i, cfg, memory=True) for i in range(count)]


def generate_io_storm_workload(seed: int, cfg: SchedulerConfig = SchedulerConfig()) -> list[Process]:
    rng = random.Random(seed)
    jobs: list[Process] = []
    for pid in range(8):
        jobs.append(Process(pid=pid, arrival_time=0, burst_time=rng.randint(1, 2), priority=rng.randint(1, 10)))
    for k in range(2):
        pid = 8 + k
        jobs.append(Process(pid=pid, arrival_time=0, burst_time=rng.randint(50, 60), priority=rng.randint(1, 10)))
    return sorted(jobs, key=lambda p: (p.arrival_time, p.pid))


def generate_io_storm_test_set(count: int, seed_offset: int, cfg: SchedulerConfig = SchedulerConfig()) -> list[list[Process]]:
    return [generate_io_storm_workload(seed_offset + i, cfg) for i in range(count)]
