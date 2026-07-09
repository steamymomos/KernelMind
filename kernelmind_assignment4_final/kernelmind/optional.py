from __future__ import annotations

import random

import numpy as np
import torch


from .baselines import evaluate_baselines
from .config import SchedulerConfig, TrainingConfig
from .evaluation import evaluate_agent
from .metrics import aggregate_metrics, calculate_metrics
from .network import DirectSchedulerNet
from .numpy_policy import NumpyPolicy
from .process import Process
from .state import ready_queue_to_state
from .training import train_agent
from .workload import clone_workload, generate_io_storm_test_set, generate_memory_test_set


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


def _agent_q_scores(policy_net: DirectSchedulerNet | NumpyPolicy, ready: list[Process], cfg: SchedulerConfig, device: torch.device) -> np.ndarray:
    state, mask = ready_queue_to_state(ready, cfg)
    policy = policy_net if isinstance(policy_net, NumpyPolicy) else NumpyPolicy(policy_net)
    return policy.q_values(state, mask)


def _assign_core(proc: Process, free_cores: list[int]) -> None:
    if proc.last_core in free_cores:
        free_cores.remove(proc.last_core)  # type: ignore[arg-type]
        return
    core = free_cores.pop(0)
    if proc.last_core is not None and proc.last_core != core:
        proc.migrations += 1
    proc.last_core = core


def simulate_multicore_policy(
    workload: list[Process],
    policy_name: str,
    cfg: SchedulerConfig = SchedulerConfig(),
    policy_net: DirectSchedulerNet | NumpyPolicy | None = None,
    device: torch.device | None = None,
    cores: int = 4,
    seed: int = 0,
) -> list[Process]:
    rng = random.Random(seed)
    device = torch.device("cpu") if device is None else device
    processes = sorted(clone_workload(workload), key=lambda p: (p.arrival_time, p.pid))
    ready: list[Process] = []
    completed: list[Process] = []
    clock = 0
    idx = _admit(processes, ready, 0, clock)

    while len(completed) < len(processes):
        idx, clock = _fast_forward(processes, ready, idx, clock)
        if not ready:
            continue
        k = min(cores, len(ready), cfg.max_queue_size)

        if policy_name == "Agent":
            if policy_net is None:
                raise ValueError("policy_net is required for Agent multicore evaluation")
            q = _agent_q_scores(policy_net, ready, cfg, device)
            selected_indices = [int(i) for i in np.argsort(q)[::-1] if i < len(ready)][:k]
        elif policy_name == "FCFS":
            selected_indices = list(range(k))
        elif policy_name == "SJF":
            selected_indices = sorted(range(len(ready)), key=lambda i: (ready[i].remaining_time, ready[i].arrival_time, ready[i].pid))[:k]
        elif policy_name == "RR":
            selected_indices = list(range(k))
        elif policy_name == "Random":
            selected_indices = rng.sample(range(len(ready)), k)
        else:
            raise ValueError(f"unknown policy {policy_name}")

        selected_set = set(selected_indices)
        selected_pids_before_update = {ready[i].pid for i in selected_indices}
        free_cores = list(range(cores))
        for i in selected_indices:
            _assign_core(ready[i], free_cores)

        for i, proc in enumerate(ready):
            if i in selected_set:
                proc.remaining_time -= 1
                proc.virtual_runtime += 1
            else:
                proc.wait_time += 1
        clock += 1

        new_ready: list[Process] = []
        for p in ready:
            if p.remaining_time == 0:
                p.finish_time = clock
                completed.append(p)
            else:
                new_ready.append(p)
        ready = new_ready

        if policy_name == "RR" and ready:
            survivors_selected = [p for p in ready if p.pid in selected_pids_before_update]
            survivors_other = [p for p in ready if p.pid not in selected_pids_before_update]
            ready = survivors_other + survivors_selected

        idx = _admit(processes, ready, idx, clock)
    return completed


def evaluate_multicore(
    policy_net: DirectSchedulerNet,
    test_set: list[list[Process]],
    cfg: SchedulerConfig = SchedulerConfig(),
    device: torch.device | None = None,
    cores: int = 4,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    cached_policy = policy_net if isinstance(policy_net, NumpyPolicy) else NumpyPolicy(policy_net)
    for policy in ["FCFS", "SJF", "RR", "Random", "Agent"]:
        rows = [
            calculate_metrics(simulate_multicore_policy(workload, policy, cfg, cached_policy, device, cores, seed=91_000 + i))
            for i, workload in enumerate(test_set)
        ]
        results[policy] = aggregate_metrics(rows)
    return results


def _memory_admit_arrivals(processes: list[Process], ready: list[Process], swap: list[Process], idx: int, clock: int, cap: int) -> int:
    used = sum(p.memory_required for p in ready)
    while idx < len(processes) and processes[idx].arrival_time <= clock:
        p = processes[idx]
        if used + p.memory_required <= cap:
            ready.append(p)
            used += p.memory_required
        else:
            p.swap_enter_time = clock
            swap.append(p)
        idx += 1
    return idx


def _memory_first_fit(ready: list[Process], swap: list[Process], clock: int, cap: int) -> None:
    used = sum(p.memory_required for p in ready)
    i = 0
    while i < len(swap):
        p = swap[i]
        if used + p.memory_required <= cap:
            if p.swap_enter_time is not None:
                p.swap_wait_time += clock - p.swap_enter_time
                p.swap_enter_time = None
            ready.append(p)
            used += p.memory_required
            swap.pop(i)
        else:
            i += 1


def _select_single(
    policy: str,
    ready: list[Process],
    rng: random.Random,
    policy_net: DirectSchedulerNet | NumpyPolicy | None,
    cfg: SchedulerConfig,
    device: torch.device,
) -> int:
    if policy == "Agent":
        if policy_net is None:
            raise ValueError("policy_net is required for Agent")
        return int(np.argmax(_agent_q_scores(policy_net, ready, cfg, device)))
    if policy in {"FCFS", "RR"}:
        return 0
    if policy == "SJF":
        return min(range(len(ready)), key=lambda i: (ready[i].remaining_time, ready[i].arrival_time, ready[i].pid))
    if policy == "Random":
        return rng.randrange(len(ready))
    raise ValueError(f"unknown policy {policy}")


def simulate_memory_policy(
    workload: list[Process],
    policy: str,
    cfg: SchedulerConfig = SchedulerConfig(),
    policy_net: DirectSchedulerNet | NumpyPolicy | None = None,
    device: torch.device | None = None,
    seed: int = 0,
) -> list[Process]:
    device = torch.device("cpu") if device is None else device
    rng = random.Random(seed)
    cap = cfg.memory_cap
    processes = sorted(clone_workload(workload), key=lambda p: (p.arrival_time, p.pid))
    ready: list[Process] = []
    swap: list[Process] = []
    completed: list[Process] = []
    clock = 0
    idx = _memory_admit_arrivals(processes, ready, swap, 0, clock, cap)
    _memory_first_fit(ready, swap, clock, cap)

    while len(completed) < len(processes):
        if not ready:
            if idx < len(processes):
                clock = max(clock + (1 if swap else 0), processes[idx].arrival_time)
                idx = _memory_admit_arrivals(processes, ready, swap, idx, clock, cap)
                _memory_first_fit(ready, swap, clock, cap)
            elif swap:
                _memory_first_fit(ready, swap, clock, cap)
                if not ready:
                    raise RuntimeError("deadlock: a swapped process exceeds memory cap")
            else:
                continue
        if not ready:
            continue

        action = _select_single(policy, ready, rng, policy_net, cfg, device)
        proc = ready[action]
        proc.remaining_time -= 1
        proc.virtual_runtime += 1
        for i, other in enumerate(ready):
            if i != action:
                other.wait_time += 1
        for p in swap:
            p.wait_time += 1
        clock += 1

        if proc.remaining_time == 0:
            proc.finish_time = clock
            completed.append(proc)
            ready.pop(action)
        elif policy == "RR":
            ready.append(ready.pop(action))

        idx = _memory_admit_arrivals(processes, ready, swap, idx, clock, cap)
        _memory_first_fit(ready, swap, clock, cap)
    return completed


def evaluate_memory_constraints(
    policy_net: DirectSchedulerNet,
    cfg: SchedulerConfig = SchedulerConfig(),
    device: torch.device | None = None,
    count: int = 40,
) -> dict[str, dict[str, float]]:
    test_set = generate_memory_test_set(count, 70_000, cfg)
    results: dict[str, dict[str, float]] = {}
    cached_policy = policy_net if isinstance(policy_net, NumpyPolicy) else NumpyPolicy(policy_net)
    for policy in ["FCFS", "SJF", "RR", "Random", "Agent"]:
        rows = [
            calculate_metrics(simulate_memory_policy(workload, policy, cfg, cached_policy, device, seed=77_000 + i))
            for i, workload in enumerate(test_set)
        ]
        results[policy] = aggregate_metrics(rows)
    return results


def evaluate_io_storm(
    policy_net: DirectSchedulerNet,
    cfg: SchedulerConfig = SchedulerConfig(),
    device: torch.device | None = None,
    count: int = 30,
) -> dict[str, dict[str, float]]:
    storm_set = generate_io_storm_test_set(count, 80_000, cfg)
    results = evaluate_baselines(storm_set, cfg)
    results["Direct-DQN"] = evaluate_agent(policy_net, storm_set, cfg, device)
    return results


def run_seed_stability(
    seeds: list[int],
    sched_cfg: SchedulerConfig,
    base_train_cfg: TrainingConfig,
    test_set: list[list[Process]],
    sjf_reference: dict[str, float],
    device: torch.device | None = None,
    episodes: int = 80,
) -> dict[str, object]:
    device = torch.device("cpu") if device is None else device
    rows: list[dict[str, float]] = []
    dominated = 0
    for seed in seeds:
        cfg = base_train_cfg.with_seed(seed, episodes=episodes)
        policy, _ = train_agent(sched_cfg, cfg, device)
        metrics = evaluate_agent(policy, test_set, sched_cfg, device)
        is_dominated = metrics["mean_wait"] > sjf_reference["mean_wait"] and metrics["jain_fairness"] < sjf_reference["jain_fairness"]
        dominated += int(is_dominated)
        rows.append({
            "seed": float(seed),
            "mean_wait": metrics["mean_wait"],
            "jain_fairness": metrics["jain_fairness"],
            "dominated_by_sjf": float(is_dominated),
        })
    return {
        "rows": rows,
        "mean_wait_mean": float(np.mean([r["mean_wait"] for r in rows])),
        "mean_wait_std": float(np.std([r["mean_wait"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "fairness_mean": float(np.mean([r["jain_fairness"] for r in rows])),
        "fairness_std": float(np.std([r["jain_fairness"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "dominated_fraction": float(dominated / max(len(seeds), 1)),
    }
