"""
KernelMind Assignment 3: The Hybrid Meta-Scheduler
A notebook-compatible implementation of CPU scheduling baselines, metrics,
an MDP environment, and a tabular Q-learning meta-scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable
from collections import deque, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Part 1: Process representation, workload generation, and baseline simulators
# -----------------------------------------------------------------------------

@dataclass
class Process:
    """Minimal Process Control Block for a CPU-scheduling simulation."""
    pid: int
    arrival: int
    burst: int
    remaining: int = field(init=False)
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    executed: int = 0

    def __post_init__(self) -> None:
        self.remaining = int(self.burst)

    def clone(self) -> "Process":
        return Process(pid=self.pid, arrival=self.arrival, burst=self.burst)

    @property
    def wait_time(self) -> int:
        if self.finish_time is None:
            raise ValueError("wait_time is only available after the process finishes.")
        return self.finish_time - self.arrival - self.burst

    @property
    def turnaround_time(self) -> int:
        if self.finish_time is None:
            raise ValueError("turnaround_time is only available after the process finishes.")
        return self.finish_time - self.arrival


def clone_workload(processes: List[Process]) -> List[Process]:
    """Return a deep-enough copy so that each simulator can mutate safely."""
    return [p.clone() for p in processes]


def generate_workload(n: int = 10, rng=None, mode: str = "mixed") -> List[Process]:
    """
    Create one test episode.

    mode='uniform': staggered arrivals and moderately varied bursts.
    mode='convoy': one long CPU-heavy job arrives before many short jobs.
    mode='io_storm': 8 tiny jobs arrive with 2 very large jobs.
    mode='mixed': stochastic mix used for training and evaluation.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng

    if mode == "uniform":
        arrivals = np.sort(rng.integers(0, 25, size=n))
        bursts = rng.integers(1, 21, size=n)

    elif mode == "convoy":
        arrivals = [0]
        bursts = [int(rng.integers(40, 70))]
        for _ in range(n - 1):
            arrivals.append(int(rng.integers(1, 9)))
            bursts.append(int(rng.integers(1, 7)))
        order = np.argsort(arrivals)
        arrivals = np.array(arrivals)[order]
        bursts = np.array(bursts)[order]

    elif mode == "io_storm":
        if n < 10:
            raise ValueError("io_storm mode assumes n >= 10")
        short_bursts = rng.integers(1, 3, size=8)
        long_bursts = rng.integers(50, 75, size=n - 8)
        bursts = np.concatenate([short_bursts, long_bursts])
        arrivals = np.zeros(n, dtype=int)
        order = rng.permutation(n)
        arrivals = arrivals[order]
        bursts = bursts[order]

    elif mode == "mixed":
        # Convoy cases make preemption valuable and expose weaknesses in non-preemptive SJF.
        if rng.random() < 0.45:
            return generate_workload(n=n, rng=rng, mode="convoy")
        arrivals = np.sort(rng.integers(0, 24, size=n))
        bursts = rng.integers(1, 20, size=n)
        # Occasionally inject one long CPU-heavy job near the front of the episode.
        if rng.random() < 0.35:
            heavy_idx = int(rng.integers(0, n))
            bursts[heavy_idx] = int(rng.integers(35, 60))
            arrivals[heavy_idx] = int(rng.integers(0, 4))
            order = np.argsort(arrivals)
            arrivals = arrivals[order]
            bursts = bursts[order]
    else:
        raise ValueError(f"Unknown workload mode: {mode}")

    return [Process(pid=i, arrival=int(arrivals[i]), burst=int(bursts[i])) for i in range(n)]


def simulate_fcfs(processes: List[Process]) -> List[Process]:
    """First-Come, First-Served: non-preemptive, arrival-order execution."""
    ps = clone_workload(processes)
    time = 0
    completed: List[Process] = []
    while len(completed) < len(ps):
        ready = [p for p in ps if p.arrival <= time and p.remaining > 0]
        if not ready:
            time = min(p.arrival for p in ps if p.remaining > 0)
            continue
        p = min(ready, key=lambda x: (x.arrival, x.pid))
        if p.start_time is None:
            p.start_time = time
        p.executed += p.remaining
        time += p.remaining
        p.remaining = 0
        p.finish_time = time
        completed.append(p)
    return sorted(completed, key=lambda x: x.pid)


def simulate_sjf(processes: List[Process]) -> List[Process]:
    """Shortest Job First: non-preemptive, shortest total burst among ready jobs."""
    ps = clone_workload(processes)
    time = 0
    completed: List[Process] = []
    while len(completed) < len(ps):
        ready = [p for p in ps if p.arrival <= time and p.remaining > 0]
        if not ready:
            time = min(p.arrival for p in ps if p.remaining > 0)
            continue
        p = min(ready, key=lambda x: (x.burst, x.arrival, x.pid))
        if p.start_time is None:
            p.start_time = time
        p.executed += p.remaining
        time += p.remaining
        p.remaining = 0
        p.finish_time = time
        completed.append(p)
    return sorted(completed, key=lambda x: x.pid)


def simulate_rr(processes: List[Process], quantum: int = 4, context_switch_cost: int = 0) -> List[Process]:
    """Round Robin with a fixed quantum and optional context-switch tax."""
    ps = clone_workload(processes)
    time = 0
    completed: List[Process] = []
    ready_q: deque[int] = deque()
    added: set[int] = set()
    last_pid: Optional[int] = None

    def add_arrivals() -> None:
        for p in sorted(ps, key=lambda x: (x.arrival, x.pid)):
            if p.pid not in added and p.arrival <= time:
                ready_q.append(p.pid)
                added.add(p.pid)

    while len(completed) < len(ps):
        add_arrivals()
        if not ready_q:
            time = min(p.arrival for p in ps if p.remaining > 0 and p.pid not in added)
            add_arrivals()
            continue

        p = ps[ready_q.popleft()]
        if last_pid is not None and last_pid != p.pid and context_switch_cost:
            time += context_switch_cost
            add_arrivals()
        last_pid = p.pid

        if p.start_time is None:
            p.start_time = time
        run = min(quantum, p.remaining)
        for _ in range(run):
            p.remaining -= 1
            p.executed += 1
            time += 1
            add_arrivals()
            if p.remaining == 0:
                break
        if p.remaining == 0:
            p.finish_time = time
            completed.append(p)
        else:
            ready_q.append(p.pid)
    return sorted(completed, key=lambda x: x.pid)


def simulate_random_agent(processes: List[Process], seed=None) -> List[Process]:
    """
    Random-agent baseline: choose one scheduling heuristic uniformly at random
    for the episode. This is intentionally non-intelligent and serves as a floor.
    """
    rng = np.random.default_rng(seed)
    choices: List[Callable[[List[Process]], List[Process]]] = [
        simulate_fcfs,
        simulate_sjf,
        lambda ps: simulate_rr(ps, quantum=4),
    ]
    return choices[int(rng.integers(len(choices)))](processes)


# -----------------------------------------------------------------------------
# Part 2: Metrics
# -----------------------------------------------------------------------------

def calculate_metrics(completed: List[Process]) -> Dict[str, float]:
    """Compute mean wait, P90 wait, Jain's fairness index, and makespan."""
    waits = np.array([p.wait_time for p in completed], dtype=float)
    if len(waits) == 0:
        return {"mean_wait": 0.0, "p90_wait": 0.0, "jain_fairness": 1.0, "makespan": 0.0}
    denom = len(waits) * np.sum(waits ** 2)
    fairness = 1.0 if denom == 0 else (np.sum(waits) ** 2) / denom
    return {
        "mean_wait": float(np.mean(waits)),
        "p90_wait": float(np.percentile(waits, 90)),
        "jain_fairness": float(fairness),
        "makespan": float(max(p.finish_time for p in completed if p.finish_time is not None)),
    }


# -----------------------------------------------------------------------------
# Part 3: MDP Environment
# -----------------------------------------------------------------------------

class SchedulerEnv:
    """
    Tabular-RL-friendly CPU scheduling environment.

    Actions: 0=FCFS selector, 1=SJF/shortest-remaining selector, 2=RR selector.
    Each environment step runs one CPU tick. The meta-scheduler can therefore
    switch traditional heuristics dynamically.
    """
    ACTIONS = ("FCFS", "SJF", "RR")

    def __init__(self, n_processes: int = 10, workload_mode: str = "mixed", context_switch_cost: float = 0.0, seed=None):
        self.n_processes = n_processes
        self.workload_mode = workload_mode
        self.context_switch_cost = context_switch_cost
        self.rng = np.random.default_rng(seed)
        self.avg_edges = (3, 8, 15, 30)
        self.wait_edges = (5, 15, 30, 60)
        self.state_space_dims = (n_processes + 1, len(self.avg_edges) + 1, len(self.wait_edges) + 1)
        self.n_actions = len(self.ACTIONS)

    def reset(self, workload: Optional[List[Process]] = None) -> Tuple[int, int, int]:
        self.processes = clone_workload(workload) if workload is not None else generate_workload(self.n_processes, self.rng, self.workload_mode)
        self.time = 0
        self.done = False
        self.completed_count = 0
        self.last_pid: Optional[int] = None
        self.context_switches = 0
        self.rr_queue: deque[int] = deque()
        self.added: set[int] = set()
        self._add_arrivals()
        return self._state()

    def _add_arrivals(self) -> None:
        for p in self.processes:
            if p.pid not in self.added and p.arrival <= self.time:
                self.rr_queue.append(p.pid)
                self.added.add(p.pid)

    def _ready(self) -> List[Process]:
        return [p for p in self.processes if p.arrival <= self.time and p.remaining > 0]

    def _current_wait(self, p: Process) -> int:
        return max(0, self.time - p.arrival - p.executed)

    @staticmethod
    def _bucket(x: float, edges: Tuple[int, ...]) -> int:
        if x <= edges[0]: return 0
        if x <= edges[1]: return 1
        if x <= edges[2]: return 2
        if x <= edges[3]: return 3
        return 4

    def _state(self) -> Tuple[int, int, int]:
        ready = self._ready()
        qlen = min(len(ready), self.n_processes)
        avg_remaining = (sum(p.remaining for p in ready) / len(ready)) if ready else 0.0
        max_wait = max((self._current_wait(p) for p in ready), default=0)
        return (qlen, self._bucket(avg_remaining, self.avg_edges), self._bucket(max_wait, self.wait_edges))

    def _select_process(self, action: int) -> Process:
        ready = self._ready()
        if action == 0:
            return min(ready, key=lambda p: (p.arrival, p.pid))
        if action == 1:
            # At each micro-decision the SJF selector uses remaining time; this lets
            # the meta-agent approximate SRTF without adding a separate action.
            return min(ready, key=lambda p: (p.remaining, p.arrival, p.pid))

        ready_ids = {p.pid for p in ready}
        for _ in range(len(self.rr_queue) + self.n_processes + 1):
            if not self.rr_queue:
                for p in sorted(ready, key=lambda x: (x.arrival, x.pid)):
                    self.rr_queue.append(p.pid)
            pid = self.rr_queue.popleft()
            if pid in ready_ids:
                return self.processes[pid]
        return min(ready, key=lambda p: (p.arrival, p.pid))

    def step(self, action: int):
        if self.done:
            return self._state(), 0.0, True, {}

        self._add_arrivals()
        if not self._ready():
            future_arrivals = [p.arrival for p in self.processes if p.remaining > 0 and p.arrival > self.time]
            if future_arrivals:
                self.time = min(future_arrivals)
                self._add_arrivals()
            if not self._ready():
                self.done = True
                return self._state(), 0.0, True, {}

        p = self._select_process(action)
        switched = self.last_pid is not None and self.last_pid != p.pid
        if switched:
            self.context_switches += 1

        if p.start_time is None:
            p.start_time = self.time

        p.remaining -= 1
        p.executed += 1
        self.time += 1
        self._add_arrivals()

        if action == 2 and p.remaining > 0:
            self.rr_queue.append(p.pid)
        if p.remaining == 0:
            p.finish_time = self.time
            self.completed_count += 1
            self.rr_queue = deque(pid for pid in self.rr_queue if pid != p.pid)
        self.last_pid = p.pid

        ready_after = self._ready()
        waits = [self._current_wait(x) for x in ready_after]
        total_delay = sum(waits)
        max_wait = max(waits, default=0)

        # Reward shaping: linear total-delay penalty plus a quadratic max-wait
        # starvation penalty. The quadratic term makes one starving process
        # increasingly expensive even when average waiting time stays low.
        reward = -(
            0.035 * total_delay
            + 0.80 * (max_wait / 10.0) ** 2
            + self.context_switch_cost * (1 if switched else 0)
        )

        done = self.completed_count == len(self.processes)
        if done:
            self.done = True
            m = calculate_metrics(self.processes)
            reward -= 0.30 * m["mean_wait"] + 0.08 * m["p90_wait"] + 12.0 * (1.0 - m["jain_fairness"])

        return self._state(), float(reward), done, {
            "pid": p.pid,
            "action": self.ACTIONS[action],
            "time": self.time,
            "switched": switched,
        }

    def completed_processes(self) -> List[Process]:
        return sorted(self.processes, key=lambda p: p.pid)


# -----------------------------------------------------------------------------
# Part 4: Q-learning meta-agent
# -----------------------------------------------------------------------------

class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration and hard cutoff."""
    def __init__(self, state_space_dims, n_actions: int, alpha: float = 0.15, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_min: float = 0.02, eps_decay: float = 0.9995, seed=None):
        self.q = np.zeros(tuple(state_space_dims) + (n_actions,), dtype=float)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: Tuple[int, int, int], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q[state]
        # Random tie-breaking prevents a permanent bias toward action 0 early in training.
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> float:
        target = reward if done else reward + self.gamma * np.max(self.q[next_state])
        index = state + (action,)
        td_error = target - self.q[index]
        self.q[index] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self, episode: int, hard_cutoff: Optional[int] = None) -> None:
        if hard_cutoff is not None and episode >= hard_cutoff:
            self.epsilon = 0.0
        else:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# -----------------------------------------------------------------------------
# Part 5: Training, evaluation, and plotting
# -----------------------------------------------------------------------------

def train_agent(episodes: int = 20_000, n_processes: int = 10, seed: int = 2026):
    env = SchedulerEnv(n_processes=n_processes, workload_mode="mixed", seed=seed)
    agent = QLearningAgent(env.state_space_dims, env.n_actions, seed=seed)
    rng = np.random.default_rng(seed + 1)
    hard_cutoff = int(0.85 * episodes)
    history = []
    action_counts = Counter()

    for ep in range(episodes):
        workload = generate_workload(n=n_processes, rng=rng, mode="mixed")
        state = env.reset(workload)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            action_counts[env.ACTIONS[action]] += 1
        m = calculate_metrics(env.completed_processes())
        history.append({
            "episode": ep,
            "mean_wait": m["mean_wait"],
            "p90_wait": m["p90_wait"],
            "jain_fairness": m["jain_fairness"],
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "steps": steps,
        })
        agent.decay_epsilon(ep, hard_cutoff=hard_cutoff)

    return agent, pd.DataFrame(history), action_counts


def evaluate_baselines(workloads: List[List[Process]]) -> pd.DataFrame:
    rows = []
    simulators = {
        "FCFS": simulate_fcfs,
        "SJF": simulate_sjf,
        "RR(q=4)": lambda w: simulate_rr(w, quantum=4),
        "Random Agent": simulate_random_agent,
    }
    for policy, simulator in simulators.items():
        for i, workload in enumerate(workloads):
            completed = simulator(workload, seed=i) if policy == "Random Agent" else simulator(workload)
            row = calculate_metrics(completed)
            row["policy"] = policy
            row["episode"] = i
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_agent(agent: QLearningAgent, workloads: List[List[Process]]) -> Tuple[pd.DataFrame, Counter]:
    rows = []
    action_counts = Counter()
    for i, workload in enumerate(workloads):
        env = SchedulerEnv(n_processes=len(workload), seed=1000 + i)
        state = env.reset(workload)
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            action_counts[env.ACTIONS[action]] += 1
            state, _, done, _ = env.step(action)
        row = calculate_metrics(env.completed_processes())
        row["policy"] = "RL Meta-Scheduler"
        row["episode"] = i
        rows.append(row)
    return pd.DataFrame(rows), action_counts


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby("policy")
        .agg(
            mean_wait=("mean_wait", "mean"),
            p90_wait=("p90_wait", "mean"),
            jain_fairness=("jain_fairness", "mean"),
            makespan=("makespan", "mean"),
        )
        .reset_index()
        .sort_values("mean_wait")
    )


def moving_average(series: pd.Series, window: int = 250) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_convergence(history: pd.DataFrame, baseline_summary: pd.DataFrame, out_path: str = "convergence_plot.png", window: int = 250) -> str:
    plt.figure(figsize=(10, 6))
    plt.plot(history["episode"], moving_average(history["mean_wait"], window=window), label=f"RL moving average ({window})")
    for _, row in baseline_summary.iterrows():
        if row["policy"] != "RL Meta-Scheduler":
            plt.axhline(row["mean_wait"], linestyle="--", linewidth=1, label=row["policy"])
    plt.title("KernelMind Meta-Scheduler convergence")
    plt.xlabel("Training episode")
    plt.ylabel("Mean wait time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def run_full_experiment(train_episodes: int = 20_000, eval_episodes: int = 500, seed: int = 2026, out_dir: str = "."):
    import os
    os.makedirs(out_dir, exist_ok=True)
    agent, history, training_action_counts = train_agent(episodes=train_episodes, seed=seed)

    eval_rng = np.random.default_rng(seed + 999)
    workloads = [generate_workload(n=10, rng=eval_rng, mode="mixed") for _ in range(eval_episodes)]
    baseline_df = evaluate_baselines(workloads)
    agent_df, eval_action_counts = evaluate_agent(agent, workloads)
    all_results = pd.concat([baseline_df, agent_df], ignore_index=True)
    summary = summarize_results(all_results)

    history_path = os.path.join(out_dir, "training_history.csv")
    metrics_path = os.path.join(out_dir, "comparative_metrics.csv")
    actions_path = os.path.join(out_dir, "action_distribution.csv")
    q_path = os.path.join(out_dir, "q_table.npy")
    plot_path = os.path.join(out_dir, "convergence_plot.png")

    history.to_csv(history_path, index=False)
    summary.to_csv(metrics_path, index=False)
    total_eval_actions = sum(eval_action_counts.values())
    action_df = pd.DataFrame({
        "action": list(SchedulerEnv.ACTIONS),
        "eval_count": [eval_action_counts[a] for a in SchedulerEnv.ACTIONS],
        "eval_fraction": [eval_action_counts[a] / total_eval_actions if total_eval_actions else 0 for a in SchedulerEnv.ACTIONS],
    })
    action_df.to_csv(actions_path, index=False)
    np.save(q_path, agent.q)
    plot_convergence(history, summary, out_path=plot_path, window=250)

    return {
        "agent": agent,
        "history": history,
        "summary": summary,
        "all_results": all_results,
        "eval_action_counts": eval_action_counts,
        "training_action_counts": training_action_counts,
        "plot_path": plot_path,
        "history_path": history_path,
        "metrics_path": metrics_path,
        "actions_path": actions_path,
        "q_path": q_path,
    }


if __name__ == "__main__":
    outputs = run_full_experiment(train_episodes=20_000, eval_episodes=500, seed=2026, out_dir=".")
    print(outputs["summary"].to_string(index=False))
    print("\nEvaluation action counts:", outputs["eval_action_counts"])
