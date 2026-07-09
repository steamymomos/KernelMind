from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SchedulerConfig:
    max_queue_size: int = 10
    feature_dim: int = 5
    max_burst: int = 60
    max_arrival: int = 30
    max_priority: int = 10
    starvation_threshold: int = 20
    rr_quantum: int = 3
    n_processes: int = 10
    arrival_window: int = 30
    burst_min: int = 1
    burst_max: int = 20
    priority_min: int = 1
    priority_max: int = 10
    memory_cap: int = 18
    memory_min: int = 1
    memory_max: int = 6


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 300
    batch_size: int = 32
    gamma: float = 0.96
    learning_rate: float = 1e-4
    replay_capacity: int = 30_000
    min_replay_size: int = 200
    target_update_steps: int = 500
    epsilon_start: float = 1.00
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    max_steps_per_episode: int = 2_000
    optimize_every_steps: int = 8
    hidden_dim: int = 64
    embed_dim: int = 32
    num_heads: int = 4
    seed: int = 7
    train_seed_offset: int = 1_000
    test_seed_offset: int = 20_000
    test_workloads: int = 60
    moving_average_window: int = 20

    def with_seed(self, seed: int, episodes: int | None = None) -> "TrainingConfig":
        return replace(
            self,
            seed=seed,
            episodes=self.episodes if episodes is None else episodes,
            train_seed_offset=50_000 + seed * 1_000,
        )
