from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Process:
    """Simulated CPU process.

    virtual_runtime stores the cumulative ticks of actual CPU service for this
    process. It is deliberately separate from wait_time, which stores time spent
    ready-but-not-running. Optional experiments also use memory_required,
    swap_enter_time, swap_wait_time, and last_core.
    """

    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 1
    remaining_time: int | None = None
    wait_time: int = 0
    finish_time: int | None = None
    virtual_runtime: int = 0
    memory_required: int = 1
    swap_enter_time: int | None = None
    swap_wait_time: int = 0
    last_core: int | None = None
    migrations: int = 0

    def __post_init__(self) -> None:
        if self.remaining_time is None:
            self.remaining_time = self.burst_time
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.burst_time <= 0:
            raise ValueError("burst_time must be positive")
        if self.priority <= 0:
            raise ValueError("priority must be positive")
        if self.memory_required <= 0:
            raise ValueError("memory_required must be positive")

    @property
    def completed(self) -> bool:
        return self.remaining_time == 0

    @property
    def turnaround_time(self) -> int:
        if self.finish_time is None:
            raise ValueError("turnaround_time requested before completion")
        return self.finish_time - self.arrival_time

    @property
    def calculated_wait_time(self) -> int:
        if self.finish_time is None:
            raise ValueError("wait_time requested before completion")
        return self.finish_time - self.arrival_time - self.burst_time

    def clone_fresh(self) -> "Process":
        return replace(
            self,
            remaining_time=self.burst_time,
            wait_time=0,
            finish_time=None,
            virtual_runtime=0,
            swap_enter_time=None,
            swap_wait_time=0,
            last_core=None,
            migrations=0,
        )
