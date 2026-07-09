from __future__ import annotations

import numpy as np

from .config import SchedulerConfig
from .process import Process


def ready_queue_to_state(ready_queue: list[Process], cfg: SchedulerConfig = SchedulerConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Convert the first N ready processes to a fixed tensor and mask.

    Features per process:
    0. normalized remaining burst time
    1. normalized wait time
    2. normalized priority
    3. normalized virtual runtime
    4. normalized arrival time
    """
    state = np.zeros((cfg.max_queue_size, cfg.feature_dim), dtype=np.float32)
    mask = np.zeros(cfg.max_queue_size, dtype=np.bool_)

    for slot, p in enumerate(ready_queue[: cfg.max_queue_size]):
        state[slot, 0] = min(float(p.remaining_time or 0) / cfg.max_burst, 1.0)
        state[slot, 1] = min(float(p.wait_time) / max(cfg.max_burst, 1), 1.0)
        state[slot, 2] = min(float(p.priority) / max(cfg.max_priority, 1), 1.0)
        state[slot, 3] = min(float(p.virtual_runtime) / max(cfg.max_burst, 1), 1.0)
        state[slot, 4] = min(float(p.arrival_time) / max(cfg.max_arrival, 1), 1.0)
        mask[slot] = True
    return state, mask
