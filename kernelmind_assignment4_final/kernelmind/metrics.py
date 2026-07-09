from __future__ import annotations

from typing import Iterable

import numpy as np

from .process import Process


def calculate_metrics(completed_processes: Iterable[Process]) -> dict[str, float]:
    completed = list(completed_processes)
    if not completed:
        return {"mean_wait": 0.0, "p90_wait": 0.0, "jain_fairness": 1.0, "avg_swap_wait": 0.0, "migrations_per_job": 0.0}

    waits = np.array([p.calculated_wait_time for p in completed], dtype=np.float64)
    mean_wait = float(np.mean(waits))
    p90_wait = float(np.percentile(waits, 90))

    # Jain's index is computed over responsiveness, not raw waits. This avoids
    # falsely calling uniformly terrible waiting times perfectly fair.
    responsiveness = 1.0 / (1.0 + waits)
    denom = len(responsiveness) * float(np.sum(responsiveness ** 2))
    jain = 0.0 if denom <= 1e-12 else float((float(np.sum(responsiveness)) ** 2) / denom)

    return {
        "mean_wait": mean_wait,
        "p90_wait": p90_wait,
        "jain_fairness": jain,
        "avg_swap_wait": float(np.mean([p.swap_wait_time for p in completed])),
        "migrations_per_job": float(np.sum([p.migrations for p in completed]) / len(completed)),
    }


def aggregate_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"mean_wait": 0.0, "p90_wait": 0.0, "jain_fairness": 1.0, "avg_swap_wait": 0.0, "migrations_per_job": 0.0}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    return {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
