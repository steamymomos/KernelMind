from __future__ import annotations

import numpy as np
import torch

from .config import SchedulerConfig
from .environment import DirectSchedulerEnvironment
from .metrics import aggregate_metrics, calculate_metrics
from .network import DirectSchedulerNet
from .numpy_policy import NumpyPolicy
from .process import Process


def run_agent_episode(
    policy_net: DirectSchedulerNet,
    workload: list[Process],
    cfg: SchedulerConfig = SchedulerConfig(),
    device: torch.device | None = None,
    numpy_policy: NumpyPolicy | None = None,
) -> list[Process]:
    env = DirectSchedulerEnvironment(workload, cfg)
    state, mask = env.reset()
    done = False
    policy = numpy_policy if numpy_policy is not None else NumpyPolicy(policy_net)
    steps = 0
    while not done:
        q_values = policy.q_values(state, mask)
        action = int(np.argmax(q_values))
        state, mask, _, done = env.step(action)
        steps += 1
        if steps > cfg.n_processes * cfg.max_burst * 3:
            raise RuntimeError("agent episode exceeded safe step bound")
    return env.completed


def evaluate_agent(
    policy_net: DirectSchedulerNet,
    test_set: list[list[Process]],
    cfg: SchedulerConfig = SchedulerConfig(),
    device: torch.device | None = None,
) -> dict[str, float]:
    policy = NumpyPolicy(policy_net)
    return aggregate_metrics([calculate_metrics(run_agent_episode(policy_net, workload, cfg, device, policy)) for workload in test_set])


def q_diagnostics_summary(logs: list[dict[str, float]], last_n: int = 50) -> dict[str, float]:
    if not logs:
        return {"initial_q_abs": 0.0, "final_q_abs": 0.0, "final_signed_td": 0.0, "final_loss": 0.0}
    tail = logs[-min(last_n, len(logs)) :]
    return {
        "initial_q_abs": float(logs[0].get("q_abs_mean", 0.0)),
        "final_q_abs": float(np.mean([r.get("q_abs_mean", 0.0) for r in tail])),
        "final_signed_td": float(np.mean([r.get("signed_td_error", 0.0) for r in tail])),
        "final_loss": float(np.mean([r.get("loss", 0.0) for r in tail])),
    }
