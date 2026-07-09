from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import SchedulerConfig, TrainingConfig
from .environment import DirectSchedulerEnvironment
from .metrics import calculate_metrics
from .network import DirectSchedulerNet
from .workload import generate_workload


@dataclass
class Transition:
    state: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: list[Transition] = []

    def push(self, transition: Transition) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self.data, batch_size)

    def __len__(self) -> int:
        return len(self.data)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def epsilon_by_step(step: int, cfg: TrainingConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    frac = step / cfg.epsilon_decay_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(
    policy_net: DirectSchedulerNet,
    state: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> int:
    valid_actions = np.flatnonzero(mask)
    if len(valid_actions) == 0:
        raise RuntimeError("select_action called with no valid actions")
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions))
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        return int(torch.argmax(policy_net(s, m), dim=1).item())


def optimize_model(
    policy_net: DirectSchedulerNet,
    target_net: DirectSchedulerNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    train_cfg: TrainingConfig,
    rng: random.Random,
    device: torch.device,
) -> tuple[float, float]:
    if len(replay) < max(train_cfg.min_replay_size, train_cfg.batch_size):
        return 0.0, 0.0

    batch = replay.sample(train_cfg.batch_size, rng)
    states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
    masks = torch.tensor(np.stack([t.mask for t in batch]), dtype=torch.bool, device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
    next_masks = torch.tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.bool, device=device)

    q_sa = policy_net(states, masks).gather(1, actions).squeeze(1)

    with torch.no_grad():
        # Double DQN: policy network selects the next action, target network evaluates it.
        next_policy_q = policy_net(next_states, next_masks)
        next_actions = torch.argmax(next_policy_q, dim=1, keepdim=True)
        next_target_q = target_net(next_states, next_masks).gather(1, next_actions).squeeze(1)
        next_target_q = torch.where(dones, torch.zeros_like(next_target_q), next_target_q)
        target = rewards + train_cfg.gamma * next_target_q

    td_error = target - q_sa
    loss = nn.functional.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2.0)
    optimizer.step()
    return float(loss.item()), float(td_error.mean().detach().cpu().item())


def build_network(sched_cfg: SchedulerConfig, train_cfg: TrainingConfig, device: torch.device) -> DirectSchedulerNet:
    return DirectSchedulerNet(
        max_queue_size=sched_cfg.max_queue_size,
        feature_dim=sched_cfg.feature_dim,
        embed_dim=train_cfg.embed_dim,
        num_heads=train_cfg.num_heads,
        hidden_dim=train_cfg.hidden_dim,
    ).to(device)


def train_agent(
    sched_cfg: SchedulerConfig = SchedulerConfig(),
    train_cfg: TrainingConfig = TrainingConfig(),
    device: torch.device | None = None,
) -> tuple[DirectSchedulerNet, list[dict[str, float]]]:
    set_global_seeds(train_cfg.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(train_cfg.seed)

    policy_net = build_network(sched_cfg, train_cfg, device)
    target_net = build_network(sched_cfg, train_cfg, device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=train_cfg.learning_rate, weight_decay=1e-4)
    replay = ReplayBuffer(train_cfg.replay_capacity)
    logs: list[dict[str, float]] = []
    global_step = 0

    for episode in range(train_cfg.episodes):
        workload = generate_workload(train_cfg.train_seed_offset + episode, sched_cfg)
        env = DirectSchedulerEnvironment(workload, sched_cfg)
        state, mask = env.reset()
        episode_reward = 0.0
        episode_loss_sum = 0.0
        episode_td_sum = 0.0
        q_abs_samples: list[float] = []
        updates = 0
        done = False

        for _ in range(train_cfg.max_steps_per_episode):
            epsilon = epsilon_by_step(global_step, train_cfg)
            action = select_action(policy_net, state, mask, epsilon, rng, device)
            next_state, next_mask, reward, done = env.step(action)
            replay.push(Transition(state, mask, action, reward, next_state, next_mask, done))
            state, mask = next_state, next_mask
            episode_reward += reward

            if global_step % train_cfg.optimize_every_steps == 0:
                loss, td_mean = optimize_model(policy_net, target_net, optimizer, replay, train_cfg, rng, device)
                if loss > 0.0:
                    episode_loss_sum += loss
                    episode_td_sum += td_mean
                    updates += 1

            if global_step % train_cfg.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if global_step % 50 == 0:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                    q = policy_net(s, m)[0]
                    valid_q = q[torch.tensor(mask, dtype=torch.bool, device=device)]
                    if valid_q.numel() > 0:
                        q_abs_samples.append(float(valid_q.abs().mean().detach().cpu().item()))

            global_step += 1
            if done:
                break

        metrics = calculate_metrics(env.completed)
        logs.append(
            {
                "episode": float(episode + 1),
                "reward": float(episode_reward),
                "mean_wait": metrics["mean_wait"],
                "p90_wait": metrics["p90_wait"],
                "jain_fairness": metrics["jain_fairness"],
                "epsilon": float(epsilon_by_step(global_step, train_cfg)),
                "loss": float(episode_loss_sum / max(updates, 1)),
                "signed_td_error": float(episode_td_sum / max(updates, 1)),
                "q_abs_mean": float(np.mean(q_abs_samples)) if q_abs_samples else 0.0,
                "global_step": float(global_step),
            }
        )
    return policy_net, logs
