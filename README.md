# KernelMind Midterm Submission - SoC 2026

This repository contains my midterm submission for the **SoC-26 KernelMind** project.

The project explores operating-system scheduling, reinforcement learning based control, and neural scheduling agents. The work progresses from basic process abstractions and traditional scheduling theory to tabular reinforcement learning, meta-scheduling, and finally a direct neural CPU scheduler.

## Completed Work

### Assignment 1: Process Abstraction and Scheduling Foundations

Assignment 1 covered the operating-system fundamentals required for the later scheduling assignments.

Implemented and studied:

- Process abstraction and process lifecycle.
- POSIX process APIs such as `fork()`, `wait()`, and related behavior.
- Parent-child process execution flow.
- Process states such as ready, running, blocked, and terminated.
- Context switching and CPU dispatching.
- Basic scheduling theory and motivation behind CPU schedulers.
- Conceptual comparison of classical scheduling policies.

This assignment established the base understanding of how processes are represented and managed by an operating system.

---

### Assignment 2: The Adrian Descent

Assignment 2 implemented a tabular Q-learning control problem in a physics-based environment.

Completed components:

- Custom environment with state, action, transition, and reward definitions.
- Stochastic wind disturbance to make the control problem non-deterministic.
- Tabular Q-learning agent.
- Epsilon-greedy exploration strategy.
- Reward shaping for stable learning.
- Training loop with episode-wise performance logging.
- Evaluation of the trained policy.
- Generated plots showing training behavior.
- Metrics and report explaining the design choices and final performance.

This assignment introduced reinforcement learning through a smaller control problem before applying RL to CPU scheduling.

---

### Assignment 3: Hybrid Meta-Scheduler

Assignment 3 implemented an RL-based CPU scheduling meta-agent.

Instead of directly selecting a process, the agent selected among traditional scheduling heuristics.

Completed components:

- Process representation and workload generator.
- Classical CPU scheduling baselines:
  - First Come First Serve (FCFS)
  - Shortest Job First (SJF)
  - Round Robin (RR)
  - Random Scheduler
- Meta-scheduler environment where the RL agent chooses which heuristic to invoke.
- Tabular Q-learning based scheduler-selection agent.
- State discretization for queue-level scheduling features.
- Reward function balancing wait time and fairness.
- Metrics suite:
  - Mean Wait Time
  - P90 Wait Time
  - Jain’s Fairness Index
- Training loop and evaluation pipeline.
- Baseline comparison table.
- Convergence plots.
- Final report explaining implementation, results, and design decisions.

This assignment showed how reinforcement learning can control scheduling indirectly by selecting from a fixed menu of hand-written scheduling policies.

---

### Assignment 4: The Direct Neural Scheduler

Assignment 4 extends the previous meta-scheduler into a direct-control neural CPU scheduler.

Unlike Assignment 3, the agent no longer selects among FCFS, SJF, or Round Robin. Instead, the agent directly observes the ready queue and chooses which specific process should run next at every tick.

Completed components:

#### 1. Extended Process Representation

The process abstraction was extended with additional runtime information needed for direct neural scheduling.

Implemented process fields include:

- Process ID.
- Arrival time.
- Burst time.
- Remaining burst time.
- Priority.
- Wait time.
- Finish time.
- Completion status.
- `virtual_runtime`, which stores the cumulative number of CPU ticks received by that process.

The `virtual_runtime` field is used for fairness-aware reward shaping.

#### 2. Workload Generator

The workload generator from Assignment 3 was reused and extended where required.

It supports generating reproducible CPU scheduling workloads with:

- Multiple processes.
- Randomized arrival times.
- Randomized burst times.
- Randomized priorities.
- Fixed random seeds for repeatable experiments.

A universal test set is used so that all schedulers are evaluated on the same workloads.

#### 3. Direct-Control MDP Environment

A new direct-control scheduling environment was implemented.

The environment exposes:

- `reset()`
- `step(action)`

At each tick, the environment:

1. Builds the current ready queue.
2. Converts the visible queue into a fixed-size state tensor.
3. Accepts an action representing the selected process slot.
4. Executes the selected process for one CPU tick.
5. Updates remaining burst time.
6. Updates wait times of non-running ready processes.
7. Updates the selected process’s `virtual_runtime`.
8. Admits newly arrived processes.
9. Computes the reward.
10. Returns the next state, validity mask, reward, and done flag.

The action space is direct:  
the action is an integer index into the visible ready queue.

#### 4. State Tensor and Action Masking

The ready queue is converted into a fixed-size tensor using the top `N` visible processes.

Each process slot contains normalized features such as:

- Remaining burst time.
- Wait time.
- Priority.
- Virtual runtime.
- Arrival-time related information.

Unused slots are padded with zeros.

A boolean validity mask is generated alongside the tensor:

- Valid slots correspond to real processes.
- Invalid slots correspond to padding.

The mask is used to prevent the neural network from selecting padded queue positions.

#### 5. Fairness-Aware Reward Function

The reward function combines efficiency and fairness.

The main reward terms include:

- Per-tick penalty to discourage slow completion.
- Completion reward when a process finishes.
- Penalty based on virtual-runtime variance across the ready queue.
- Starvation penalty when a process waits beyond a fixed threshold.
- Bounded penalty terms to avoid unstable reward spikes.

The virtual-runtime variance term encourages the agent to distribute CPU service more evenly, instead of only minimizing aggregate wait time.

#### 6. DirectSchedulerNet

A neural network scheduler was implemented to replace the Assignment 3 Q-table.

The network architecture includes:

- Per-process feature projection layer.
- Positional encoding.
- Multi-head self-attention layer over the ready queue.
- Feedforward output head.
- One Q-value output per visible process slot.

The self-attention layer allows the model to compare processes against one another instead of treating the flattened queue as a fixed-position vector.

#### 7. Masking Inside the Network

Two separate masking mechanisms are used:

1. **Attention key padding mask**

   Prevents padded process slots from affecting attention computation.

2. **Output Q-value mask**

   Forces Q-values of padded slots to `-inf` before action selection.

Both masks are required. The attention mask prevents padded slots from influencing internal representations, while the output mask prevents the agent from selecting invalid padded actions.

#### 8. Double DQN Training Loop

The agent is trained using Double DQN.

Implemented training components:

- Experience replay buffer.
- Policy network.
- Target network.
- Epsilon-greedy exploration.
- Epsilon decay.
- Bellman update.
- Double DQN target computation.
- Periodic target-network synchronization.
- Q-value magnitude logging for stability diagnosis.
- Episode-wise metric tracking.

Double DQN is used to reduce overestimation bias by selecting the best next action using the policy network and evaluating it using the target network.

#### 9. Evaluation Metrics

The metrics suite from Assignment 3 was reused.

Reported metrics include:

- Mean Wait Time.
- P90 Wait Time.
- Jain’s Fairness Index.

These metrics are computed for:

- FCFS
- SJF
- Round Robin
- Random Scheduler
- Direct Neural Scheduler

#### 10. Baseline Benchmarking

The trained neural scheduler is compared against traditional CPU scheduling baselines on the same fixed universal test set.

Baselines included:

- FCFS
- SJF
- Round Robin
- Random

The final evaluation produces:

- A comparison table.
- A convergence plot.
- A tradeoff scatter plot of Mean Wait Time vs Jain’s Fairness Index.

#### 11. Required Plots

Assignment 4 includes the following generated plots:

- Training convergence plot showing moving average Mean Wait Time over episodes.
- Horizontal baseline reference lines for FCFS, SJF, RR, and Random.
- Tradeoff scatter plot comparing Mean Wait Time and Jain’s Fairness Index for all policies.

#### 12. Report

A complete `report.pdf` is included.

The report contains:

- Explanation of the implementation.
- Description of the direct-control environment.
- Explanation of the neural network architecture.
- Reward function design.
- Double DQN training details.
- Answers to all four required design questions.
- Final comparison table.
- Analysis of the convergence plot.
- Analysis of the tradeoff scatter plot.
- Honest discussion of whether the neural agent improves over, matches, or is dominated by the baselines.
- Discussion of training stability and Q-value behavior.

#### 13. Optional Objectives

The Assignment 4 submission also includes discussion and implementation support for optional extensions:

- Symmetric multiprocessing / multi-core scheduling.
- Memory and resource constraints.
- I/O Storm workload stress test.
- Seed stability as a first-class metric.

These optional objectives extend the direct scheduler beyond the single-core uniform-workload setup and test whether the learned policy generalizes under more difficult conditions.

---

## Repository Contents

The repository includes:

- Python source code for all implemented assignments.
- Reinforcement learning agents.
- CPU scheduling environments.
- Baseline scheduling algorithms.
- Training scripts.
- Evaluation scripts.
- Generated plots.
- Trained tables or model checkpoints where applicable.
- Reports for the assignments.
- Metrics and comparison outputs.

## Reproducibility

The code is structured so that experiments can be reproduced using fixed seeds and predefined workloads.

For the scheduling assignments, all baseline schedulers and RL-based agents are evaluated on shared workload sets to ensure fair comparison.

## Summary

Across the completed assignments, the project builds progressively from operating-system fundamentals to reinforcement learning based scheduling:

1. Assignment 1 establishes process and scheduling foundations.
2. Assignment 2 introduces tabular Q-learning in a controlled environment.
3. Assignment 3 applies RL to select among classical CPU scheduling heuristics.
4. Assignment 4 builds a direct neural scheduler that selects processes directly using a self-attention based Double DQN agent.

The final result is a complete KernelMind submission containing implementations, reports, plots, metrics, and reproducible experiments for the completed assignments.
