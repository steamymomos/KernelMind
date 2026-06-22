# The Adrian Descent — Complete Submission

This folder contains a complete solution for SoC '26 KernelMind Assignment 2.

## Files

- `env.py` — `ProbeEnv` with deterministic physics, stochastic Markov wind, reward structure, and ASCII renderer.
- `agent.py` — `ProbeAgent` with discretization, epsilon-greedy action selection, and Bellman Q-learning.
- `train.py` — training loop that runs 15,000 episodes by default and saves plots/metrics/Q-table.
- `run.py` — greedy evaluation loop with optional ASCII rendering.
- `bonus_variable_spin.py` — bonus OFF/50%/100% thrust setup.
- `report.md` — answers to all design questions and mathematical calculations.
- `learning_curve.png` — included sample learning curve; regenerate with `train.py`.

## How to run

```bash
python train.py --episodes 15000 --out-dir .
python run.py --q-table adrian_q_table.npz --no-render
```

For a faster smoke test:

```bash
python train.py --episodes 3000 --out-dir .
python run.py --q-table adrian_q_table.npz --no-render
```

## Bonus action space

```bash
python bonus_variable_spin.py
```
