# KernelMind Assignment 3 - Hybrid Meta-Scheduler

This archive contains a complete notebook-style solution for the assignment.

## Main files

- `KernelMind_Assignment3.ipynb` - complete notebook with Part 1 to Part 5 code and report answers.
- `kernelmind_assignment3.py` - same implementation as a reusable Python script/module.
- `report.pdf` - final written report with implementation summary, design-question answers, comparison table, action distribution, and convergence plot.
- `convergence_plot.png` - training moving-average plot against static baselines.
- `comparative_metrics.csv` - held-out evaluation summary.
- `training_history.csv` - per-episode training metrics from the 20,000-episode run.
- `action_distribution.csv` - greedy policy action counts and fractions.
- `all_episode_metrics.csv` - episode-level baseline and RL evaluation metrics.
- `q_table.npy` - trained tabular Q-table.

## How to reproduce

Install the usual Python scientific stack if needed:

```bash
pip install numpy pandas matplotlib
```

Then run:

```bash
python kernelmind_assignment3.py
```

Or open `KernelMind_Assignment3.ipynb`. The notebook loads the precomputed outputs by default. To rerun training from inside the notebook, set:

```python
RUN_FULL_EXPERIMENT = True
```

The full experiment uses 20,000 training episodes and 500 evaluation episodes.
