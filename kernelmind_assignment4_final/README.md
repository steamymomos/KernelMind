# KernelMind Assignment 4 - Direct Neural Scheduler

## Run

```bash
pip install -r requirements.txt
python run_all.py
```

The script trains the Direct-DQN scheduler, evaluates baselines and optional objectives, generates the two required plots, and writes `outputs/report.pdf`.

## Included deliverables

- Extended process representation and workload generators
- Direct-control MDP environment with bounded fairness reward
- `DirectSchedulerNet` and Double DQN training loop
- Evaluation and metrics suite
- Training/evaluation execution script
- `outputs/convergence_plot.png`
- `outputs/tradeoff_scatter.png`
- `outputs/report.pdf`
