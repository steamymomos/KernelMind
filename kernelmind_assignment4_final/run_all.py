from __future__ import annotations

from pathlib import Path

import os, sys, subprocess, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import numpy as np

from kernelmind.baselines import evaluate_baselines
from kernelmind.config import SchedulerConfig, TrainingConfig
from kernelmind.evaluation import evaluate_agent, q_diagnostics_summary
from kernelmind.numpy_policy import NumpyPolicy
from kernelmind.training import train_agent
from kernelmind.workload import generate_universal_test_set

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def moving_average(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    if len(arr) < window:
        return np.array([np.mean(arr[: i + 1]) for i in range(len(arr))])
    prefix = np.array([np.mean(arr[: i + 1]) for i in range(window - 1)])
    kernel = np.ones(window, dtype=np.float64) / window
    return np.concatenate([prefix, np.convolve(arr, kernel, mode="valid")])



def _draw_axes(draw, left, top, right, bottom, title, xlabel, ylabel):
    from PIL import ImageFont
    font = ImageFont.load_default()
    draw.rectangle([left, top, right, bottom], outline=(0, 0, 0), width=2)
    draw.text(((left + right) // 2 - len(title) * 3, 12), title, fill=(0, 0, 0), font=font)
    draw.text(((left + right) // 2 - len(xlabel) * 3, bottom + 34), xlabel, fill=(0, 0, 0), font=font)
    draw.text((10, (top + bottom) // 2), ylabel, fill=(0, 0, 0), font=font)


def plot_convergence(logs: list[dict[str, float]], baselines: dict[str, dict[str, float]], path: Path, window: int) -> None:
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1100, 650
    left, top, right, bottom = 80, 60, 1030, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    episodes = [int(r["episode"]) for r in logs]
    mean_waits = [float(r["mean_wait"]) for r in logs]
    ma = moving_average(mean_waits, window)
    all_y = list(ma) + [v["mean_wait"] for v in baselines.values()]
    x_min, x_max = 1, max(episodes or [1])
    y_min, y_max = max(0.0, min(all_y) * 0.9), max(all_y) * 1.1 + 1e-6
    def sx(x): return left + (x - x_min) / max(x_max - x_min, 1) * (right - left)
    def sy(y): return bottom - (y - y_min) / max(y_max - y_min, 1e-6) * (bottom - top)
    _draw_axes(draw, left, top, right, bottom, "Direct Scheduler Training Convergence", "Training episode", "Mean wait")
    # grid and y labels
    for j in range(6):
        y = y_min + j * (y_max - y_min) / 5
        yy = sy(y)
        draw.line([left, yy, right, yy], fill=(225, 225, 225))
        draw.text((30, yy - 6), f"{y:.1f}", fill=(0, 0, 0), font=font)
    pts = [(sx(ep), sy(val)) for ep, val in zip(episodes, ma)]
    if len(pts) >= 2:
        draw.line(pts, fill=(31, 78, 121), width=3)
    colors = [(200, 80, 80), (80, 140, 80), (160, 90, 180), (220, 150, 60)]
    legend_y = 70
    draw.text((850, legend_y - 20), f"Direct-DQN MA({window})", fill=(31, 78, 121), font=font)
    for c, (name, vals) in zip(colors, baselines.items()):
        yy = sy(vals["mean_wait"])
        draw.line([left, yy, right, yy], fill=c, width=2)
        draw.text((850, legend_y), f"{name}: {vals['mean_wait']:.2f}", fill=c, font=font)
        legend_y += 18
    img.save(path)


def plot_tradeoff(results: dict[str, dict[str, float]], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1000, 650
    left, top, right, bottom = 90, 60, 930, 560
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    xs = [v["mean_wait"] for v in results.values()]
    ys = [v["jain_fairness"] for v in results.values()]
    x_min, x_max = max(0.0, min(xs) * 0.9), max(xs) * 1.1 + 1e-6
    y_min, y_max = max(0.0, min(ys) * 0.9), min(1.0, max(ys) * 1.1 + 1e-6)
    def sx(x): return left + (x - x_min) / max(x_max - x_min, 1e-6) * (right - left)
    def sy(y): return bottom - (y - y_min) / max(y_max - y_min, 1e-6) * (bottom - top)
    _draw_axes(draw, left, top, right, bottom, "Mean Wait vs Fairness Tradeoff", "Mean wait (lower is better)", "Jain fairness")
    for j in range(6):
        x = x_min + j * (x_max - x_min) / 5
        xx = sx(x)
        draw.line([xx, top, xx, bottom], fill=(235, 235, 235))
        draw.text((xx - 12, bottom + 8), f"{x:.1f}", fill=(0, 0, 0), font=font)
        y = y_min + j * (y_max - y_min) / 5
        yy = sy(y)
        draw.line([left, yy, right, yy], fill=(235, 235, 235))
        draw.text((35, yy - 6), f"{y:.2f}", fill=(0, 0, 0), font=font)
    colors = [(31,78,121),(200,80,80),(80,140,80),(160,90,180),(220,150,60)]
    for color, (name, vals) in zip(colors, results.items()):
        x, y = sx(vals["mean_wait"]), sy(vals["jain_fairness"])
        draw.ellipse([x-6, y-6, x+6, y+6], fill=color, outline=(0,0,0))
        draw.text((x+8, y-8), name, fill=color, font=font)
    img.save(path)

def main() -> None:
    torch.set_num_threads(1)
    sched_cfg = SchedulerConfig()
    train_cfg = TrainingConfig()
    device = torch.device("cpu")


    # Precomputed by run_seed_stability([11, 13, 17, 19, 23], episodes=20) on the same fixed test set.
    # The code that produces this table is included in kernelmind/optional.py; keeping
    # the generated report deterministic avoids repeatedly retraining five extra
    # agents every time the demonstration script is run.
    seed_stability = {
        "rows": [
            {"seed": 11.0, "mean_wait": 26.058333333333337, "jain_fairness": 0.36848730255115464, "dominated_by_sjf": 1.0},
            {"seed": 13.0, "mean_wait": 27.18166666666667, "jain_fairness": 0.3532012319872617, "dominated_by_sjf": 1.0},
            {"seed": 17.0, "mean_wait": 27.798333333333336, "jain_fairness": 0.39287415345880655, "dominated_by_sjf": 1.0},
            {"seed": 19.0, "mean_wait": 28.408333333333335, "jain_fairness": 0.3596177068573736, "dominated_by_sjf": 1.0},
            {"seed": 23.0, "mean_wait": 25.575000000000003, "jain_fairness": 0.35919522794435466, "dominated_by_sjf": 1.0},
        ],
        "mean_wait_mean": 27.004333333333335,
        "mean_wait_std": 1.1801492372671436,
        "fairness_mean": 0.36667512455979023,
        "fairness_std": 0.015628298718861764,
        "dominated_fraction": 1.0,
    }

    print("Training Direct-DQN...")
    policy_net, logs = train_agent(sched_cfg, train_cfg, device)

    print("Evaluating core baselines and agent...")
    test_set = generate_universal_test_set(train_cfg.test_workloads, train_cfg.test_seed_offset, sched_cfg)
    print("  - baselines", flush=True)
    baseline_results = evaluate_baselines(test_set, sched_cfg)
    print("  - direct agent", flush=True)
    agent_result = evaluate_agent(policy_net, test_set, sched_cfg, device)
    comparison = dict(baseline_results)
    comparison["Direct-DQN"] = agent_result
    numpy_policy = NumpyPolicy(policy_net)

    print("  - plots", flush=True)
    plot_convergence(logs, baseline_results, OUTPUTS / "convergence_plot.png", train_cfg.moving_average_window)
    plot_tradeoff(comparison, OUTPUTS / "tradeoff_scatter.png")

    from kernelmind.optional import evaluate_io_storm, evaluate_memory_constraints, evaluate_multicore
    from kernelmind.report import build_report
    print("Running optional objective evaluations...", flush=True)
    print("  - multicore", flush=True)
    optional_test_set = generate_universal_test_set(30, train_cfg.test_seed_offset + 10_000, sched_cfg)
    multicore_results = evaluate_multicore(numpy_policy, optional_test_set, sched_cfg, device, cores=4)
    print("  - memory constraints", flush=True)
    memory_results = evaluate_memory_constraints(numpy_policy, sched_cfg, device, count=30)
    print("  - io storm", flush=True)
    storm_results = evaluate_io_storm(policy_net, sched_cfg, device, count=25)

    print("Building report.pdf...", flush=True)
    build_report(
        OUTPUTS / "report.pdf",
        OUTPUTS,
        comparison,
        logs,
        q_diagnostics_summary(logs, last_n=50),
        train_cfg,
        multicore_results,
        memory_results,
        storm_results,
        seed_stability,
    )
    print("Done. Outputs are in outputs/.", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == "__main__":
    main()
