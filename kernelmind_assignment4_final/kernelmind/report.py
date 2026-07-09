from __future__ import annotations

from pathlib import Path
from typing import Any
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

PAGE_W, PAGE_H = A4
MARGIN = 0.55 * inch
LINE = 12


def fmt(x: Any, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


class PDFWriter:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.y = PAGE_H - MARGIN
        self.page_no = 0
        self.new_page()

    def new_page(self) -> None:
        if self.page_no > 0:
            self.footer()
            self.c.showPage()
        self.page_no += 1
        self.y = PAGE_H - MARGIN
        self.c.setFont("Helvetica-Bold", 9)
        self.c.drawString(MARGIN, PAGE_H - 0.35 * inch, "SoC '26 KernelMind - Assignment 4")
        self.c.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.35 * inch, "The Direct Neural Scheduler")
        self.c.line(MARGIN, PAGE_H - 0.42 * inch, PAGE_W - MARGIN, PAGE_H - 0.42 * inch)
        self.y -= 0.25 * inch

    def footer(self) -> None:
        self.c.setFont("Helvetica", 8)
        self.c.line(MARGIN, 0.38 * inch, PAGE_W - MARGIN, 0.38 * inch)
        self.c.drawCentredString(PAGE_W / 2, 0.22 * inch, str(self.page_no))

    def ensure(self, h: float) -> None:
        if self.y - h < 0.55 * inch:
            self.new_page()

    def title(self, text: str) -> None:
        self.ensure(42)
        self.c.setFont("Helvetica-Bold", 18)
        self.c.drawCentredString(PAGE_W / 2, self.y, text)
        self.y -= 28

    def heading(self, text: str, level: int = 1) -> None:
        self.ensure(28)
        self.y -= 4
        self.c.setFont("Helvetica-Bold", 13 if level == 1 else 11)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= 16

    def para(self, text: str, width_chars: int = 104) -> None:
        lines = []
        for block in text.split("\n"):
            lines.extend(textwrap.wrap(block, width=width_chars) or [""])
        self.ensure(len(lines) * LINE + 6)
        self.c.setFont("Helvetica", 9.2)
        for line in lines:
            self.c.drawString(MARGIN, self.y, line)
            self.y -= LINE
        self.y -= 4

    def bullet(self, text: str) -> None:
        lines = textwrap.wrap(text, width=98)
        self.ensure(len(lines) * LINE + 4)
        self.c.setFont("Helvetica", 9.2)
        self.c.drawString(MARGIN + 8, self.y, "- ")
        for i, line in enumerate(lines):
            self.c.drawString(MARGIN + 20, self.y, line)
            self.y -= LINE
        self.y -= 2

    def table(self, rows: list[list[str]], col_widths: list[float]) -> None:
        row_h = 17
        total_h = row_h * len(rows) + 4
        self.ensure(total_h)
        x0 = MARGIN
        for r, row in enumerate(rows):
            x = x0
            if r == 0:
                self.c.setFillColor(colors.HexColor("#1f4e79"))
                self.c.rect(x0, self.y - row_h + 3, sum(col_widths), row_h, fill=1, stroke=0)
                self.c.setFillColor(colors.white)
                self.c.setFont("Helvetica-Bold", 8.2)
            else:
                self.c.setFillColor(colors.black)
                self.c.setFont("Helvetica", 8.2)
            for cell, w in zip(row, col_widths):
                self.c.drawString(x + 3, self.y - 10, str(cell)[:45])
                self.c.setStrokeColor(colors.HexColor("#b8c6d1"))
                self.c.rect(x, self.y - row_h + 3, w, row_h, fill=0, stroke=1)
                x += w
            self.y -= row_h
        self.y -= 8
        self.c.setFillColor(colors.black)

    def image(self, path: Path, caption: str, width: float = 6.6 * inch) -> None:
        if not path.exists():
            return
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        height = width * ih / iw
        self.ensure(height + 30)
        x = (PAGE_W - width) / 2
        self.c.drawImage(img, x, self.y - height, width=width, height=height, preserveAspectRatio=True)
        self.y -= height + 12
        self.c.setFont("Helvetica", 8)
        self.c.drawCentredString(PAGE_W / 2, self.y, caption)
        self.y -= 18

    def save(self) -> None:
        self.footer()
        self.c.save()


def basic_results_rows(results: dict[str, dict[str, float]], extras: list[str] | None = None) -> list[list[str]]:
    extras = extras or []
    header = ["Policy", "Mean Wait", "P90 Wait", "Jain Fairness"] + extras
    rows = [header]
    for name, vals in results.items():
        row = [name, fmt(vals.get("mean_wait", 0)), fmt(vals.get("p90_wait", 0)), fmt(vals.get("jain_fairness", 0))]
        for e in extras:
            key = "avg_swap_wait" if "Swap" in e else "migrations_per_job"
            row.append(fmt(vals.get(key, 0)))
        rows.append(row)
    return rows


def build_report(
    path: Path,
    outputs_dir: Path,
    comparison: dict[str, dict[str, float]],
    logs: list[dict[str, float]],
    q_summary: dict[str, float],
    train_cfg: Any,
    multicore_results: dict[str, dict[str, float]],
    memory_results: dict[str, dict[str, float]],
    storm_results: dict[str, dict[str, float]],
    seed_stability: dict[str, Any],
) -> None:
    pdf = PDFWriter(path)
    pdf.title("KernelMind Assignment 4: The Direct Neural Scheduler")
    pdf.heading("Implementation summary")
    pdf.para("This submission implements a direct-control CPU scheduler. The agent chooses a specific ready-queue process each tick, not a hand-written heuristic. The code includes the extended process model, workload generators, direct MDP environment, DirectSchedulerNet, Double DQN loop, baseline simulators, metrics, plots, and optional-objective evaluators.")
    for item in [
        "Process includes virtual_runtime, the cumulative CPU ticks actually received by each process.",
        "The state tensor encodes the top N=10 ready processes with normalized remaining time, wait time, priority, virtual runtime, and arrival time. Unused slots are zero-padded.",
        "The validity mask is used twice: as an attention key mask and as an output action mask that forces padded Q-values to a very negative value.",
        "The environment rejects invalid padded actions, executes one tick, updates waits and virtual runtime, admits arrivals, computes reward, and returns next state, mask, reward, and done.",
        "Training uses replay, target network, epsilon-greedy decay, Huber loss, gradient clipping, and Double DQN target computation.",
    ]:
        pdf.bullet(item)
    pdf.heading("Reward function")
    pdf.para("The reward is bounded by construction: tick penalty -0.020, queue penalty in [-0.040, 0], completion bonus +0.800, virtual-runtime variance penalty in [-0.180, 0], and starvation-cap penalty in [-0.250, 0]. The fairness term uses variance of virtual_runtime so it penalizes uneven accumulated CPU service, not just raw waiting.")

    pdf.new_page()
    pdf.heading("Answers to required design questions")
    pdf.heading("Design Question 1 - Cost of direct control", 2)
    pdf.para("The direct action space makes credit assignment harder because the agent chooses among up to N raw processes every tick. A bad choice at tick 1 of a 60-tick episode can affect roughly the next 59 rewards through changed waits, completions, queue composition, and service imbalance. Bellman backups propagate this blame one step at a time. In Assignment 3, one heuristic decision could control several ticks, so there were fewer decision points and a shorter effective credit chain.")
    pdf.heading("Design Question 2 - Reward shaping", 2)
    pdf.para("Starvation is addressed by two bounded signals: virtual-runtime variance and the wait-time starvation cap. The virtual-runtime term makes it expensive to repeatedly serve jobs that have already received CPU while another ready job remains underserved. This is less susceptible to the raw-wait trap because running a starved long job immediately reduces accumulated-service imbalance, even though the job may remain in the queue for many ticks.")
    pdf.heading("Design Question 3 - Diagnosing instability", 2)
    pdf.para(f"The code logs Q-value magnitude and signed TD-error. In this generated run, initial mean absolute Q was {fmt(q_summary['initial_q_abs'])}, the final-window mean absolute Q was {fmt(q_summary['final_q_abs'])}, signed TD-error was {fmt(q_summary['final_signed_td'])}, and Huber loss was {fmt(q_summary['final_loss'])}. The five-seed stability check produced mean wait std {fmt(seed_stability['mean_wait_std'])}, fairness std {fmt(seed_stability['fairness_std'])}, and dominated-by-SJF fraction {fmt(seed_stability['dominated_fraction'])}. If instability were worse, I would first inspect raw reward-term magnitudes under random policy and signed TD-error/target refresh behavior.")
    pdf.heading("Design Question 4 - Output analysis", 2)
    dqn, sjf = comparison["Direct-DQN"], comparison["SJF"]
    dominated = dqn["mean_wait"] > sjf["mean_wait"] and dqn["jain_fairness"] < sjf["jain_fairness"]
    pdf.para(("The Direct-DQN result is dominated by SJF on the core test set. " if dominated else "The Direct-DQN result is not strictly dominated by SJF on the core test set. ") + f"Direct-DQN mean wait={fmt(dqn['mean_wait'])}, fairness={fmt(dqn['jain_fairness'])}; SJF mean wait={fmt(sjf['mean_wait'])}, fairness={fmt(sjf['jain_fairness'])}. If the neural policy only ties or loses to a one-line heuristic, that is not proof of useful learning; next steps are longer stable training, smaller learning-rate sweeps, prioritized replay, and reward-weight sweeps.")

    pdf.new_page()
    pdf.heading("Core benchmark results")
    pdf.para(f"The generated run trains for {train_cfg.episodes} episodes and evaluates all policies on the same fixed universal test set.")
    pdf.table(basic_results_rows(comparison), [1.35*inch, 1.15*inch, 1.15*inch, 1.35*inch])
    pdf.image(outputs_dir / "convergence_plot.png", "Figure 1. Moving-average training mean wait with baseline reference lines.")
    pdf.image(outputs_dir / "tradeoff_scatter.png", "Figure 2. Mean wait versus Jain fairness for all core policies.")

    pdf.new_page()
    pdf.heading("Optional objective answers")
    pdf.heading("Symmetric multiprocessing", 2)
    pdf.para("Implemented a 4-core evaluator. Each policy ranks visible processes once per tick and greedily assigns the top distinct processes to cores. last_core and migrations are tracked, and all policies use the same core-assignment logic.")
    pdf.table(basic_results_rows(multicore_results, ["Migrations / Job"]), [1.1*inch, 1.0*inch, 1.0*inch, 1.15*inch, 1.2*inch])
    pdf.heading("Memory/resource constraints", 2)
    pdf.para("Implemented memory_required, a fixed memory cap, a swap queue, and deterministic first-fit admission every tick. The agent does not receive memory as an input feature, so swap-time differences are indirect consequences of clearing resident jobs.")
    pdf.table(basic_results_rows(memory_results, ["Avg Swap Wait"]), [1.1*inch, 1.0*inch, 1.0*inch, 1.15*inch, 1.2*inch])

    pdf.new_page()
    pdf.heading("I/O Storm stress test", 2)
    pdf.para("The I/O Storm generator creates 8 short jobs with bursts 1-2 and 2 CPU-heavy jobs with bursts 50-60, all arriving simultaneously. The network is evaluated without retraining, so a collapse here indicates out-of-distribution weakness rather than a training-set improvement.")
    pdf.table(basic_results_rows(storm_results), [1.35*inch, 1.15*inch, 1.15*inch, 1.35*inch])
    pdf.heading("Seed stability", 2)
    seed_rows = [["Seed", "Mean Wait", "Jain Fairness", "Dominated by SJF"]]
    for r in seed_stability["rows"]:
        seed_rows.append([str(int(r["seed"])), fmt(r["mean_wait"]), fmt(r["jain_fairness"]), "yes" if r["dominated_by_sjf"] else "no"])
    seed_rows.append(["Mean/Std", f"{fmt(seed_stability['mean_wait_mean'])}/{fmt(seed_stability['mean_wait_std'])}", f"{fmt(seed_stability['fairness_mean'])}/{fmt(seed_stability['fairness_std'])}", fmt(seed_stability["dominated_fraction"])])
    pdf.table(seed_rows, [1.0*inch, 1.15*inch, 1.15*inch, 1.45*inch])
    pdf.heading("Submitted files")
    pdf.para("The archive contains only the requested deliverables: source code, run_all.py, requirements.txt, convergence_plot.png, tradeoff_scatter.png, and report.pdf. It excludes caches, checkpoints, and unrelated intermediate artifacts.")
    pdf.save()
