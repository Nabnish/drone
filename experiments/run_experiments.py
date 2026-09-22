"""Batch experiment driver for ARIS-Nav.

Executes multiple randomized trials across benchmarks to verify statistical consistency
and compute mean and standard deviation for all telemetry metrics.
"""

from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.benchmark import run_all_benchmarks, run_benchmark_trial, build_open_field_scenario, build_narrow_corridor_scenario, build_dead_end_maze_scenario


def run_batch_experiments(num_runs: int = 5, output_csv: str = "results/batch_experiments.csv"):
    """Execute multiple repeated runs across all scenarios and planners."""
    print(f"Starting batch experiments ({num_runs} runs per scenario/planner)...")

    records = defaultdict(lambda: defaultdict(list))

    scenarios = [
        ("open_field", build_open_field_scenario),
        ("narrow_corridor", build_narrow_corridor_scenario),
        ("dead_end_maze", build_dead_end_maze_scenario),
    ]

    planners = [
        ("Plain A*", "astar"),
        ("ARIS-Lite", "aris-scratch"),
        ("Incremental ARIS", "aris"),
    ]

    for run_idx in range(1, num_runs + 1):
        print(f"Executing trial {run_idx}/{num_runs}...")
        for sc_name, sc_factory in scenarios:
            for p_name, p_mode in planners:
                res = run_benchmark_trial(
                    scenario_name=sc_name,
                    occupancy_factory=sc_factory,
                    planner_name=p_name,
                    mode=p_mode,
                    save_viz=(run_idx == 1),  # Save visualization only on first run
                )
                key = (sc_name, p_name)
                records[key]["plan_ms"].append(res.planning_time_ms)
                records[key]["replan_ms"].append(res.replanning_time_ms)
                records[key]["length"].append(res.path_length)
                records[key]["min_c"].append(res.min_clearance)
                records[key]["avg_c"].append(res.avg_clearance)
                records[key]["nodes"].append(res.nodes_expanded)
                records[key]["coll"].append(res.collision_count)

    # Save to summary CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario",
            "Planner",
            "Plan Time Mean (ms)",
            "Plan Time Std (ms)",
            "Replan Time Mean (ms)",
            "Replan Time Std (ms)",
            "Path Length",
            "Min Clearance",
            "Avg Clearance",
            "Nodes Mean",
            "Collision Count",
        ])

        print("\n" + "=" * 125)
        print(f"{'BATCH EXPERIMENT STATISTICAL SUMMARY (' + str(num_runs) + ' TRIALS)':^125}")
        print("=" * 125)
        header = (
            f"| {'Scenario':<16} | {'Planner':<18} | {'Plan (ms)':<16} | {'Replan (ms)':<16} "
            f"| {'Length':<8} | {'Min Clr':<8} | {'Avg Clr':<8} | {'Nodes':<7} | {'Coll':<4} |"
        )
        print(header)
        print("|" + "-" * 18 + "|" + "-" * 20 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 6 + "|")

        for (sc_name, p_name), metrics in records.items():
            plan_mean = np.mean(metrics["plan_ms"])
            plan_std = np.std(metrics["plan_ms"])
            replan_mean = np.mean(metrics["replan_ms"])
            replan_std = np.std(metrics["replan_ms"])
            length = np.mean(metrics["length"])
            min_c = np.mean(metrics["min_c"])
            avg_c = np.mean(metrics["avg_c"])
            nodes_mean = int(round(np.mean(metrics["nodes"])))
            total_coll = sum(metrics["coll"])

            writer.writerow([
                sc_name,
                p_name,
                f"{plan_mean:.3f}",
                f"{plan_std:.3f}",
                f"{replan_mean:.3f}",
                f"{replan_std:.3f}",
                f"{length:.2f}",
                f"{min_c:.3f}",
                f"{avg_c:.3f}",
                nodes_mean,
                total_coll,
            ])

            plan_str = f"{plan_mean:.2f} +/- {plan_std:.2f}"
            replan_str = f"{replan_mean:.2f} +/- {replan_std:.2f}"
            row = (
                f"| {sc_name:<16} | {p_name:<18} | {plan_str:>16} | {replan_str:>16} "
                f"| {length:>8.2f} | {min_c:>8.3f} | {avg_c:>8.3f} | {nodes_mean:>7} | {total_coll:>4} |"
            )
            print(row)

        print("=" * 125 + "\n")
        print(f"Batch results saved to: {output_csv}")


if __name__ == "__main__":
    run_batch_experiments(num_runs=5)
