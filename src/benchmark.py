"""Benchmark suite comparing Plain A*, ARIS-Lite, and Incremental ARIS across 3 scenarios."""

from __future__ import annotations
import csv
import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Callable, NamedTuple
import numpy as np

import src.planner as planner
from src.costmap import CostMap
from src.aris_planner import compute_path_clearance_metrics, compute_path_length
from src.debug_viz import show


class BenchmarkResult(NamedTuple):
    scenario: str
    planner: str
    planning_time_ms: float
    replanning_time_ms: float
    path_length: float
    min_clearance: float
    avg_clearance: float
    nodes_expanded: int
    collision_count: int


def build_open_field_scenario() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Scenario 1: Open field with scattered obstacles."""
    occ = np.zeros((35, 35), dtype=np.int8)
    start = (3, 3)
    goal = (31, 31)

    # Scattered obstacle clusters
    occ[10:15, 10:15] = 1
    occ[18:23, 20:25] = 1
    occ[25:29, 8:13] = 1
    occ[8:12, 24:28] = 1

    return occ, start, goal


def build_narrow_corridor_scenario() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Scenario 2: Narrow corridor vs wide safe bypass."""
    occ = np.zeros((35, 35), dtype=np.int8)
    start = (3, 17)
    goal = (31, 17)

    # Wall spanning horizontally at row 16-18
    occ[16:19, :] = 1

    # Primary narrow corridor in center (col 16-17)
    occ[16:19, 16:18] = 0

    # Wide bypass corridor on the left (col 4-9)
    occ[16:19, 4:10] = 0

    return occ, start, goal


def build_dead_end_maze_scenario() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Scenario 3: Maze with branching paths and potential dead-ends."""
    occ = np.zeros((35, 35), dtype=np.int8)
    start = (2, 2)
    goal = (32, 32)

    # Maze walls
    occ[6, 2:28] = 1
    occ[12, 8:33] = 1
    occ[18, 2:28] = 1
    occ[24, 8:33] = 1

    # Vertical connectors
    occ[7:12, 18:20] = 1
    occ[19:24, 14:16] = 1

    return occ, start, goal


def run_benchmark_trial(
    scenario_name: str,
    occupancy_factory: Callable[[], tuple[np.ndarray, tuple[int, int], tuple[int, int]]],
    planner_name: str,
    mode: str,
    save_viz: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark trial for a given scenario and planner mode."""
    occ, start, goal = occupancy_factory()

    # Step 1: Initial planning via locked planner API
    init_path = planner.plan(occ, start, goal, mode=mode)
    state = planner.get_planner_state()
    plan_time = state.last_stats["planning_time_ms"]
    plan_nodes = state.last_stats["nodes_expanded"]

    if not init_path:
        return BenchmarkResult(
            scenario=scenario_name,
            planner=planner_name,
            planning_time_ms=plan_time,
            replanning_time_ms=0.0,
            path_length=0.0,
            min_clearance=0.0,
            avg_clearance=0.0,
            nodes_expanded=plan_nodes,
            collision_count=1,
        )

    # Step 2: Simulate agent moving mid-way along initial path
    advance_steps = min(len(init_path) // 3, max(2, len(init_path) - 3))
    current_pos = init_path[advance_steps]

    # Pick dynamic obstacle cell slightly ahead on the path
    obstacle_idx = min(advance_steps + 2, len(init_path) - 2)
    obstacle_cell = init_path[obstacle_idx]

    # Step 3: Trigger replan via locked planner API
    replan_path = planner.replan([(obstacle_cell, True)], current_pos=current_pos)
    state = planner.get_planner_state()
    replan_time = state.last_stats["replanning_time_ms"]
    replan_nodes = state.last_stats["replan_nodes_expanded"]

    # Full executed trajectory
    executed_trajectory = init_path[:advance_steps] + replan_path

    # Verify collisions
    grid = state.grid
    collision_count = 0
    for cell in executed_trajectory:
        if grid.is_obstacle(cell):
            collision_count += 1

    # Calculate metrics
    costmap = state.costmap
    min_c, avg_c = compute_path_clearance_metrics(executed_trajectory, costmap)
    total_len = compute_path_length(executed_trajectory)
    total_nodes = plan_nodes + replan_nodes

    if save_viz and executed_trajectory:
        os.makedirs("results", exist_ok=True)
        img_filename = f"results/{scenario_name}_{mode.replace('-', '_')}.png"
        show(
            grid,
            path=executed_trajectory,
            title=f"{scenario_name.title()} - {planner_name}\n(MinC: {min_c:.2f}, AvgC: {avg_c:.2f})",
            save_path=img_filename,
        )

    return BenchmarkResult(
        scenario=scenario_name,
        planner=planner_name,
        planning_time_ms=round(plan_time, 3),
        replanning_time_ms=round(replan_time, 3),
        path_length=round(total_len, 2),
        min_clearance=round(min_c, 3),
        avg_clearance=round(avg_c, 3),
        nodes_expanded=total_nodes,
        collision_count=collision_count,
    )


def run_all_benchmarks(output_csv: str = "results/metrics.csv") -> list[BenchmarkResult]:
    """Execute all benchmark combinations and output results."""
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

    results: list[BenchmarkResult] = []

    for sc_name, sc_factory in scenarios:
        for p_name, p_mode in planners:
            res = run_benchmark_trial(sc_name, sc_factory, p_name, p_mode)
            results.append(res)

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario",
            "Planner",
            "Planning Time (ms)",
            "Replanning Time (ms)",
            "Path Length",
            "Min Clearance",
            "Avg Clearance",
            "Nodes Expanded",
            "Collision Count",
        ])
        for r in results:
            writer.writerow([
                r.scenario,
                r.planner,
                r.planning_time_ms,
                r.replanning_time_ms,
                r.path_length,
                r.min_clearance,
                r.avg_clearance,
                r.nodes_expanded,
                r.collision_count,
            ])

    # Print markdown summary table
    print("\n" + "=" * 110)
    print(f"{'ARIS-NAV BENCHMARK SUMMARY TABLE':^110}")
    print("=" * 110)
    header = (
        f"| {'Scenario':<16} | {'Planner':<18} | {'Plan (ms)':<10} | {'Replan (ms)':<11} "
        f"| {'Length':<8} | {'Min Clr':<8} | {'Avg Clr':<8} | {'Nodes':<6} | {'Coll':<4} |"
    )
    print(header)
    print("|" + "-" * 18 + "|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 6 + "|")

    for r in results:
        row = (
            f"| {r.scenario:<16} | {r.planner:<18} | {r.planning_time_ms:>10.3f} | {r.replanning_time_ms:>11.3f} "
            f"| {r.path_length:>8.2f} | {r.min_clearance:>8.3f} | {r.avg_clearance:>8.3f} | {r.nodes_expanded:>6} | {r.collision_count:>4} |"
        )
        print(row)
    print("=" * 110 + "\n")
    print(f"Results saved to: {output_csv}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
