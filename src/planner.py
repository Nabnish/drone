"""Integration API for ARIS-Nav.

LOCKED CONTRACT: This is the ONLY module external simulation and visualization
consumers import from. Plain numpy arrays, tuples, and lists crossing boundary.
No rendering or GUI dependencies.
"""

from __future__ import annotations
import time
from typing import Optional, Union, Dict, Any
import numpy as np

from src.grid import Grid
from src.costmap import CostMap
from src.astar import AStarPlanner
from src.aris_planner import ARISPlanner, compute_path_clearance_metrics, compute_path_length


class _PlannerState:
    """Internal session state managing persistent planning context."""

    def __init__(self) -> None:
        self.grid: Optional[Grid] = None
        self.costmap: Optional[CostMap] = None
        self.start: Optional[tuple[int, int]] = None
        self.goal: Optional[tuple[int, int]] = None
        self.mode: str = "aris"
        self.astar_planner: AStarPlanner = AStarPlanner()
        self.aris_planner: ARISPlanner = ARISPlanner()
        self.last_path: list[tuple[int, int]] = []
        self.last_stats: dict[str, Any] = {
            "planning_time_ms": 0.0,
            "replanning_time_ms": 0.0,
            "nodes_expanded": 0,
            "replan_nodes_expanded": 0,
            "path_cost": 0.0,
        }

    def reset(self) -> None:
        self.grid = None
        self.costmap = None
        self.start = None
        self.goal = None
        self.mode = "aris"
        self.last_path = []
        self.astar_planner = AStarPlanner()
        self.aris_planner = ARISPlanner()


_state = _PlannerState()


def get_planner_state() -> _PlannerState:
    """Access internal state for diagnostic/benchmark telemetry."""
    return _state


def plan(
    occupancy: np.ndarray,
    start: tuple,
    goal: tuple,
    mode: str = "aris",
) -> list[tuple]:
    """Compute an initial path from start to goal on the given occupancy grid.

    Args:
        occupancy: 2D numpy array (0 = free, 1 = obstacle).
        start: (row, col) start coordinates.
        goal: (row, col) goal coordinates.
        mode: Planning algorithm ("astar", "aris", or "aris-scratch").

    Returns:
        Ordered list of (row, col) tuples from start to goal, or [] if no path.
    """
    start_tuple = (int(start[0]), int(start[1]))
    goal_tuple = (int(goal[0]), int(goal[1]))
    mode_str = mode.lower().strip()

    valid_modes = ("astar", "aris", "aris-scratch", "aris-lite", "aris-incremental")
    if mode_str not in valid_modes:
        raise ValueError(f"Unsupported planning mode '{mode}'. Choose from {valid_modes}.")

    grid = Grid(occupancy, start=start_tuple, goal=goal_tuple)
    costmap = CostMap(grid, r_safe=3.0, w_r=5.0, p=2.0)

    _state.grid = grid
    _state.costmap = costmap
    _state.start = start_tuple
    _state.goal = goal_tuple
    _state.mode = mode_str

    t0 = time.perf_counter()

    if mode_str == "astar":
        path, nodes_expanded, path_cost = _state.astar_planner.plan(
            grid, start=start_tuple, goal=goal_tuple
        )
    else:  # aris variants
        path, nodes_expanded, path_cost = _state.aris_planner.plan(
            grid, start=start_tuple, goal=goal_tuple, costmap=costmap
        )

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    _state.last_path = path
    _state.last_stats = {
        "planning_time_ms": elapsed_ms,
        "replanning_time_ms": 0.0,
        "nodes_expanded": nodes_expanded,
        "replan_nodes_expanded": 0,
        "path_cost": path_cost,
    }

    return [(int(r), int(c)) for r, c in path]


def replan(
    changed_cells: list[tuple[tuple, bool]],
    current_pos: tuple,
) -> list[tuple]:
    """Replan from current_pos to original goal given dynamic obstacle updates.

    Args:
        changed_cells: List of [((row, col), is_now_occupied), ...]
        current_pos: (row, col) current position of the agent.

    Returns:
        Updated ordered path of (row, col) tuples from current_pos to original goal.
    """
    if _state.grid is None or _state.goal is None or _state.costmap is None:
        raise RuntimeError("Must call plan() before replan().")

    current_tuple = (int(current_pos[0]), int(current_pos[1]))
    norm_changed: list[tuple[tuple[int, int], bool]] = [
        ((int(cell[0]), int(cell[1])), bool(occ)) for cell, occ in changed_cells
    ]

    t0 = time.perf_counter()

    if _state.mode == "astar":
        for cell, occ in norm_changed:
            _state.grid.set_obstacle(cell, occ)
        _state.costmap.recompute()
        path, nodes_expanded, path_cost = _state.astar_planner.plan(
            _state.grid, start=current_tuple, goal=_state.goal
        )
    elif _state.mode in ("aris-scratch", "aris-lite"):
        # Replan from scratch using fresh ARIS without search tree reuse
        for cell, occ in norm_changed:
            _state.grid.set_obstacle(cell, occ)
        _state.costmap.recompute()
        fresh_planner = ARISPlanner(r_safe=3.0, w_r=5.0, p=2.0)
        path, nodes_expanded, path_cost = fresh_planner.plan(
            _state.grid, start=current_tuple, goal=_state.goal, costmap=_state.costmap
        )
    else:  # "aris" / "aris-incremental"
        path, nodes_expanded, path_cost = _state.aris_planner.replan_incremental(
            grid=_state.grid,
            costmap=_state.costmap,
            changed_cells=norm_changed,
            current_pos=current_tuple,
            goal=_state.goal,
        )

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    _state.last_path = path
    _state.last_stats["replanning_time_ms"] = elapsed_ms
    _state.last_stats["replan_nodes_expanded"] = nodes_expanded
    _state.last_stats["path_cost"] = path_cost

    return [(int(r), int(c)) for r, c in path]
