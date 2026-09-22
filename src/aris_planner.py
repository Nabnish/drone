"""ARIS-Lite path planner: safety-aware A* accumulating CostMap risk costs."""

from __future__ import annotations
import heapq
import math
from typing import Optional, Union
import numpy as np

from src.grid import Grid
from src.astar import octile_distance
from src.costmap import CostMap


def compute_path_clearance_metrics(
    path: list[tuple[int, int]], costmap: CostMap
) -> tuple[float, float]:
    """Compute (min_clearance, avg_clearance) for a path.

    Returns (0.0, 0.0) if path is empty.
    """
    if not path:
        return 0.0, 0.0
    clearances = [costmap.clearance(cell) for cell in path]
    return float(min(clearances)), float(sum(clearances) / len(clearances))


def compute_path_length(path: list[tuple[int, int]]) -> float:
    """Compute Euclidean trajectory length of the path."""
    if len(path) <= 1:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        length += math.hypot(r2 - r1, c2 - c1)
    return length


class ARISPlanner:
    """ARIS-Lite: Safety-aware A* accumulating non-linear risk costs in g(n)."""

    def __init__(
        self,
        r_safe: float = 3.0,
        w_r: float = 5.0,
        p: float = 2.0,
        allow_corner_cut: bool = True,
    ) -> None:
        self.r_safe = r_safe
        self.w_r = w_r
        self.p = p
        self.allow_corner_cut = allow_corner_cut

        # Stored search state for incremental replanning reuse
        self.last_g_score: dict[tuple[int, int], float] = {}
        self.last_came_from: dict[tuple[int, int], tuple[int, int]] = {}
        self.last_goal: Optional[tuple[int, int]] = None
        self.last_path: list[tuple[int, int]] = []

    def plan(
        self,
        grid: Grid,
        start: Optional[tuple[int, int]] = None,
        goal: Optional[tuple[int, int]] = None,
        costmap: Optional[CostMap] = None,
    ) -> tuple[list[tuple[int, int]], int, float]:
        """Plan a safety-aware path using ARIS-Lite.

        Identical to standard A* except g(n) accumulates CostMap.cost(cell)
        instead of uniform step cost.

        Returns:
            (path, nodes_expanded, path_cost)
        """
        start_cell = start if start is not None else grid.start
        goal_cell = goal if goal is not None else grid.goal

        if start_cell is None or goal_cell is None:
            raise ValueError("Start and goal must be specified either in grid or as arguments")

        if not grid.is_free(start_cell) or not grid.is_free(goal_cell):
            return [], 0, float("inf")

        if start_cell == goal_cell:
            self.last_path = [start_cell]
            self.last_goal = goal_cell
            return [start_cell], 0, 0.0

        if costmap is None:
            costmap = CostMap(grid, r_safe=self.r_safe, w_r=self.w_r, p=self.p)

        counter = 0
        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        h_start = octile_distance(start_cell, goal_cell)
        heapq.heappush(open_heap, (h_start, counter, start_cell))

        g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        closed_set: set[tuple[int, int]] = set()

        nodes_expanded = 0

        while open_heap:
            f_curr, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            if current == goal_cell:
                # Reconstruct path
                path = [current]
                curr = current
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()

                # Cache search state for replanning
                self.last_g_score = g_score
                self.last_came_from = came_from
                self.last_goal = goal_cell
                self.last_path = path

                return path, nodes_expanded, g_score[goal_cell]

            for neighbor, base_step_cost in grid.get_neighbors(
                current, allow_corner_cut=self.allow_corner_cut
            ):
                if neighbor in closed_set:
                    continue

                step_cost = costmap.cost(neighbor, base_step_cost=base_step_cost)
                if math.isinf(step_cost):
                    continue

                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + octile_distance(neighbor, goal_cell)
                    counter += 1
                    heapq.heappush(open_heap, (f_neighbor, counter, neighbor))

        # Goal unreachable
        self.last_g_score = g_score
        self.last_came_from = came_from
        self.last_goal = goal_cell
        self.last_path = []
        return [], nodes_expanded, float("inf")

    def replan_incremental(
        self,
        grid: Grid,
        costmap: CostMap,
        changed_cells: list[tuple[tuple[int, int], bool]],
        current_pos: tuple[int, int],
        goal: Optional[tuple[int, int]] = None,
    ) -> tuple[list[tuple[int, int]], int, float]:
        """Bounded incremental replanning from current_pos to goal.

        This is a simplified, locality-bounded re-search inspired by D* Lite/LPA*'s
        reuse principle, NOT a formally correct implementation of either algorithm.

        Workflow:
        1. Apply dynamic cell changes to grid.
        2. Identify dirty region within radius ~ 2 * R_safe of changed cells.
        3. Recompute clearance, risk, and cost in the local patch.
        4. Invalidate dirty cells and their descendants in the cached search tree.
        5. Reuse clean cells' g-values and parent pointers; re-expand from current_pos
           and affected dirty boundaries.
        6. Re-solve to goal.

        Args:
            grid: Grid instance.
            costmap: CostMap instance.
            changed_cells: List of ((row, col), is_now_occupied).
            current_pos: Agent's current cell (row, col).
            goal: Destination cell (row, col). Defaults to self.last_goal.

        Returns:
            (path, nodes_expanded, path_cost)
        """
        target_goal = goal if goal is not None else self.last_goal
        if target_goal is None:
            raise ValueError("Goal must be specified or cached from a previous plan() call")

        # 1. Update grid with changed cells
        raw_changed = [cell for cell, _ in changed_cells]
        for cell, occupied in changed_cells:
            grid.set_obstacle(cell, occupied)

        # 2 & 3. Identify dirty patch and recompute costmap
        patch_radius = 2.0 * self.r_safe
        dirty_cells = costmap.update_patch(raw_changed, patch_radius=patch_radius)
        for c in raw_changed:
            dirty_cells.add(c)

        # Basic validity checks
        if not grid.is_free(current_pos) or not grid.is_free(target_goal):
            return [], 0, float("inf")

        if current_pos == target_goal:
            self.last_path = [current_pos]
            return [current_pos], 0, 0.0

        # Check if the existing path is still completely valid and avoids all dirty/occupied cells
        existing_path_valid = False
        if self.last_path and current_pos in self.last_path:
            idx = self.last_path.index(current_pos)
            remaining_subpath = self.last_path[idx:]
            # If no cell in remaining subpath is an obstacle or inside dirty region
            if not any(grid.is_obstacle(c) or c in dirty_cells for c in remaining_subpath):
                existing_path_valid = True
                # Path is clean; we can safely return remaining path with 0 expansions
                path_cost = sum(
                    costmap.cost(
                        remaining_subpath[i + 1],
                        base_step_cost=math.hypot(
                            remaining_subpath[i + 1][0] - remaining_subpath[i][0],
                            remaining_subpath[i + 1][1] - remaining_subpath[i][1],
                        ),
                    )
                    for i in range(len(remaining_subpath) - 1)
                )
                self.last_path = remaining_subpath
                return remaining_subpath, 0, path_cost

        # 4 & 5. Graph reuse: Identify clean downstream nodes on last_path
        downstream_targets: dict[tuple[int, int], int] = {}
        if self.last_path and current_pos in self.last_path:
            cur_idx = self.last_path.index(current_pos)
            # Find the first obstacle encounter along the path
            obs_idx = -1
            for i in range(cur_idx, len(self.last_path)):
                if self.last_path[i] in dirty_cells or grid.is_obstacle(self.last_path[i]):
                    obs_idx = i
                    break

            if obs_idx != -1:
                # Downstream nodes that are strictly beyond the dirty/obstacle area
                for i in range(obs_idx + 1, len(self.last_path)):
                    cell = self.last_path[i]
                    if cell not in dirty_cells and grid.is_free(cell):
                        # Verify the rest of the path from this cell to goal is completely free
                        if all(grid.is_free(c) for c in self.last_path[i:]):
                            downstream_targets[cell] = i

        # Priority queue for search from current_pos
        counter = 0
        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        h_start = octile_distance(current_pos, target_goal)
        heapq.heappush(open_heap, (h_start, counter, current_pos))

        g_score: dict[tuple[int, int], float] = {current_pos: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        closed_set: set[tuple[int, int]] = set()
        nodes_expanded = 0

        # 6. Re-solve from current_pos to goal or downstream clean target
        while open_heap:
            f_curr, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            # Check if we reached goal or a clean downstream reconnect node
            if current == target_goal or current in downstream_targets:
                # Reconstruct local bypass path from current_pos to current
                bypass = [current]
                curr = current
                while curr in came_from and curr != current_pos:
                    curr = came_from[curr]
                    bypass.append(curr)
                if bypass[-1] != current_pos:
                    bypass.append(current_pos)
                bypass.reverse()

                if current in downstream_targets and current != target_goal:
                    target_idx = downstream_targets[current]
                    tail = self.last_path[target_idx + 1 :]
                    full_path = bypass + tail
                else:
                    full_path = bypass

                # Cache search state
                self.last_g_score = g_score
                self.last_came_from = came_from
                self.last_goal = target_goal
                self.last_path = full_path

                path_cost = sum(
                    costmap.cost(
                        full_path[i + 1],
                        base_step_cost=math.hypot(
                            full_path[i + 1][0] - full_path[i][0],
                            full_path[i + 1][1] - full_path[i][1],
                        ),
                    )
                    for i in range(len(full_path) - 1)
                )

                return full_path, nodes_expanded, path_cost

            for neighbor, base_step_cost in grid.get_neighbors(
                current, allow_corner_cut=self.allow_corner_cut
            ):
                if neighbor in closed_set:
                    continue

                step_cost = costmap.cost(neighbor, base_step_cost=base_step_cost)
                if math.isinf(step_cost):
                    continue

                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + octile_distance(neighbor, target_goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_neighbor, counter, neighbor))

        # If bounded incremental reuse didn't reach goal (e.g. tree disconnected),
        # fall back to fresh ARIS plan from current_pos
        fallback_path, fallback_nodes, fallback_cost = self.plan(
            grid, start=current_pos, goal=target_goal, costmap=costmap
        )
        return fallback_path, nodes_expanded + fallback_nodes, fallback_cost


def aris_plan(
    grid: Grid,
    start: Optional[tuple[int, int]] = None,
    goal: Optional[tuple[int, int]] = None,
    r_safe: float = 3.0,
    w_r: float = 5.0,
    p: float = 2.0,
    costmap: Optional[CostMap] = None,
    allow_corner_cut: bool = True,
) -> tuple[list[tuple[int, int]], int, float]:
    """Convenience function to run ARIS-Lite planning."""
    planner = ARISPlanner(
        r_safe=r_safe, w_r=w_r, p=p, allow_corner_cut=allow_corner_cut
    )
    return planner.plan(grid, start=start, goal=goal, costmap=costmap)
