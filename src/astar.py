"""Standard 8-connected A* path planner with octile heuristic."""

from __future__ import annotations
import heapq
import math
from typing import Optional
from src.grid import Grid

SQRT2 = math.sqrt(2.0)
SQRT2_MINUS_ONE = SQRT2 - 1.0


def octile_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Compute exact octile distance heuristic between two 2D grid cells.

    Admissible and consistent for 8-connected grids with costs 1.0 and sqrt(2).
    h = max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)
    """
    dr = abs(p1[0] - p2[0])
    dc = abs(p1[1] - p2[1])
    if dr > dc:
        return dr + SQRT2_MINUS_ONE * dc
    return dc + SQRT2_MINUS_ONE * dr


class AStarPlanner:
    """Standard 8-connected A* path planner (baseline/control group)."""

    def __init__(self, allow_corner_cut: bool = True) -> None:
        self.allow_corner_cut = allow_corner_cut

    def plan(
        self,
        grid: Grid,
        start: Optional[tuple[int, int]] = None,
        goal: Optional[tuple[int, int]] = None,
    ) -> tuple[list[tuple[int, int]], int, float]:
        """Find the shortest 8-connected path on the grid.

        Args:
            grid: Grid instance.
            start: Start cell (row, col). Defaults to grid.start.
            goal: Goal cell (row, col). Defaults to grid.goal.

        Returns:
            Tuple of (path, nodes_expanded, path_cost):
                path: List of (row, col) from start to goal, or [] if unreachable.
                nodes_expanded: Number of nodes expanded during search.
                path_cost: Total g-cost of the path, or float('inf') if no path.
        """
        start_cell = start if start is not None else grid.start
        goal_cell = goal if goal is not None else grid.goal

        if start_cell is None or goal_cell is None:
            raise ValueError("Start and goal must be specified either in grid or as arguments")

        if not grid.is_free(start_cell) or not grid.is_free(goal_cell):
            return [], 0, float("inf")

        if start_cell == goal_cell:
            return [start_cell], 0, 0.0

        # Open set priority queue: (f_score, tie_breaker, current_node)
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
                return path, nodes_expanded, g_score[goal_cell]

            for neighbor, step_cost in grid.get_neighbors(
                current, allow_corner_cut=self.allow_corner_cut
            ):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + octile_distance(neighbor, goal_cell)
                    counter += 1
                    heapq.heappush(open_heap, (f_neighbor, counter, neighbor))

        # Goal unreachable
        return [], nodes_expanded, float("inf")


def astar(
    grid: Grid,
    start: Optional[tuple[int, int]] = None,
    goal: Optional[tuple[int, int]] = None,
    allow_corner_cut: bool = True,
) -> tuple[list[tuple[int, int]], int]:
    """Convenience function for running standard A* search.

    Returns:
        (path, nodes_expanded)
    """
    planner = AStarPlanner(allow_corner_cut=allow_corner_cut)
    path, nodes_expanded, _ = planner.plan(grid, start=start, goal=goal)
    return path, nodes_expanded
