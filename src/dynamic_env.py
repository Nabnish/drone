"""Dynamic environment utility for mid-run obstacle injection and replanning validation."""

from __future__ import annotations
from typing import Optional
import numpy as np

import src.planner as planner


class DynamicEnvironment:
    """Simulates a live grid environment for testing replanning without a full simulator.

    Allows stepping an agent along an active plan, dynamically injecting or clearing
    obstacles, and triggering planner.replan().
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        mode: str = "aris",
    ) -> None:
        self.occupancy = np.array(occupancy, dtype=np.int8)
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.mode = mode
        self.current_pos = self.start
        self.collision_count = 0
        self.execution_history: list[tuple[int, int]] = [self.current_pos]

        # Initial plan
        self.path = planner.plan(self.occupancy, self.start, self.goal, mode=self.mode)

    def step(self, num_steps: int = 1) -> tuple[int, int]:
        """Advance agent along the current path by num_steps.

        Returns:
            The new current position.
        """
        if not self.path:
            return self.current_pos

        for _ in range(num_steps):
            if self.current_pos == self.goal:
                break

            # Find next cell in path
            if self.current_pos in self.path:
                idx = self.path.index(self.current_pos)
                if idx + 1 < len(self.path):
                    next_cell = self.path[idx + 1]
                else:
                    break
            else:
                next_cell = self.path[0]

            # Check collision
            if self.occupancy[next_cell[0], next_cell[1]] == 1:
                self.collision_count += 1

            self.current_pos = next_cell
            self.execution_history.append(self.current_pos)

        return self.current_pos

    def inject_obstacle(self, cell: tuple[int, int]) -> list[tuple]:
        """Inject an obstacle at cell and invoke replan()."""
        c = (int(cell[0]), int(cell[1]))
        self.occupancy[c[0], c[1]] = 1
        self.path = planner.replan([(c, True)], current_pos=self.current_pos)
        return self.path

    def remove_obstacle(self, cell: tuple[int, int]) -> list[tuple]:
        """Remove an obstacle at cell and invoke replan()."""
        c = (int(cell[0]), int(cell[1]))
        self.occupancy[c[0], c[1]] = 0
        self.path = planner.replan([(c, False)], current_pos=self.current_pos)
        return self.path

    def inject_obstacle_ahead(
        self, steps_ahead: int = 2
    ) -> tuple[Optional[tuple[int, int]], list[tuple]]:
        """Inject an obstacle directly in the path steps_ahead from current_pos.

        Returns:
            (blocked_cell, updated_path)
        """
        if not self.path or self.current_pos not in self.path:
            return None, self.path

        idx = self.path.index(self.current_pos)
        target_idx = idx + steps_ahead
        if target_idx >= len(self.path) - 1:
            target_idx = len(self.path) - 2  # Don't overwrite goal

        if target_idx <= idx:
            return None, self.path

        blocked_cell = self.path[target_idx]
        new_path = self.inject_obstacle(blocked_cell)
        return blocked_cell, new_path

    def is_at_goal(self) -> bool:
        """Check if agent has reached the goal."""
        return self.current_pos == self.goal
