"""Grid module representing a 2D occupancy grid world."""

from __future__ import annotations
import math
from typing import Optional


class Grid:
    """Wraps a 2D numpy occupancy array where 0 = free and 1 = obstacle.

    Supports start/goal tracking, 8-connected neighbor lookup with Euclidean
    step costs (1.0 orthogonal, sqrt(2) diagonal), and dynamic obstacle modification.
    """

    SQRT2 = math.sqrt(2.0)

    # (dr, dc, base_cost)
    ORTHOGONAL_DIRECTIONS = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
    ]

    DIAGONAL_DIRECTIONS = [
        (-1, -1, SQRT2),
        (-1, 1, SQRT2),
        (1, -1, SQRT2),
        (1, 1, SQRT2),
    ]

    ALL_DIRECTIONS = ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS

    def __init__(
        self,
        occupancy,
        start: Optional[tuple[int, int]] = None,
        goal: Optional[tuple[int, int]] = None,
    ) -> None:
        import numpy as np

        if not isinstance(occupancy, np.ndarray):
            occupancy = np.array(occupancy, dtype=np.int8)
        else:
            occupancy = occupancy.astype(np.int8, copy=True)

        if occupancy.ndim != 2:
            raise ValueError(f"Occupancy grid must be 2-dimensional, got ndim={occupancy.ndim}")

        self.occupancy = occupancy
        self._start = start
        self._goal = goal

    @property
    def rows(self) -> int:
        return self.occupancy.shape[0]

    @property
    def cols(self) -> int:
        return self.occupancy.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    @property
    def start(self) -> Optional[tuple[int, int]]:
        return self._start

    @start.setter
    def start(self, cell: Optional[tuple[int, int]]) -> None:
        if cell is not None and not self.in_bounds(cell):
            raise ValueError(f"Start cell {cell} is out of bounds for grid shape {self.shape}")
        self._start = cell

    @property
    def goal(self) -> Optional[tuple[int, int]]:
        return self._goal

    @goal.setter
    def goal(self, cell: Optional[tuple[int, int]]) -> None:
        if cell is not None and not self.in_bounds(cell):
            raise ValueError(f"Goal cell {cell} is out of bounds for grid shape {self.shape}")
        self._goal = cell

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        """Check if cell (row, col) is within grid boundaries."""
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: tuple[int, int]) -> bool:
        """Check if cell is in bounds and free (0)."""
        return self.in_bounds(cell) and self.occupancy[cell[0], cell[1]] == 0

    def is_obstacle(self, cell: tuple[int, int]) -> bool:
        """Check if cell is in bounds and an obstacle (1)."""
        return self.in_bounds(cell) and self.occupancy[cell[0], cell[1]] == 1

    def set_obstacle(self, cell: tuple[int, int], occupied: bool) -> None:
        """Dynamically set or clear an obstacle at the given cell."""
        if not self.in_bounds(cell):
            raise ValueError(f"Cell {cell} is out of bounds for grid shape {self.shape}")
        self.occupancy[cell[0], cell[1]] = 1 if occupied else 0

    def get_neighbors(
        self, cell: tuple[int, int], allow_corner_cut: bool = True
    ) -> list[tuple[tuple[int, int], float]]:
        """Return 8-connected free neighbors and their transition base step cost.

        Args:
            cell: Current cell (row, col).
            allow_corner_cut: If True, allows diagonal movement as long as destination
                is free. If False, disallows diagonal movement if cutting between
                adjacent orthogonal obstacles.

        Returns:
            List of ((neighbor_row, neighbor_col), step_cost)
        """
        r, c = cell
        neighbors: list[tuple[tuple[int, int], float]] = []

        for dr, dc, cost in self.ALL_DIRECTIONS:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if not self.is_free(neighbor):
                continue

            # If diagonal and corner cutting is disabled, check adjacent orthogonal cells
            if not allow_corner_cut and dr != 0 and dc != 0:
                # Squeeze through two obstacles: both (r+dr, c) and (r, c+dc) are obstacles
                if self.is_obstacle((r + dr, c)) and self.is_obstacle((r, c + dc)):
                    continue

            neighbors.append((neighbor, cost))

        return neighbors

    def copy(self) -> Grid:
        """Create a deep copy of the grid."""
        return Grid(
            occupancy=self.occupancy.copy(),
            start=self.start,
            goal=self.goal,
        )
