"""CostMap module calculating obstacle clearance, risk field, and safety costs."""

from __future__ import annotations
import math
from typing import Optional, Union
import numpy as np
from scipy.ndimage import distance_transform_edt
from src.grid import Grid


class CostMap:
    """Calculates clearance, risk, and safety-penalized edge traversal costs.

    Formulas:
        - clearance: Euclidean distance to nearest obstacle (cells) via distance_transform_edt.
        - risk(cell):
            0 if clearance >= R_safe
            else (R_safe - clearance) / R_safe, normalized to [0, 1].
        - cost(cell, base_step_cost):
            inf if occupied
            else base_step_cost * (1 + w_r * risk(cell) ** p)
    """

    def __init__(
        self,
        grid_or_occupancy: Union[Grid, np.ndarray],
        r_safe: float = 3.0,
        w_r: float = 5.0,
        p: float = 2.0,
    ) -> None:
        """Initialize CostMap.

        Args:
            grid_or_occupancy: Grid object or 2D numpy occupancy array (0=free, 1=obstacle).
            r_safe: Safety margin radius (default 3.0).
            w_r: Risk weight factor (default 5.0).
            p: Risk penalty exponent (default 2.0).
        """
        if isinstance(grid_or_occupancy, Grid):
            self.occupancy = grid_or_occupancy.occupancy
        else:
            self.occupancy = np.array(grid_or_occupancy, dtype=np.int8)

        self.r_safe = float(r_safe)
        self.w_r = float(w_r)
        self.p = float(p)

        self.clearance_map = np.zeros(self.occupancy.shape, dtype=np.float64)
        self.risk_map = np.zeros(self.occupancy.shape, dtype=np.float64)

        self.recompute()

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    @property
    def rows(self) -> int:
        return self.occupancy.shape[0]

    @property
    def cols(self) -> int:
        return self.occupancy.shape[1]

    def recompute(self) -> None:
        """Recompute full clearance and risk fields from occupancy array."""
        has_obstacles = np.any(self.occupancy == 1)

        if not has_obstacles:
            # When there are no obstacles in the world, clearance is infinite / very large
            self.clearance_map.fill(float("inf"))
            self.risk_map.fill(0.0)
            return

        # distance_transform_edt calculates distance to the nearest zero.
        # Inverted occupancy mask: 1 where free, 0 where obstacle.
        inv_mask = (self.occupancy == 0).astype(np.int8)
        self.clearance_map = distance_transform_edt(inv_mask)

        # risk(cell) = 0 if clearance >= R_safe, else (R_safe - clearance) / R_safe
        # Mask out any occupied cells (they have clearance 0, risk 1.0)
        risk = np.where(
            self.clearance_map >= self.r_safe,
            0.0,
            (self.r_safe - self.clearance_map) / self.r_safe,
        )
        self.risk_map = np.clip(risk, 0.0, 1.0)

    def clearance(self, cell: tuple[int, int]) -> float:
        """Get distance (in cells) to the nearest obstacle."""
        r, c = cell
        return float(self.clearance_map[r, c])

    def risk(self, cell: tuple[int, int]) -> float:
        """Get risk value in [0, 1] for the specified cell."""
        r, c = cell
        return float(self.risk_map[r, c])

    def cost(self, cell: tuple[int, int], base_step_cost: float = 1.0) -> float:
        """Get traversal cost to enter the specified cell.

        Args:
            cell: Destination cell (row, col).
            base_step_cost: Base Euclidean distance step cost (1.0 or sqrt(2)).

        Returns:
            inf if occupied, else base_step_cost * (1 + w_r * risk(cell) ** p).
        """
        r, c = cell
        if self.occupancy[r, c] == 1:
            return float("inf")

        r_val = self.risk_map[r, c]
        if r_val <= 0.0:
            return float(base_step_cost)

        return float(base_step_cost * (1.0 + self.w_r * (r_val**self.p)))

    def update_patch(
        self,
        changed_cells: list[tuple[int, int]],
        patch_radius: Optional[float] = None,
    ) -> set[tuple[int, int]]:
        """Bounded update: recomputes clearance/risk around changed cells.

        Returns:
            The set of all cells identified as dirty within the patch radius.
        """
        if patch_radius is None:
            patch_radius = 2.0 * self.r_safe

        radius_int = int(math.ceil(patch_radius))
        dirty_cells: set[tuple[int, int]] = set()

        for cr, cc in changed_cells:
            r_min = max(0, cr - radius_int)
            r_max = min(self.rows, cr + radius_int + 1)
            c_min = max(0, cc - radius_int)
            c_max = min(self.cols, cc + radius_int + 1)

            for r in range(r_min, r_max):
                for c in range(c_min, c_max):
                    # Check Euclidean distance to changed cell
                    if math.hypot(r - cr, c - cc) <= patch_radius:
                        dirty_cells.add((r, c))

        # Recompute clearance & risk
        # Note: on 2D grids, EDT is so fast (sub-millisecond) that full recompute
        # ensures zero edge artifacts while dirty_cells delineates the bounded re-expansion region.
        self.recompute()
        return dirty_cells
