"""Checkpoint validation: verify ARIS-Lite strictly improves min-clearance over plain A*."""

import unittest
import numpy as np
from src.grid import Grid
from src.astar import astar
from src.costmap import CostMap
from src.aris_planner import aris_plan, compute_path_clearance_metrics, compute_path_length
from src.debug_viz import show


class TestCheckpoint(unittest.TestCase):
    def test_aris_lite_beats_astar_min_clearance(self):
        # 20x20 grid
        occ = np.zeros((20, 20), dtype=np.int8)

        # Place a square obstacle block in the center: rows 6..12, cols 6..12
        occ[6:13, 6:13] = 1

        start = (2, 2)
        goal = (17, 17)

        grid = Grid(occ, start=start, goal=goal)
        costmap = CostMap(grid, r_safe=3.0, w_r=5.0, p=2.0)

        # 1. Plan with plain A*
        astar_path, astar_nodes = astar(grid, start=start, goal=goal)
        self.assertGreater(len(astar_path), 0, "A* should find a path")
        astar_min_c, astar_avg_c = compute_path_clearance_metrics(astar_path, costmap)
        astar_len = compute_path_length(astar_path)

        # 2. Plan with ARIS-Lite
        aris_path, aris_nodes, _ = aris_plan(
            grid, start=start, goal=goal, r_safe=3.0, w_r=5.0, p=2.0, costmap=costmap
        )
        self.assertGreater(len(aris_path), 0, "ARIS-Lite should find a path")
        aris_min_c, aris_avg_c = compute_path_clearance_metrics(aris_path, costmap)
        aris_len = compute_path_length(aris_path)

        print("\n" + "=" * 55)
        print("CHECKPOINT COMPARISON: A* vs ARIS-Lite")
        print("=" * 55)
        print(f"Plain A*     -> Min Clearance: {astar_min_c:.3f}, Avg Clearance: {astar_avg_c:.3f}, Length: {astar_len:.2f}, Nodes: {astar_nodes}")
        print(f"ARIS-Lite    -> Min Clearance: {aris_min_c:.3f}, Avg Clearance: {aris_avg_c:.3f}, Length: {aris_len:.2f}, Nodes: {aris_nodes}")
        print("=" * 55)

        # Save debug visualization for verification
        show(grid, path=astar_path, title=f"A* Path (Min Clearance: {astar_min_c:.2f})", save_path="results/checkpoint_astar.png")
        show(grid, path=aris_path, title=f"ARIS-Lite Path (Min Clearance: {aris_min_c:.2f})", save_path="results/checkpoint_aris.png")

        # Strict assertion: ARIS-Lite min clearance MUST BE GREATER than A* min clearance
        self.assertGreater(
            aris_min_c,
            astar_min_c,
            f"ARIS-Lite min clearance ({aris_min_c:.3f}) should strictly exceed A* ({astar_min_c:.3f})"
        )


if __name__ == "__main__":
    unittest.main()
