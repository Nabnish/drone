import unittest
import numpy as np
from src.grid import Grid
from src.astar import astar, AStarPlanner, octile_distance


class TestAStar(unittest.TestCase):
    def test_octile_heuristic(self):
        # Orthogonal: dx=5, dy=0 -> 5.0
        self.assertAlmostEqual(octile_distance((0, 0), (5, 0)), 5.0)
        # Diagonal: dx=3, dy=3 -> 3 * sqrt(2)
        self.assertAlmostEqual(octile_distance((0, 0), (3, 3)), 3 * (2**0.5))
        # Mixed: dx=4, dy=2 -> 2*sqrt(2) + 2*1.0
        self.assertAlmostEqual(octile_distance((0, 0), (4, 2)), 2 * (2**0.5) + 2.0)

    def test_open_grid_path(self):
        occ = np.zeros((10, 10), dtype=np.int8)
        grid = Grid(occ)
        path, nodes_expanded = astar(grid, start=(0, 0), goal=(5, 5))
        # Straight diagonal path should be length 6 (0,0 to 5,5)
        self.assertEqual(len(path), 6)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (5, 5))
        self.assertGreater(nodes_expanded, 0)

    def test_obstacle_wall_navigation(self):
        # Wall across row 4 from col 0 to 7
        occ = np.zeros((10, 10), dtype=np.int8)
        occ[4, 0:8] = 1
        grid = Grid(occ)
        path, _ = astar(grid, start=(1, 1), goal=(8, 1))
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], (1, 1))
        self.assertEqual(path[-1], (8, 1))
        # Path must not intersect any obstacle
        for cell in path:
            self.assertFalse(grid.is_obstacle(cell))

    def test_blocked_goal(self):
        occ = np.zeros((5, 5), dtype=np.int8)
        occ[0, 1] = 1
        occ[1, 0] = 1
        occ[1, 1] = 1
        grid = Grid(occ)
        path, _ = astar(grid, start=(0, 0), goal=(4, 4))
        self.assertEqual(path, [])

    def test_start_equals_goal(self):
        grid = Grid(np.zeros((5, 5), dtype=np.int8))
        path, _, cost = AStarPlanner().plan(grid, start=(2, 2), goal=(2, 2))
        self.assertEqual(path, [(2, 2)])
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
