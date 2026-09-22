import unittest
import numpy as np
from src.grid import Grid


class TestGrid(unittest.TestCase):
    def test_grid_initialization(self):
        occ = np.zeros((10, 10), dtype=np.int8)
        occ[3, 4] = 1
        grid = Grid(occ, start=(0, 0), goal=(9, 9))

        self.assertEqual(grid.rows, 10)
        self.assertEqual(grid.cols, 10)
        self.assertEqual(grid.shape, (10, 10))
        self.assertEqual(grid.start, (0, 0))
        self.assertEqual(grid.goal, (9, 9))
        self.assertTrue(grid.is_obstacle((3, 4)))
        self.assertTrue(grid.is_free((0, 0)))
        self.assertFalse(grid.is_free((3, 4)))
        self.assertFalse(grid.in_bounds((-1, 0)))
        self.assertFalse(grid.in_bounds((10, 10)))

    def test_dynamic_set_obstacle(self):
        grid = Grid(np.zeros((5, 5), dtype=np.int8))
        self.assertTrue(grid.is_free((2, 2)))
        grid.set_obstacle((2, 2), True)
        self.assertTrue(grid.is_obstacle((2, 2)))
        grid.set_obstacle((2, 2), False)
        self.assertTrue(grid.is_free((2, 2)))

    def test_neighbors_8_connected(self):
        grid = Grid(np.zeros((3, 3), dtype=np.int8))
        # Center cell in open 3x3 has 8 neighbors
        neighbors = grid.get_neighbors((1, 1))
        self.assertEqual(len(neighbors), 8)

        # Corner cell has 3 neighbors
        corner_neighbors = grid.get_neighbors((0, 0))
        self.assertEqual(len(corner_neighbors), 3)

        # Check step costs
        costs = {n: round(c, 3) for n, c in corner_neighbors}
        self.assertEqual(costs[(0, 1)], 1.0)
        self.assertEqual(costs[(1, 0)], 1.0)
        self.assertEqual(costs[(1, 1)], round(Grid.SQRT2, 3))


if __name__ == "__main__":
    unittest.main()
