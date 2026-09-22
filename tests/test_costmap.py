import unittest
import numpy as np
from src.costmap import CostMap
from src.grid import Grid


class TestCostMap(unittest.TestCase):
    def test_single_obstacle_clearance_and_risk(self):
        occ = np.zeros((7, 7), dtype=np.int8)
        occ[3, 3] = 1  # Single obstacle at center

        costmap = CostMap(occ, r_safe=3.0, w_r=5.0, p=2.0)

        # Center obstacle
        self.assertEqual(costmap.clearance((3, 3)), 0.0)
        self.assertEqual(costmap.risk((3, 3)), 1.0)
        self.assertEqual(costmap.cost((3, 3)), float("inf"))

        # Orthogonal neighbor (3, 4): clearance 1.0
        self.assertAlmostEqual(costmap.clearance((3, 4)), 1.0)
        # risk = (3.0 - 1.0) / 3.0 = 2/3
        expected_risk = (3.0 - 1.0) / 3.0
        self.assertAlmostEqual(costmap.risk((3, 4)), expected_risk)
        # cost = 1.0 * (1 + 5 * (2/3)^2) = 1 + 5 * (4/9) = 1 + 20/9 = 29/9
        expected_cost = 1.0 * (1.0 + 5.0 * (expected_risk**2))
        self.assertAlmostEqual(costmap.cost((3, 4), base_step_cost=1.0), expected_cost)

        # Diagonal neighbor (4, 4): clearance sqrt(2)
        self.assertAlmostEqual(costmap.clearance((4, 4)), 2**0.5)

        # Far away cell (0, 0): dist = hypot(3, 3) = sqrt(18) = 4.24 >= 3.0
        self.assertGreaterEqual(costmap.clearance((0, 0)), 3.0)
        self.assertEqual(costmap.risk((0, 0)), 0.0)
        self.assertEqual(costmap.cost((0, 0), base_step_cost=1.0), 1.0)

    def test_open_grid(self):
        occ = np.zeros((5, 5), dtype=np.int8)
        costmap = CostMap(occ, r_safe=3.0)
        self.assertEqual(costmap.risk((2, 2)), 0.0)
        self.assertEqual(costmap.cost((2, 2)), 1.0)


if __name__ == "__main__":
    unittest.main()
