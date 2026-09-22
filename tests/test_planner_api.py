import unittest
import numpy as np
import src.planner as planner
from src.dynamic_env import DynamicEnvironment


class TestPlannerAPI(unittest.TestCase):
    def test_plan_contract(self):
        occ = np.zeros((10, 10), dtype=np.int8)
        start = (0, 0)
        goal = (8, 8)

        # Test A* mode
        path_astar = planner.plan(occ, start, goal, mode="astar")
        self.assertIsInstance(path_astar, list)
        self.assertGreater(len(path_astar), 0)
        self.assertIsInstance(path_astar[0], tuple)
        self.assertEqual(path_astar[0], start)
        self.assertEqual(path_astar[-1], goal)

        # Test ARIS mode
        path_aris = planner.plan(occ, start, goal, mode="aris")
        self.assertIsInstance(path_aris, list)
        self.assertGreater(len(path_aris), 0)
        self.assertIsInstance(path_aris[0], tuple)
        self.assertEqual(path_aris[0], start)
        self.assertEqual(path_aris[-1], goal)

    def test_replan_obstacle_avoidance(self):
        occ = np.zeros((10, 10), dtype=np.int8)
        start = (0, 0)
        goal = (5, 5)

        path = planner.plan(occ, start, goal, mode="aris")
        self.assertGreater(len(path), 2)

        # Pick the second step on the path and block it
        blocked_cell = path[2]
        current_pos = path[1]

        updated_path = planner.replan([(blocked_cell, True)], current_pos=current_pos)
        self.assertGreater(len(updated_path), 0)
        self.assertEqual(updated_path[0], current_pos)
        self.assertEqual(updated_path[-1], goal)
        # Blocked cell must not be present in the new path
        self.assertNotIn(blocked_cell, updated_path)

    def test_dynamic_environment_mid_run(self):
        occ = np.zeros((12, 12), dtype=np.int8)
        start = (0, 0)
        goal = (10, 10)

        env = DynamicEnvironment(occ, start, goal, mode="aris")
        self.assertGreater(len(env.path), 0)

        # Move 3 steps
        pos = env.step(3)
        self.assertNotEqual(pos, start)

        # Inject obstacle 2 steps ahead
        blocked_cell, new_path = env.inject_obstacle_ahead(steps_ahead=2)
        self.assertIsNotNone(blocked_cell)
        self.assertNotIn(blocked_cell, new_path)

        # Step until goal
        max_steps = 30
        steps = 0
        while not env.is_at_goal() and steps < max_steps:
            env.step(1)
            steps += 1

        self.assertTrue(env.is_at_goal())
        self.assertEqual(env.collision_count, 0)


if __name__ == "__main__":
    unittest.main()
