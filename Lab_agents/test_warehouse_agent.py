"""
Unit and Integration Tests for Warehouse Navigation Goal-Based Agent
"""

import unittest
from Lab_agents.warehouse_agent import WarehouseEnvironment, GoalBasedAgent, WAREHOUSE_MAP


class TestWarehouseAgent(unittest.TestCase):
    """Test suite verifying agent planning, obstacle avoidance, and edge case handling."""

    def setUp(self):
        self.env = WarehouseEnvironment(WAREHOUSE_MAP)
        self.agent = GoalBasedAgent(self.env)

    def test_start_and_goal_identification(self):
        """Verify correct identification of start and goal coordinates."""
        self.assertEqual(self.env.start, (1, 1))
        self.assertEqual(self.env.goal, (1, 19))
        self.assertEqual(self.agent.current_state, (1, 1))
        self.assertEqual(self.agent.goal, (1, 19))

    def test_baseline_path_validity(self):
        """Verify that the baseline warehouse path is found and optimal."""
        path = self.agent.search_path()
        self.assertIsNotNone(path, "A path should be found for the baseline warehouse map.")
        
        # Verify length: exactly 20 steps (21 states)
        self.assertEqual(len(path) - 1, 20)
        self.assertEqual(path[0], self.env.start)
        self.assertEqual(path[-1], self.env.goal)

        # Verify continuity: each step must be exactly 1 cardinal unit move
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
            self.assertEqual(manhattan_dist, 1, f"Discontinuous move between {path[i]} and {path[i+1]}")

        # Verify obstacle avoidance: no coordinate in path can be '#'
        for r, c in path:
            self.assertNotEqual(self.env.grid[r][c], '#', f"Collision detected at ({r}, {c})")

    def test_action_sequence_generation(self):
        """Verify that coordinates translate accurately to cardinal actions."""
        path = self.agent.search_path()
        actions = self.agent.path_to_actions(path)
        self.assertEqual(len(actions), 20)

        # Re-simulate path from actions
        r, c = self.env.start
        action_map = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
        simulated_path = [(r, c)]
        for act in actions:
            dr, dc = action_map[act]
            r, c = r + dr, c + dc
            simulated_path.append((r, c))

        self.assertEqual(simulated_path, path)
        self.assertEqual(simulated_path[-1], self.env.goal)

    def test_unreachable_goal(self):
        """Verify agent handles maps where the goal is completely sealed off."""
        walled_map = [
            "#######",
            "#S#..G#",
            "#######"
        ]
        env = WarehouseEnvironment(walled_map)
        agent = GoalBasedAgent(env)
        path = agent.search_path()
        self.assertIsNone(path, "Search must return None when no collision-free path exists.")

    def test_start_equals_goal(self):
        """Verify agent handles initial state being identical to the goal state."""
        env = WarehouseEnvironment(WAREHOUSE_MAP)
        agent = GoalBasedAgent(env)
        agent.current_state = env.goal
        path = agent.search_path()
        self.assertEqual(path, [env.goal])
        self.assertEqual(agent.path_to_actions(path), [])

    def test_missing_start_or_goal_exception(self):
        """Verify environment instantiation enforces presence of S and G."""
        no_start = ["###", "#.G", "###"]
        no_goal = ["###", "#S.", "###"]
        with self.assertRaises(ValueError):
            WarehouseEnvironment(no_start)
        with self.assertRaises(ValueError):
            WarehouseEnvironment(no_goal)


if __name__ == "__main__":
    unittest.main()
