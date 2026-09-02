"""
Warehouse Navigation Problem - Goal-Based Intelligent Agent

Algorithm Chosen: Breadth-First Search (BFS)
Why BFS is appropriate:
1. Optimality: In an unweighted grid where each step has uniform cost (1 step = 1 move),
   BFS is guaranteed to find the shortest path (minimum number of steps) from start (S) to goal (G).
2. Completeness: BFS will always find a solution if one exists in a finite grid.
3. Simplicity & Predictability: BFS systematically explores states layer by layer, avoiding loops
   by maintaining a visited set.
"""

from collections import deque
from typing import List, Tuple, Optional, Dict

# The Warehouse Map as specified in the lab problem
WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]


class WarehouseEnvironment:
    """Represents the 2D warehouse environment."""

    def __init__(self, grid_map: List[str]):
        self.grid = [list(row) for row in grid_map]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.start = self._find_position('S')
        self.goal = self._find_position('G')

        if not self.start:
            raise ValueError("Start position 'S' not found in the grid.")
        if not self.goal:
            raise ValueError("Goal position 'G' not found in the grid.")

    def _find_position(self, symbol: str) -> Optional[Tuple[int, int]]:
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == symbol:
                    return (r, c)
        return None

    def is_valid_move(self, r: int, c: int) -> bool:
        """Checks if a cell (r, c) is within bounds and not an obstacle."""
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r][c] != '#'
        return False

    def render(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """Renders the grid map, optionally superimposing the solution path with '*'."""
        display_grid = [row.copy() for row in self.grid]
        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    display_grid[r][c] = '*'
        return "\n".join("".join(row) for row in display_grid)


class GoalBasedAgent:
    """Goal-based autonomous agent that plans and navigates to the target."""

    # 4 Cardinal actions: (row_delta, col_delta, action_name)
    ACTIONS = [
        (-1, 0, "Up"),
        (1, 0, "Down"),
        (0, -1, "Left"),
        (0, 1, "Right")
    ]

    def __init__(self, env: WarehouseEnvironment):
        self.env = env
        self.current_state = env.start
        self.goal = env.goal

    def search_path(self) -> Optional[List[Tuple[int, int]]]:
        """
        Uses Breadth-First Search (BFS) to find the shortest collision-free path.
        Returns a list of coordinates [(r0, c0), (r1, c1), ...] if a path exists,
        or None otherwise.
        """
        start = self.current_state
        goal = self.goal

        # Queue contains tuples of: (current_coordinate, path_taken_so_far)
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()

            # Goal test
            if current_pos == goal:
                return path

            r, c = current_pos

            # Explore all possible valid actions
            for dr, dc, _ in self.ACTIONS:
                next_pos = (r + dr, c + dc)
                if next_pos not in visited and self.env.is_valid_move(next_pos[0], next_pos[1]):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return None

    def execute_plan(self):
        """Plans and prints the navigation result."""
        print("=" * 60)
        print("WAREHOUSE NAVIGATION GOAL-BASED AGENT")
        print("=" * 60)
        print(f"Start Position (S) : {self.current_state}")
        print(f"Goal Position (G)  : {self.goal}")
        print("\nInitial Warehouse Map:")
        print(self.env.render())
        print("-" * 60)

        print("\nSearching for collision-free path using Breadth-First Search (BFS)...")
        path = self.search_path()

        if path:
            print(f"\n[SUCCESS] Path found in {len(path) - 1} steps!")
            
            # Translate coordinates to action sequence
            action_sequence = []
            for i in range(len(path) - 1):
                cur_r, cur_c = path[i]
                next_r, next_c = path[i + 1]
                for dr, dc, name in self.ACTIONS:
                    if (cur_r + dr, cur_c + dc) == (next_r, next_c):
                        action_sequence.append(name)
                        break

            print(f"\nAction Sequence ({len(action_sequence)} actions):")
            print(" -> ".join(action_sequence))

            print("\nCoordinates Path:")
            for step, pos in enumerate(path):
                print(f"  Step {step:02d}: {pos}")

            print("\nWarehouse Map with Navigation Path ('*'):")
            print(self.env.render(path))
        else:
            print("\n[FAILURE] No collision-free path exists from Start (S) to Goal (G).")
        print("=" * 60)


if __name__ == "__main__":
    env = WarehouseEnvironment(WAREHOUSE_MAP)
    agent = GoalBasedAgent(env)
    agent.execute_plan()
