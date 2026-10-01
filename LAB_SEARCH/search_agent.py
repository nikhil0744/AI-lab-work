#!/usr/bin/env python3
"""
search_agent.py - AI Search Laboratory: A* and Blind Search Implementation

This module implements:
1. Search problem formulation for warehouse robot navigation:
   P = (S, A, T, s0, G, c)
2. A* Graph Search with multiple heuristic functions:
   - Manhattan distance: h(n) = |x - xG| + |y - yG|
   - Euclidean distance: h(n) = sqrt((x - xG)^2 + (y - yG)^2)
   - Zero heuristic: h(n) = 0 (Uniform-Cost Search / Dijkstra)
   - Weighted Manhattan: 2 * h(n) (Inadmissible / Aggressive)
3. Breadth-First Search (BFS) for comparative evaluation against blind search.
4. Comprehensive test suites corresponding to Tasks 1-6 in search_lab_ex.pdf.
"""

import sys
import math
import heapq
import time
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable

# Coordinate representation: (row, col)
# In 2D Cartesian terms: x = col, y = row
Coordinate = Tuple[int, int]


@dataclass
class SearchResult:
    """Encapsulates the outcome and performance metrics of a search execution."""
    algorithm: str
    heuristic_name: str
    found: bool
    path: List[Coordinate]
    actions: List[str]
    path_length: int
    states_expanded: int
    execution_time_ms: float

    def summary(self) -> str:
        status = "FOUND" if self.found else "NO SOLUTION"
        return (f"[{self.algorithm} | {self.heuristic_name}] Status: {status} | "
                f"Path Length: {self.path_length} | States Expanded: {self.states_expanded} | "
                f"Time: {self.execution_time_ms:.3f} ms")


class WarehouseGrid:
    """
    Represents the warehouse grid environment.
    Handles parsing, bounds checking, obstacle detection, and rendering.
    """

    # Up, Down, Left, Right movement deltas in (row, col) format
    ACTIONS: List[Tuple[str, Coordinate]] = [
        ("Up", (-1, 0)),
        ("Down", (1, 0)),
        ("Left", (0, -1)),
        ("Right", (0, 1)),
    ]

    def __init__(self, ascii_map: str):
        lines = [line.rstrip() for line in ascii_map.strip().split("\n") if line.strip()]
        self.grid = lines
        self.height = len(lines)
        self.width = max(len(row) for row in lines)

        # Pad rows if necessary to ensure uniform rectangular width
        self.grid = [row.ljust(self.width, ' ') for row in lines]

        self.start: Optional[Coordinate] = self._find_char('S')
        self.goal: Optional[Coordinate] = self._find_char('G')

        if not self.start:
            raise ValueError("Start state 'S' not found in grid.")
        if not self.goal:
            raise ValueError("Goal state 'G' not found in grid.")

    def _find_char(self, char: str) -> Optional[Coordinate]:
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == char:
                    return (r, c)
        return None

    def is_valid(self, state: Coordinate) -> bool:
        """Determines if a state is within grid bounds and not an obstacle ('#')."""
        r, c = state
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r][c] != '#'

    def get_successors(self, state: Coordinate) -> List[Tuple[str, Coordinate, int]]:
        """
        Returns valid transitions from state: (action_name, next_state, step_cost).
        Each valid step has uniform cost 1.
        """
        r, c = state
        successors = []
        for action_name, (dr, dc) in self.ACTIONS:
            next_state = (r + dr, c + dc)
            if self.is_valid(next_state):
                successors.append((action_name, next_state, 1))
        return successors

    def render_with_path(self, path: Optional[List[Coordinate]] = None) -> str:
        """
        Renders the grid as ASCII text.
        If a path is provided, intermediate path cells are overlaid with '*'.
        Preserves original 'S' and 'G' markers.
        """
        grid_copy = [list(row) for row in self.grid]
        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    grid_copy[r][c] = '*'
        return "\n".join("".join(row) for row in grid_copy)


# ==============================================================================
# Heuristic Functions
# ==============================================================================

def manhattan_distance(s1: Coordinate, s2: Coordinate) -> float:
    """
    Manhattan distance: h(n) = |r - rG| + |c - cG| = |x - xG| + |y - yG|.
    Admissible and consistent on 4-connected unit-cost grid.
    """
    return float(abs(s1[0] - s2[0]) + abs(s1[1] - s2[1]))


def euclidean_distance(s1: Coordinate, s2: Coordinate) -> float:
    """
    Euclidean distance: h(n) = sqrt((r - rG)^2 + (c - cG)^2).
    Admissible and consistent, but strictly dominated by Manhattan on 4-connected grid.
    (i.e., h_euclidean(n) <= h_manhattan(n) <= h*(n)).
    """
    return float(math.hypot(s1[0] - s2[0], s1[1] - s2[1]))


def zero_heuristic(s1: Coordinate, s2: Coordinate) -> float:
    """
    Zero heuristic: h(n) = 0.
    Reduces A* to Uniform-Cost Search (Dijkstra's Algorithm).
    Admissible, but completely uninformed.
    """
    return 0.0


def weighted_manhattan(weight: float = 2.0) -> Callable[[Coordinate, Coordinate], float]:
    """
    Weighted Manhattan heuristic: h(n) = w * (|r - rG| + |c - cG|).
    For w > 1, this heuristic is inadmissible (can overestimate true distance).
    Behaves aggressively, trading theoretical optimality for search speed.
    """
    def h(s1: Coordinate, s2: Coordinate) -> float:
        return weight * manhattan_distance(s1, s2)
    return h


# ==============================================================================
# Search Algorithms
# ==============================================================================

def a_star_search(
    env: WarehouseGrid,
    heuristic_fn: Callable[[Coordinate, Coordinate], float] = manhattan_distance,
    heuristic_name: str = "Manhattan"
) -> SearchResult:
    """
    Executes A* Graph Search to find the shortest path from env.start to env.goal.
    
    Data structures:
    - Frontier: Priority queue implemented via heapq, storing (f(n), tie_breaker, state).
    - g_score: Dict mapping state -> lowest known path cost g(n) from start.
    - came_from: Dict mapping state -> (predecessor_state, action_taken) for path reconstruction.
    - closed_set: Set of expanded states to prevent repeated redundant expansions.
    """
    start_time = time.perf_counter()

    start = env.start
    goal = env.goal
    if not start or not goal:
        return SearchResult("A*", heuristic_name, False, [], [], 0, 0, 0.0)

    # Unique counter to prevent state tuple comparisons in case of tie in f(n)
    counter = itertools.count()

    # Priority queue stores tuples of: (f_score, tie_breaker, state)
    initial_h = heuristic_fn(start, goal)
    initial_f = 0.0 + initial_h
    frontier: List[Tuple[float, int, Coordinate]] = [(initial_f, next(counter), start)]

    g_score: Dict[Coordinate, float] = {start: 0.0}
    came_from: Dict[Coordinate, Tuple[Coordinate, str]] = {}
    closed_set: Set[Coordinate] = set()
    states_expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)

        # Skip if state has already been expanded (graph search closed list)
        if current in closed_set:
            continue

        closed_set.add(current)
        states_expanded += 1

        # Goal test upon expansion guarantees optimality with admissible heuristic
        if current == goal:
            # Path reconstruction
            path: List[Coordinate] = []
            actions: List[str] = []
            curr: Coordinate = goal
            while curr != start:
                path.append(curr)
                prev, act = came_from[curr]
                actions.append(act)
                curr = prev
            path.append(start)
            path.reverse()
            actions.reverse()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm="A*",
                heuristic_name=heuristic_name,
                found=True,
                path=path,
                actions=actions,
                path_length=len(actions),
                states_expanded=states_expanded,
                execution_time_ms=elapsed_ms
            )

        # Expand node: generate successors
        for action_name, neighbor, step_cost in env.get_successors(current):
            tentative_g = g_score[current] + step_cost

            # If discovered a shorter path to neighbor
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = (current, action_name)
                g_score[neighbor] = tentative_g
                h = heuristic_fn(neighbor, goal)
                f_neighbor = tentative_g + h
                heapq.heappush(frontier, (f_neighbor, next(counter), neighbor))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm="A*",
        heuristic_name=heuristic_name,
        found=False,
        path=[],
        actions=[],
        path_length=0,
        states_expanded=states_expanded,
        execution_time_ms=elapsed_ms
    )


def bfs_search(env: WarehouseGrid) -> SearchResult:
    """
    Executes Breadth-First Search (BFS) to find the shortest unweighted path.
    
    Data structures:
    - Frontier: FIFO Queue implemented via collections.deque.
    - visited: Set of states added to frontier or closed to prevent cycling.
    - came_from: Dict mapping state -> (predecessor_state, action_taken) for path reconstruction.
    """
    start_time = time.perf_counter()

    start = env.start
    goal = env.goal
    if not start or not goal:
        return SearchResult("BFS", "None (Blind)", False, [], [], 0, 0, 0.0)

    frontier = deque([start])
    visited: Set[Coordinate] = {start}
    came_from: Dict[Coordinate, Tuple[Coordinate, str]] = {}
    states_expanded = 0

    while frontier:
        current = frontier.popleft()
        states_expanded += 1

        # Goal test upon expansion (consistent with A* expansion metric)
        if current == goal:
            path: List[Coordinate] = []
            actions: List[str] = []
            curr: Coordinate = goal
            while curr != start:
                path.append(curr)
                prev, act = came_from[curr]
                actions.append(act)
                curr = prev
            path.append(start)
            path.reverse()
            actions.reverse()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(
                algorithm="BFS",
                heuristic_name="None (Blind)",
                found=True,
                path=path,
                actions=actions,
                path_length=len(actions),
                states_expanded=states_expanded,
                execution_time_ms=elapsed_ms
            )

        for action_name, neighbor, _ in env.get_successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = (current, action_name)
                frontier.append(neighbor)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm="BFS",
        heuristic_name="None (Blind)",
        found=False,
        path=[],
        actions=[],
        path_length=0,
        states_expanded=states_expanded,
        execution_time_ms=elapsed_ms
    )


# ==============================================================================
# Laboratory Test Maps (Tasks 3, 5, 6)
# ==============================================================================

ORIGINAL_WAREHOUSE_MAP = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""

TRIVIAL_MAP = """
#####
#SG##
#####
"""

NO_SOLUTION_MAP = """
#######
#S....#
###.###
#...#G#
#######
"""

ALTERNATIVE_PATHS_MAP = """
#################
#S.............G#
#.#############.#
#...............#
#################
"""


# ==============================================================================
# Test Execution Suites
# ==============================================================================

def run_task3_tests() -> Dict[str, SearchResult]:
    """Runs the 4 systematic test cases required by Task 3."""
    print("=" * 72)
    print("TASK 3: SYSTEMATIC TESTING OF A* SEARCH AGENT")
    print("=" * 72)

    results = {}

    # Test 1: Original warehouse
    print("\n--- Test 1: Original Warehouse Map ---")
    env1 = WarehouseGrid(ORIGINAL_WAREHOUSE_MAP)
    res1 = a_star_search(env1, manhattan_distance, "Manhattan")
    results["Test 1"] = res1
    print(res1.summary())
    print("\nPath Coordinates (length = 40):")
    coords_formatted = " -> ".join(f"({r},{c})" for r, c in res1.path)
    # Print wrapped coordinates
    step_chunks = [coords_formatted[i:i+80] for i in range(0, len(coords_formatted), 80)]
    for chunk in step_chunks:
        print("  " + chunk)
    print(f"\nAction Sequence ({len(res1.actions)} actions):")
    print("  " + ", ".join(res1.actions))
    print("\nPath Visualization ('*' indicates path steps):")
    print(env1.render_with_path(res1.path))
    assert res1.found, "Test 1 failed: Path should be found"
    assert res1.path_length == 40, f"Test 1 failed: Expected length 40, got {res1.path_length}"
    assert res1.states_expanded == 64, f"Test 1 failed: Expected 64 states expanded, got {res1.states_expanded}"

    # Test 2: Trivial case (goal adjacent to start)
    print("\n--- Test 2: Trivial Case (Goal Immediately Adjacent) ---")
    env2 = WarehouseGrid(TRIVIAL_MAP)
    res2 = a_star_search(env2, manhattan_distance, "Manhattan")
    results["Test 2"] = res2
    print(res2.summary())
    print(f"Path: {res2.path}")
    print(f"Actions: {res2.actions}")
    print(env2.render_with_path(res2.path))
    assert res2.found, "Test 2 failed: Path should be found"
    assert res2.path_length == 1, f"Test 2 failed: Expected length 1, got {res2.path_length}"
    assert res2.actions == ["Right"], f"Test 2 failed: Expected ['Right'], got {res2.actions}"

    # Test 3: No solution (goal trapped/inaccessible)
    print("\n--- Test 3: No Solution (Goal Inaccessible) ---")
    env3 = WarehouseGrid(NO_SOLUTION_MAP)
    res3 = a_star_search(env3, manhattan_distance, "Manhattan")
    results["Test 3"] = res3
    print(res3.summary())
    print("Path returned: None (search cleanly terminated without infinite loop)")
    print(f"Total reachable states expanded: {res3.states_expanded}")
    assert not res3.found, "Test 3 failed: Path should NOT be found"
    assert res3.path_length == 0, f"Test 3 failed: Expected length 0, got {res3.path_length}"

    # Test 4: Alternative paths (two routes of different lengths)
    print("\n--- Test 4: Alternative Paths ---")
    env4 = WarehouseGrid(ALTERNATIVE_PATHS_MAP)
    res4 = a_star_search(env4, manhattan_distance, "Manhattan")
    results["Test 4"] = res4
    print(res4.summary())
    print(f"Returned Shortest Path Length: {res4.path_length}")
    print(f"Action Sequence: {', '.join(res4.actions)}")
    print(env4.render_with_path(res4.path))
    # Direct top path: (1,1) -> (1,15) = 14 steps
    # Bottom detour path: (1,1) -> (3,1) -> (3,15) -> (1,15) = 2 + 14 + 2 = 18 steps
    assert res4.found, "Test 4 failed: Path should be found"
    assert res4.path_length == 14, f"Test 4 failed: Expected shortest length 14, got {res4.path_length}"

    print("\n>>> All Task 3 tests passed successfully!\n")
    return results


def run_task5_comparison() -> Tuple[SearchResult, SearchResult]:
    """Compares BFS with A* on the original warehouse map (Task 5)."""
    print("=" * 72)
    print("TASK 5: COMPARATIVE EVALUATION - BFS VS A* SEARCH")
    print("=" * 72)

    env = WarehouseGrid(ORIGINAL_WAREHOUSE_MAP)
    bfs_res = bfs_search(env)
    astar_res = a_star_search(env, manhattan_distance, "Manhattan")

    print(f"\nWarehouse Dimensions: {env.width} cols x {env.height} rows (Total walkable cells: 64)")
    print(f"Start: {env.start} | Goal: {env.goal}")
    print("-" * 72)
    print(f"{'Measure':<25} | {'BFS (Blind Search)':<20} | {'A* (Manhattan Heuristic)':<20}")
    print("-" * 72)
    print(f"{'Solution Found':<25} | {str(bfs_res.found):<20} | {str(astar_res.found):<20}")
    print(f"{'Path Length':<25} | {bfs_res.path_length:<20} | {astar_res.path_length:<20}")
    print(f"{'States Expanded':<25} | {bfs_res.states_expanded:<20} | {astar_res.states_expanded:<20}")
    print(f"{'Runtime (ms)':<25} | {bfs_res.execution_time_ms:<20.3f} | {astar_res.execution_time_ms:<20.3f}")
    print("-" * 72)

    return bfs_res, astar_res


def run_task6_investigation():
    """Investigates heuristic variants (Task 6) on warehouse and alternative maps."""
    print("=" * 72)
    print("TASK 6: HEURISTIC INVESTIGATION")
    print("=" * 72)

    heuristics = [
        ("h(n) = 0 (Zero / UCS)", zero_heuristic),
        ("Euclidean Distance", euclidean_distance),
        ("Manhattan Distance", manhattan_distance),
        ("2 * Manhattan (Inadmissible)", weighted_manhattan(2.0)),
    ]

    print("\n--- Part A: Heuristics on Original Warehouse Map ---")
    env_orig = WarehouseGrid(ORIGINAL_WAREHOUSE_MAP)
    print(f"{'Heuristic':<30} | {'Found':<6} | {'Path Len':<9} | {'Expanded':<9} | {'Time (ms)':<10}")
    print("-" * 74)
    for name, h_fn in heuristics:
        res = a_star_search(env_orig, h_fn, name)
        print(f"{name:<30} | {str(res.found):<6} | {res.path_length:<9} | {res.states_expanded:<9} | {res.execution_time_ms:<10.3f}")

    print("\n--- Part B: Heuristics on Alternative Paths Map (Branching/Open) ---")
    env_alt = WarehouseGrid(ALTERNATIVE_PATHS_MAP)
    print(f"{'Heuristic':<30} | {'Found':<6} | {'Path Len':<9} | {'Expanded':<9} | {'Time (ms)':<10}")
    print("-" * 74)
    for name, h_fn in heuristics:
        res = a_star_search(env_alt, h_fn, name)
        print(f"{name:<30} | {str(res.found):<6} | {res.path_length:<9} | {res.states_expanded:<9} | {res.execution_time_ms:<10.3f}")


def main():
    """Main execution orchestrating all laboratory tasks."""
    print("\n" + "#" * 72)
    print("# AI SEARCH LABORATORY: WAREHOUSE ROBOT NAVIGATION EXPERIMENTS")
    print("#" * 72 + "\n")

    run_task3_tests()
    run_task5_comparison()
    print()
    run_task6_investigation()
    print("\nAll laboratory benchmarks, validations, and assertions passed successfully.\n")


if __name__ == "__main__":
    main()
