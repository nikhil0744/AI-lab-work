"""
warehouse_env.py
Warehouse Grid Environment for CS F407 Reinforcement Learning Lab.
Implements:
1. WarehouseEnv: Single-robot warehouse navigation MDP.
2. WarehouseTwoRobotEnv: Two-robot warehouse navigation with collision detection.
"""

from typing import Tuple, Dict, Any, Optional, Set
import numpy as np

# Grid dimensions
GRID_ROWS = 5
GRID_COLS = 5

# Obstacle coordinates in the 5x5 grid
OBSTACLES: Set[Tuple[int, int]] = {(0, 2), (1, 2), (3, 2), (4, 2)}

# Doorway coordinate
DOORWAY: Tuple[int, int] = (2, 2)

# Action space: 0: Up, 1: Down, 2: Left, 3: Right, 4: Wait
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
    4: (0, 0),   # Wait
}
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Wait"]


def pos_to_idx(r: int, c: int) -> int:
    """Map 2D grid position (r, c) to 1D index 5*r + c."""
    return 5 * r + c


def idx_to_pos(idx: int) -> Tuple[int, int]:
    """Map 1D index to 2D grid position (r, c)."""
    return divmod(idx, 5)


def pos_to_one_hot(pos: Tuple[int, int]) -> np.ndarray:
    """Convert (r, c) position into a 25-dimensional one-hot vector."""
    vec = np.zeros(25, dtype=np.float32)
    vec[pos_to_idx(pos[0], pos[1])] = 1.0
    return vec


class WarehouseEnv:
    """
    Single-robot warehouse navigation environment.
    Grid: 5x5 with obstacles at column 2 (except door at (2, 2)).
    Start: (2, 0)
    Goal: (2, 4)
    Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Wait
    Rewards:
      - Goal reached: +10 (true termination)
      - Invalid move (obstacle / boundary): -1
      - Any other valid action (including wait): -0.1
    Episode cutoff: 40 steps (truncation).
    """

    def __init__(self, start_pos: Tuple[int, int] = (2, 0), goal_pos: Tuple[int, int] = (2, 4), max_steps: int = 40):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.max_steps = max_steps
        self.obstacles = set(OBSTACLES)

        self.robot_pos = self.start_pos
        self.step_count = 0

    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        """Reset environment to initial or custom start position."""
        if start_pos is not None:
            self.robot_pos = start_pos
        else:
            self.robot_pos = self.start_pos
        self.step_count = 0
        return self.robot_pos, {"step_count": 0}

    def set_state(self, pos: Tuple[int, int]) -> None:
        """Manually set robot position (useful for unit tests)."""
        self.robot_pos = pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.
        Returns:
            next_state (Tuple[int, int]): robot's new (r, c)
            reward (float): scalar reward
            terminated (bool): true termination (reached goal)
            truncated (bool): episode step cutoff (40 steps reached)
            info (dict): auxiliary debugging info
        """
        assert action in ACTIONS, f"Invalid action {action}"
        self.step_count += 1

        dr, dc = ACTIONS[action]
        candidate_r = self.robot_pos[0] + dr
        candidate_c = self.robot_pos[1] + dc

        # Check bounds and obstacles
        is_out_of_bounds = not (0 <= candidate_r < GRID_ROWS and 0 <= candidate_c < GRID_COLS)
        is_obstacle = (candidate_r, candidate_c) in self.obstacles

        is_invalid_move = is_out_of_bounds or is_obstacle

        if is_invalid_move:
            # Robot stays in place
            next_pos = self.robot_pos
            reward = -1.0
            terminated = False
        else:
            next_pos = (candidate_r, candidate_c)
            if next_pos == self.goal_pos:
                reward = 10.0
                terminated = True
            else:
                reward = -0.1
                terminated = False

        self.robot_pos = next_pos

        # Truncation occurs if cutoff reached without true termination
        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        info = {
            "invalid_move": is_invalid_move,
            "goal_reached": terminated,
            "step_count": self.step_count,
        }

        return next_pos, reward, terminated, truncated, info


class WarehouseTwoRobotEnv:
    """
    Two-robot warehouse navigation environment with simultaneous movement.
    Robot A: Start (2, 0), Goal (2, 4)
    Robot B: Start (2, 4), Goal (2, 0)
    Joint action: j in [0, 24] -> (a_A, a_B) = divmod(j, 5).
    Rewards:
      - Attempted collision: -10 (terminates immediately)
      - Otherwise: -0.1 - b + 5*k + 5*I
        b: number of invalid moves by robots not already at goal
        k: number newly reaching goal (first arrival rewarded once)
        I: 1 if both robots now at goal, 0 otherwise
    Cutoff: 40 joint steps (truncation).
    """

    def __init__(
        self,
        start_A: Tuple[int, int] = (2, 0),
        goal_A: Tuple[int, int] = (2, 4),
        start_B: Tuple[int, int] = (2, 4),
        goal_B: Tuple[int, int] = (2, 0),
        max_steps: int = 40,
    ):
        self.start_A = start_A
        self.goal_A = goal_A
        self.start_B = start_B
        self.goal_B = goal_B
        self.max_steps = max_steps
        self.obstacles = set(OBSTACLES)

        self.pos_A = self.start_A
        self.pos_B = self.start_B
        self.arrived_A = False
        self.arrived_B = False
        self.step_count = 0

    def reset(
        self,
        start_A: Optional[Tuple[int, int]] = None,
        start_B: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, Any]]:
        """Reset environment to initial or custom positions."""
        self.pos_A = start_A if start_A is not None else self.start_A
        self.pos_B = start_B if start_B is not None else self.start_B
        self.arrived_A = (self.pos_A == self.goal_A)
        self.arrived_B = (self.pos_B == self.goal_B)
        self.step_count = 0
        return (self.pos_A, self.pos_B), {"step_count": 0}

    def set_state(self, pos_A: Tuple[int, int], pos_B: Tuple[int, int], arrived_A: bool = False, arrived_B: bool = False):
        """Manually set positions and arrival status (for unit tests)."""
        self.pos_A = pos_A
        self.pos_B = pos_B
        self.arrived_A = arrived_A or (pos_A == self.goal_A)
        self.arrived_B = arrived_B or (pos_B == self.goal_B)

    def step(self, joint_action: int) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float, bool, bool, Dict[str, Any]]:
        """
        Execute one joint action j = 5 * a_A + a_B.
        Returns:
            next_state: ((rA, cA), (rB, cB))
            reward: float team reward
            terminated: bool (collision or both reached goals)
            truncated: bool (40 steps reached)
            info: dict
        """
        self.step_count += 1
        a_A, a_B = divmod(joint_action, 5)

        old_A = self.pos_A
        old_B = self.pos_B

        # 1. Compute proposed move for Robot A
        b_A = 0
        if self.arrived_A:
            proposed_A = old_A
        else:
            dr_A, dc_A = ACTIONS[a_A]
            cand_A = (old_A[0] + dr_A, old_A[1] + dc_A)
            if (not (0 <= cand_A[0] < GRID_ROWS and 0 <= cand_A[1] < GRID_COLS)) or (cand_A in self.obstacles):
                proposed_A = old_A
                b_A = 1
            else:
                proposed_A = cand_A

        # 2. Compute proposed move for Robot B
        b_B = 0
        if self.arrived_B:
            proposed_B = old_B
        else:
            dr_B, dc_B = ACTIONS[a_B]
            cand_B = (old_B[0] + dr_B, old_B[1] + dc_B)
            if (not (0 <= cand_B[0] < GRID_ROWS and 0 <= cand_B[1] < GRID_COLS)) or (cand_B in self.obstacles):
                proposed_B = old_B
                b_B = 1
            else:
                proposed_B = cand_B

        # 3. Check for collisions
        # Same-cell collision: proposed positions are equal
        same_cell = (proposed_A == proposed_B)
        # Swap collision: robots exchange old positions in one step
        swap = (proposed_A == old_B and proposed_B == old_A)

        collision = same_cell or swap

        if collision:
            # On collision, retain the old positions for reporting and terminate
            reward = -10.0
            terminated = True
            truncated = False
            info = {
                "collision": True,
                "same_cell": same_cell,
                "swap": swap,
                "invalid_moves": b_A + b_B,
                "both_arrived": False,
                "step_count": self.step_count,
            }
            return (old_A, old_B), reward, terminated, truncated, info

        # 4. Movement successful (no collision)
        self.pos_A = proposed_A
        self.pos_B = proposed_B

        # Check goal arrivals
        k = 0
        if not self.arrived_A and self.pos_A == self.goal_A:
            self.arrived_A = True
            k += 1

        if not self.arrived_B and self.pos_B == self.goal_B:
            self.arrived_B = True
            k += 1

        I_both = 1 if (self.arrived_A and self.arrived_B) else 0
        b_total = b_A + b_B

        reward = -0.1 - b_total + 5.0 * k + 5.0 * I_both

        both_arrived = (self.arrived_A and self.arrived_B)
        terminated = both_arrived

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        info = {
            "collision": False,
            "same_cell": False,
            "swap": False,
            "invalid_moves": b_total,
            "newly_arrived": k,
            "both_arrived": both_arrived,
            "step_count": self.step_count,
        }

        return (self.pos_A, self.pos_B), reward, terminated, truncated, info


def get_joint_state_vector(pos_A: Tuple[int, int], pos_B: Tuple[int, int]) -> np.ndarray:
    """Concatenate two 25-dim one-hot vectors into a 50-dim vector."""
    vec = np.zeros(50, dtype=np.float32)
    vec[pos_to_idx(pos_A[0], pos_A[1])] = 1.0
    vec[25 + pos_to_idx(pos_B[0], pos_B[1])] = 1.0
    return vec
