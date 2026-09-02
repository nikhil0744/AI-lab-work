"""
Logical Reasoning for Planning
Laboratory Implementation: Simple STRIPS-style BFS Planning Agent
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional, FrozenSet


@dataclass
class Action:
    name: str
    positive_preconditions: Set[str] = field(default_factory=set)
    negative_preconditions: Set[str] = field(default_factory=set)
    positive_effects: Set[str] = field(default_factory=set)
    negative_effects: Set[str] = field(default_factory=set)

    def is_applicable(self, state: Set[str]) -> bool:
        """
        Check if the action is logically applicable in the given state.
        S |= Preconditions(a):
          1. All positive preconditions must be present in the state.
          2. No negative preconditions must be present in the state.
        """
        pos_satisfied = self.positive_preconditions.issubset(state)
        neg_satisfied = self.negative_preconditions.isdisjoint(state)
        return pos_satisfied and neg_satisfied

    def apply(self, state: Set[str]) -> Set[str]:
        """
        Apply effects to state:
        1. Remove negative effects
        2. Add positive effects
        """
        new_state = (state - self.negative_effects) | self.positive_effects
        return new_state

    def __repr__(self):
        return self.name


class BFSPlanner:
    def __init__(self, actions: List[Action]):
        self.actions = actions

    def plan(
        self, initial_state: Set[str], goal_state: Set[str]
    ) -> Optional[Tuple[List[Action], List[Set[str]]]]:
        """
        Find a valid sequence of actions from initial_state to achieve goal_state
        using Breadth-First Search (BFS).
        Returns (action_sequence, state_sequence) or None if no plan exists.
        """
        # A state satisfies the goal if all goal propositions hold in the state
        if goal_state.issubset(initial_state):
            return [], [initial_state]

        # Queue contains: (current_state, action_history, state_history)
        queue = deque([(initial_state, [], [initial_state])])
        visited: Set[FrozenSet[str]] = {frozenset(initial_state)}

        while queue:
            current_state, plan_history, state_history = queue.popleft()

            for action in self.actions:
                if action.is_applicable(current_state):
                    next_state = action.apply(current_state)
                    next_state_frozen = frozenset(next_state)

                    if next_state_frozen not in visited:
                        new_plan = plan_history + [action]
                        new_states = state_history + [next_state]

                        # Check goal satisfaction
                        if goal_state.issubset(next_state):
                            return new_plan, new_states

                        visited.add(next_state_frozen)
                        queue.append((next_state, new_plan, new_states))

        return None


def print_plan_results(
    test_name: str,
    initial_state: Set[str],
    goal: Set[str],
    result: Optional[Tuple[List[Action], List[Set[str]]]],
):
    print("=" * 60)
    print(f"TEST: {test_name}")
    print(f"Initial State : {sorted(list(initial_state))}")
    print(f"Goal          : {sorted(list(goal))}")
    print("-" * 60)

    if result is None:
        print("Result: No plan found")
    else:
        plan, states = result
        print(f"Result: Plan found ({len(plan)} actions)\n")
        print(f"S0 (Initial) : {sorted(list(states[0]))}")
        for i, (action, st) in enumerate(zip(plan, states[1:]), start=1):
            print(f"  Step {i}: Execute -> {action.name}")
            print(f"  S{i} State   : {sorted(list(st))}")
    print("=" * 60 + "\n")


def build_warehouse_actions(allow_pickup: bool = True) -> List[Action]:
    """
    Constructs the warehouse domain actions:
    Locations: A, B, C
    Connected paths: A <-> B, B <-> C
    """
    actions = []
    locations = ["A", "B", "C"]
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]

    # 1. Move actions
    for src, dst in connections:
        actions.append(
            Action(
                name=f"Move({src}, {dst})",
                positive_preconditions={f"At(Robot, {src})"},
                negative_preconditions=set(),
                positive_effects={f"At(Robot, {dst})"},
                negative_effects={f"At(Robot, {src})"},
            )
        )

    # 2. PickUp actions
    if allow_pickup:
        for loc in locations:
            actions.append(
                Action(
                    name=f"PickUp(Package, {loc})",
                    positive_preconditions={
                        f"At(Robot, {loc})",
                        f"At(Package, {loc})",
                    },
                    negative_preconditions={"Holding(Package)"},
                    positive_effects={"Holding(Package)"},
                    negative_effects={f"At(Package, {loc})"},
                )
            )

    # 3. Drop actions
    for loc in locations:
        actions.append(
            Action(
                name=f"Drop(Package, {loc})",
                positive_preconditions={f"At(Robot, {loc})", "Holding(Package)"},
                negative_preconditions=set(),
                positive_effects={f"At(Package, {loc})"},
                negative_effects={"Holding(Package)"},
            )
        )

    return actions


def run_all_tests():
    # Initial state and Goal for standard scenario
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}

    # --- Test A: Solvable Problem ---
    actions_a = build_warehouse_actions(allow_pickup=True)
    planner_a = BFSPlanner(actions_a)
    result_a = planner_a.plan(initial_state, goal)
    print_plan_results("Test A (Solvable Warehouse Problem)", initial_state, goal, result_a)

    # --- Test B: Impossible Problem (No PickUp action) ---
    actions_b = build_warehouse_actions(allow_pickup=False)
    planner_b = BFSPlanner(actions_b)
    result_b = planner_b.plan(initial_state, goal)
    print_plan_results("Test B (Impossible Problem - PickUp Removed)", initial_state, goal, result_b)

    # --- Test C: Irrelevant Actions ---
    # With irrelevant actions, verify planner doesn't reach goal by merely moving robot to C without package
    actions_c = build_warehouse_actions(allow_pickup=True)
    planner_c = BFSPlanner(actions_c)
    result_c = planner_c.plan(initial_state, goal)
    print_plan_results("Test C (Irrelevant Actions / Goal Discrimination)", initial_state, goal, result_c)


if __name__ == "__main__":
    run_all_tests()
