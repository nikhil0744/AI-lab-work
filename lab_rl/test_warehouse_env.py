"""
test_warehouse_env.py
Unit tests for single and two-robot warehouse environments as specified in Tasks 4 & 7.
"""

from warehouse_env import WarehouseEnv, WarehouseTwoRobotEnv, ACTIONS, pos_to_idx, idx_to_pos


def test_task4_single_robot_env():
    print("=== Running Task 4: Single-Robot Environment Unit Tests ===")
    env = WarehouseEnv()

    test_cases = [
        # (initial_pos, action_code, expected_next_pos, expected_reward, expected_term, description)
        ((0, 0), 0, (0, 0), -1.0, False, "(0, 0) up -> boundary hit"),
        ((1, 1), 3, (1, 1), -1.0, False, "(1, 1) right -> obstacle hit at (1, 2)"),
        ((2, 1), 3, (2, 2), -0.1, False, "(2, 1) right -> move into door (2, 2)"),
        ((2, 1), 4, (2, 1), -0.1, False, "(2, 1) wait -> stay at (2, 1)"),
        ((2, 3), 3, (2, 4), 10.0, True, "(2, 3) right -> reach goal (2, 4) and terminate"),
    ]

    for start_pos, action, exp_pos, exp_r, exp_term, desc in test_cases:
        env.reset(start_pos=start_pos)
        next_pos, r, term, trunc, info = env.step(action)
        assert next_pos == exp_pos, f"Failed {desc}: got pos {next_pos}, expected {exp_pos}"
        assert abs(r - exp_r) < 1e-6, f"Failed {desc}: got reward {r}, expected {exp_r}"
        assert term == exp_term, f"Failed {desc}: got term {term}, expected {exp_term}"
        assert trunc is False, f"Failed {desc}: truncation was unexpectedly True"
        print(f"  [PASS] {desc}")

    # Test 40-step cutoff truncation
    print("  Testing 40-step cutoff truncation...")
    env.reset(start_pos=(0, 0))
    for step in range(39):
        _, _, term, trunc, _ = env.step(4)  # Wait
        assert not term and not trunc, f"Premature stop at step {step+1}"
    # 40th step
    _, _, term, trunc, _ = env.step(4)
    assert not term, "Goal should not be reached by waiting at (0, 0)"
    assert trunc, "40th step must trigger truncation"
    print("  [PASS] 40-step truncation behaves correctly.")


def test_task7_two_robot_env():
    print("\n=== Running Task 7: Two-Robot Environment Unit Tests ===")
    env = WarehouseTwoRobotEnv()

    # Case 1: Same-cell collision
    # Old A=(2,1), B=(2,3); A=right, B=left -> proposed (2,2) for both
    env.reset(start_A=(2, 1), start_B=(2, 3))
    joint_action = 5 * 3 + 2  # A=right(3), B=left(2)
    next_state, r, term, trunc, info = env.step(joint_action)
    assert next_state == ((2, 1), (2, 3)), f"Expected retained old positions, got {next_state}"
    assert r == -10.0, f"Expected reward -10.0, got {r}"
    assert term is True and info["collision"] and info["same_cell"], "Expected same-cell collision termination"
    print("  [PASS] Same-cell collision at (2, 2) correctly detected and terminated with -10 reward.")

    # Case 2: Swap collision
    # Old A=(2,1), B=(2,2); A=right, B=left -> A to (2,2), B to (2,1)
    env.reset(start_A=(2, 1), start_B=(2, 2))
    joint_action = 5 * 3 + 2  # A=right(3), B=left(2)
    next_state, r, term, trunc, info = env.step(joint_action)
    assert next_state == ((2, 1), (2, 2)), f"Expected retained old positions, got {next_state}"
    assert r == -10.0, f"Expected reward -10.0, got {r}"
    assert term is True and info["collision"] and info["swap"], "Expected swap collision termination"
    print("  [PASS] Swap collision correctly detected and terminated with -10 reward.")

    # Case 3: Following into vacated cell
    # Old A=(2,1), B=(2,2); A=right, B=right -> A to (2,2), B to (2,3)
    env.reset(start_A=(2, 1), start_B=(2, 2))
    joint_action = 5 * 3 + 3  # A=right(3), B=right(3)
    next_state, r, term, trunc, info = env.step(joint_action)
    assert next_state == ((2, 2), (2, 3)), f"Expected ((2,2), (2,3)), got {next_state}"
    assert abs(r - (-0.1)) < 1e-6, f"Expected reward -0.1, got {r}"
    assert not term and not info["collision"], "Following vacated cell should be allowed"
    print("  [PASS] Following vacated cell is allowed without collision.")

    # Case 4: Move and wait
    # Old A=(2,1), B=(2,2); A=up, B=wait -> A to (1,1), B at (2,2)
    env.reset(start_A=(2, 1), start_B=(2, 2))
    joint_action = 5 * 0 + 4  # A=up(0), B=wait(4)
    next_state, r, term, trunc, info = env.step(joint_action)
    assert next_state == ((1, 1), (2, 2)), f"Expected ((1,1), (2,2)), got {next_state}"
    assert abs(r - (-0.1)) < 1e-6, f"Expected reward -0.1, got {r}"
    assert not term and not info["collision"], "Valid move and wait should be allowed"
    print("  [PASS] Valid simultaneous move and wait allowed.")

    # Case 5: Robot at goal stays and cannot receive repeated rewards
    # Set B at (1, 0) so B is NOT at its goal (2, 0)
    env.reset(start_A=(2, 3), start_B=(1, 0))
    # Step A into goal (2,4)
    next_state, r, term, trunc, info = env.step(5 * 3 + 4)  # A=right, B=wait
    assert next_state == ((2, 4), (1, 0))
    # r = -0.1 + 5*1 (since k=1, I=0) = +4.9 (A reached goal, B hasn't)
    assert abs(r - 4.9) < 1e-6, f"Expected 4.9 on first arrival, got {r}"
    assert not term, "Should not terminate until both arrive"

    # Now A tries to move left, but is already at goal
    next_state, r, term, trunc, info = env.step(5 * 2 + 4)  # A=left, B=wait
    assert next_state[0] == (2, 4), "Robot A at goal must stay at goal"
    assert abs(r - (-0.1)) < 1e-6, f"Expected -0.1, got {r} (no repeated goal reward, no invalid move penalty)"
    print("  [PASS] Robot at goal remains frozen, receiving no repeated arrival rewards.")


if __name__ == "__main__":
    test_task4_single_robot_env()
    test_task7_two_robot_env()
    print("\nAll environment tests passed successfully!")
