"""
tabular_q.py
Tabular Q-learning Agent for CS F407 Reinforcement Learning Lab (Part I).
Includes:
- Synthetic Q-update unit tests (Task 4)
- Epsilon-greedy action selection
- Q-learning update step (distinguishing termination vs truncation)
- Training loop (1500 episodes, linear epsilon decay)
- Greedy evaluation and random baseline evaluation
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import os

from warehouse_env import WarehouseEnv, pos_to_idx, idx_to_pos, ACTION_NAMES


def select_action_epsilon_greedy(
    Q: np.ndarray,
    state_idx: int,
    epsilon: float,
    rng: np.random.Generator,
    training: bool = True
) -> int:
    """
    Select action using epsilon-greedy policy.
    During training: break ties randomly among max Q actions.
    During evaluation (training=False): use fixed deterministic tie-breaking (e.g. np.argmax).
    """
    if training and rng.uniform() < epsilon:
        return int(rng.integers(0, Q.shape[1]))

    q_values = Q[state_idx]
    if training:
        max_val = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_val))
        return int(rng.choice(best_actions))
    else:
        # Fixed deterministic tie-breaking for evaluation
        return int(np.argmax(q_values))


def q_learning_update(
    Q: np.ndarray,
    state_idx: int,
    action: int,
    reward: float,
    next_state_idx: int,
    terminated: bool,
    truncated: bool,
    alpha: float,
    gamma: float
) -> float:
    """
    Perform a single Q-learning update:
    target y = r, if terminated (true termination)
               r + gamma * max_a' Q(s', a'), otherwise (including truncation!)
    Q(s, a) <- Q(s, a) + alpha * (y - Q(s, a))

    Returns the updated Q(s, a) value.
    """
    if terminated:
        td_target = reward
    else:
        # Bootstrap at truncation or standard ongoing transition
        td_target = reward + gamma * np.max(Q[next_state_idx])

    td_error = td_target - Q[state_idx, action]
    Q[state_idx, action] += alpha * td_error
    return Q[state_idx, action]


def run_synthetic_q_update_tests():
    """
    Task 4 synthetic update tests:
    1. Q(s, a) = 0.4, r = 1, max Q(s', a') = 2, alpha = 0.2, gamma = 0.95
       target = 1 + 0.95*2 = 2.9, Q_new = 0.4 + 0.2*(2.9 - 0.4) = 0.90
    2. Truly terminal with r = 5:
       target = 5, Q_new = 0.4 + 0.2*(5 - 0.4) = 1.32
    3. Truncated with r = 1, max Q' = 2:
       target = 2.9, Q_new = 0.90 (bootstrapping retained)
    """
    print("=== Running Task 4: Synthetic Q-learning Update Tests ===")
    alpha = 0.2
    gamma = 0.95

    # Test 1: Non-terminal transition
    Q = np.zeros((25, 5), dtype=np.float64)
    s, a, s_next = 0, 1, 5
    Q[s, a] = 0.4
    Q[s_next, :] = [1.0, 2.0, 0.5, 1.5, 0.0]  # max is 2.0

    updated_val = q_learning_update(Q, s, a, reward=1.0, next_state_idx=s_next,
                                    terminated=False, truncated=False, alpha=alpha, gamma=gamma)
    assert abs(updated_val - 0.90) < 1e-6, f"Expected 0.90, got {updated_val}"
    print(f"  [PASS] Non-terminal update: Q(s,a) = {updated_val:.4f} (expected 0.90)")

    # Test 2: Truly terminal transition (r = 5)
    Q = np.zeros((25, 5), dtype=np.float64)
    Q[s, a] = 0.4
    Q[s_next, :] = [1.0, 2.0, 0.5, 1.5, 0.0]
    updated_val_term = q_learning_update(Q, s, a, reward=5.0, next_state_idx=s_next,
                                         terminated=True, truncated=False, alpha=alpha, gamma=gamma)
    assert abs(updated_val_term - 1.32) < 1e-6, f"Expected 1.32, got {updated_val_term}"
    print(f"  [PASS] Terminal update (r=5): Q(s,a) = {updated_val_term:.4f} (expected 1.32)")

    # Test 3: Truncated transition (r = 1, max Q' = 2, truncated=True)
    Q = np.zeros((25, 5), dtype=np.float64)
    Q[s, a] = 0.4
    Q[s_next, :] = [1.0, 2.0, 0.5, 1.5, 0.0]
    updated_val_trunc = q_learning_update(Q, s, a, reward=1.0, next_state_idx=s_next,
                                          terminated=False, truncated=True, alpha=alpha, gamma=gamma)
    assert abs(updated_val_trunc - 0.90) < 1e-6, f"Expected 0.90, got {updated_val_trunc}"
    print(f"  [PASS] Truncated update (bootstraps next state): Q(s,a) = {updated_val_trunc:.4f} (expected 0.90)")


def train_tabular_q(
    episodes: int = 1500,
    alpha: float = 0.2,
    gamma: float = 0.95,
    seed: int = 7,
    decay_episodes: int = 1200,
    eps_start: float = 1.0,
    eps_end: float = 0.05
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Train Tabular Q-learning agent for 1500 episodes with linear epsilon decay.
    """
    rng = np.random.default_rng(seed)
    env = WarehouseEnv(start_pos=(2, 0), goal_pos=(2, 4), max_steps=40)
    Q = np.zeros((25, 5), dtype=np.float64)

    history = {
        "episode_returns": [],
        "goal_reached": [],
        "episode_lengths": [],
        "epsilons": [],
        "invalid_moves": [],
        "timeouts": [],
    }

    for ep in range(episodes):
        # Linear epsilon decay over decay_episodes
        if ep < decay_episodes:
            eps = eps_start - (eps_start - eps_end) * (ep / decay_episodes)
        else:
            eps = eps_end

        pos, _ = env.reset()
        state_idx = pos_to_idx(pos[0], pos[1])

        ep_return = 0.0
        ep_invalid = 0
        done = False

        while not done:
            action = select_action_epsilon_greedy(Q, state_idx, eps, rng, training=True)
            next_pos, reward, terminated, truncated, info = env.step(action)
            next_state_idx = pos_to_idx(next_pos[0], next_pos[1])

            # Apply Q-learning update BEFORE resetting
            q_learning_update(Q, state_idx, action, reward, next_state_idx, terminated, truncated, alpha, gamma)

            ep_return += reward
            if info["invalid_move"]:
                ep_invalid += 1

            state_idx = next_state_idx
            done = terminated or truncated

        history["episode_returns"].append(ep_return)
        history["goal_reached"].append(1 if info["goal_reached"] else 0)
        history["episode_lengths"].append(env.step_count)
        history["epsilons"].append(eps)
        history["invalid_moves"].append(ep_invalid)
        history["timeouts"].append(1 if (truncated and not terminated) else 0)

    return Q, history


def evaluate_greedy(
    env: WarehouseEnv,
    Q: np.ndarray,
    start_pos: Tuple[int, int] = (2, 0),
    max_steps: int = 40
) -> Dict[str, Any]:
    """
    Evaluate deterministic greedy policy (eps=0, no learning updates).
    """
    env.max_steps = max_steps
    pos, _ = env.reset(start_pos=start_pos)
    trajectory = [pos]
    actions_taken = []
    total_reward = 0.0
    invalid_count = 0
    done = False

    while not done:
        state_idx = pos_to_idx(pos[0], pos[1])
        action = select_action_epsilon_greedy(Q, state_idx, epsilon=0.0, rng=np.random.default_rng(), training=False)
        actions_taken.append(action)

        pos, r, terminated, truncated, info = env.step(action)
        trajectory.append(pos)
        total_reward += r
        if info["invalid_move"]:
            invalid_count += 1
        done = terminated or truncated

    return {
        "trajectory": trajectory,
        "actions": actions_taken,
        "total_reward": total_reward,
        "goal_reached": terminated,
        "timeout": (truncated and not terminated),
        "steps": env.step_count,
        "invalid_moves": invalid_count,
    }


def evaluate_random_policy(
    env: WarehouseEnv,
    n_episodes: int = 100,
    seed: int = 7
) -> Dict[str, Any]:
    """
    Evaluate uniform random policy for n_episodes.
    """
    rng = np.random.default_rng(seed)
    successes = 0
    timeouts = 0
    successful_steps = []
    total_invalid = 0

    for _ in range(n_episodes):
        env.reset(start_pos=(2, 0))
        done = False
        while not done:
            action = int(rng.integers(0, 5))
            _, _, term, trunc, info = env.step(action)
            if info["invalid_move"]:
                total_invalid += 1
            if term:
                successes += 1
                successful_steps.append(env.step_count)
                done = True
            elif trunc:
                timeouts += 1
                done = True

    return {
        "success_fraction": successes / n_episodes,
        "timeout_fraction": timeouts / n_episodes,
        "mean_steps_successful": np.mean(successful_steps) if successful_steps else None,
        "mean_invalid_moves_per_ep": total_invalid / n_episodes,
    }


def plot_tabular_training(history: Dict[str, Any], output_path: str = "plots/tabular_q_training.png", window: int = 50):
    """Plot moving average training return and success rate."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    returns = np.array(history["episode_returns"])
    success = np.array(history["goal_reached"])

    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    ma_returns = moving_avg(returns, window)
    ma_success = moving_avg(success, window)
    x_axis = np.arange(window, len(returns) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(returns, alpha=0.2, color='dodgerblue', label='Episode Return (Raw)')
    ax1.plot(x_axis, ma_returns, color='blue', linewidth=2, label=f'Moving Avg ({window} ep)')
    ax1.set_ylabel('Return')
    ax1.set_title('Tabular Q-learning: Training Performance over 1,500 Episodes')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(x_axis, ma_success * 100, color='forestgreen', linewidth=2, label=f'Success Rate ({window} ep avg)')
    ax2.plot(history["epsilons"], color='orange', linestyle='--', label='Epsilon (exploration)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success % / Epsilon')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training plot to {output_path}")


if __name__ == "__main__":
    run_synthetic_q_update_tests()

    print("\n=== Training Tabular Q-learning Agent (1,500 Episodes, Seed 7) ===")
    Q, hist = train_tabular_q(episodes=1500, alpha=0.2, gamma=0.95, seed=7)
    plot_tabular_training(hist, output_path="plots/tabular_q_training.png")

    env = WarehouseEnv(start_pos=(2, 0), goal_pos=(2, 4))
    eval_result = evaluate_greedy(env, Q)
    print("\n--- Greedy Policy Evaluation (Single Robot) ---")
    print(f"Goal Reached: {eval_result['goal_reached']}")
    print(f"Path Length: {eval_result['steps']} steps")
    print(f"Total Return: {eval_result['total_reward']:.2f}")
    print(f"Invalid Moves: {eval_result['invalid_moves']}")
    print(f"Trajectory: {eval_result['trajectory']}")
    print(f"Actions Taken: {[ACTION_NAMES[a] for a in eval_result['actions']]}")

    random_eval = evaluate_random_policy(env, n_episodes=100, seed=7)
    print("\n--- Uniform Random Policy Baseline (100 Episodes) ---")
    print(f"Success Fraction: {random_eval['success_fraction']:.2f}")
    print(f"Timeout Fraction: {random_eval['timeout_fraction']:.2f}")
    print(f"Mean Steps (Successful): {random_eval['mean_steps_successful']}")
    print(f"Mean Invalid Moves / Ep: {random_eval['mean_invalid_moves_per_ep']:.2f}")
