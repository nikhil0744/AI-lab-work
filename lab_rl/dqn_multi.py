"""
dqn_multi.py
Two-Robot Warehouse Navigation with Collision Avoidance using DQN (Part III).
Features:
- Joint centralized controller: 50 inputs (two concatenated 25-dim one-hot vectors)
- 25 joint actions (5x5): j = 5 * a_A + a_B
- Deep Q-Network: 50 -> 64 ReLU -> 64 ReLU -> 25 Linear
- ReplayBuffer (20,000 capacity)
- Target network synced every 250 optimizer steps
- 60,000 training steps (seed 7)
- Greedy trajectory evaluation, step-by-step reporting table, and trajectory plotting
"""

from typing import Tuple, List, Dict, Any, Optional
import random
import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from warehouse_env import (
    WarehouseTwoRobotEnv,
    get_joint_state_vector,
    pos_to_idx,
    idx_to_pos,
    ACTION_NAMES,
    OBSTACLES,
    GRID_ROWS,
    GRID_COLS
)


class MultiAgentQNetwork(nn.Module):
    """Joint Q-Network: 50 -> 64 -> 64 -> 25."""

    def __init__(self, input_dim: int = 50, hidden_dim: int = 64, output_dim: int = 25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 25 linear Q-values


class MultiAgentReplayBuffer:
    """Experience replay buffer for two-robot joint transitions."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated, truncated))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds, _ = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(terminateds, dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


def compute_multi_dqn_loss(
    online_net: nn.Module,
    target_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.95
) -> torch.Tensor:
    """Compute MSE TD error loss for multi-agent joint controller."""
    with torch.no_grad():
        target_q_next = target_net(next_states)
        max_q_next = target_q_next.max(dim=1).values
        td_targets = rewards + gamma * (1.0 - dones) * max_q_next

    online_q = online_net(states)
    chosen_action_q = online_q.gather(1, actions.unsqueeze(1)).squeeze(1)
    return F.mse_loss(chosen_action_q, td_targets)


def train_dqn_multi(
    total_steps: int = 60000,
    seed: int = 7,
    warmup_steps: int = 500,
    update_freq: int = 4,
    target_sync_freq: int = 250,
    batch_size: int = 64,
    lr: float = 0.001,
    gamma: float = 0.95,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    decay_ratio: float = 0.80
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train two-robot DQN agent for total_steps joint environment steps.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = WarehouseTwoRobotEnv(
        start_A=(2, 0), goal_A=(2, 4),
        start_B=(2, 4), goal_B=(2, 0),
        max_steps=40
    )
    online_net = MultiAgentQNetwork()
    target_net = MultiAgentQNetwork()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    replay_buffer = MultiAgentReplayBuffer(capacity=20000)

    decay_steps = int(total_steps * decay_ratio)
    opt_updates = 0

    history = {
        "episode_returns": [],
        "episode_lengths": [],
        "both_arrived": [],
        "collisions": [],
        "timeouts": [],
        "epsilons": [],
        "losses": [],
    }

    (pos_A, pos_B), _ = env.reset()
    state_vec = get_joint_state_vector(pos_A, pos_B)

    ep_return = 0.0
    ep_len = 0

    for global_step in range(1, total_steps + 1):
        # Epsilon schedule
        if global_step <= decay_steps:
            eps = eps_start - (eps_start - eps_end) * (global_step / decay_steps)
        else:
            eps = eps_end

        # Action selection
        if random.random() < eps:
            joint_action = random.randint(0, 24)
        else:
            with torch.no_grad():
                s_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                q_vals = online_net(s_t).squeeze(0).numpy()
                max_q = np.max(q_vals)
                best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
                joint_action = int(np.random.choice(best_actions))

        # Env step
        (next_A, next_B), reward, terminated, truncated, info = env.step(joint_action)
        next_state_vec = get_joint_state_vector(next_A, next_B)

        replay_buffer.push(state_vec, joint_action, reward, next_state_vec, terminated, truncated)

        ep_return += reward
        ep_len += 1

        # Train step
        if global_step >= warmup_steps and (global_step % update_freq == 0):
            if len(replay_buffer) >= batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                loss = compute_multi_dqn_loss(online_net, target_net, b_s, b_a, b_r, b_ns, b_d, gamma=gamma)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                opt_updates += 1
                history["losses"].append(loss.item())

                if opt_updates % target_sync_freq == 0:
                    target_net.load_state_dict(online_net.state_dict())

        # Check episode end
        if terminated or truncated:
            history["episode_returns"].append(ep_return)
            history["episode_lengths"].append(ep_len)
            history["both_arrived"].append(1 if info.get("both_arrived", False) else 0)
            history["collisions"].append(1 if info.get("collision", False) else 0)
            history["timeouts"].append(1 if (truncated and not terminated) else 0)
            history["epsilons"].append(eps)

            (pos_A, pos_B), _ = env.reset()
            state_vec = get_joint_state_vector(pos_A, pos_B)
            ep_return = 0.0
            ep_len = 0
        else:
            state_vec = next_state_vec

    return online_net, history


def evaluate_dqn_multi_greedy(
    model: nn.Module,
    start_A: Tuple[int, int] = (2, 0),
    start_B: Tuple[int, int] = (2, 4),
    max_steps: int = 40
) -> Dict[str, Any]:
    """
    Greedy evaluation of two-robot policy without exploration or weight updates.
    """
    env = WarehouseTwoRobotEnv(
        start_A=start_A, goal_A=(2, 4),
        start_B=start_B, goal_B=(2, 0),
        max_steps=max_steps
    )
    (pos_A, pos_B), _ = env.reset(start_A=start_A, start_B=start_B)

    trajectory_A = [pos_A]
    trajectory_B = [pos_B]
    joint_actions = []
    rewards = []
    status_history = []
    total_invalid = 0

    model.eval()
    done = False

    while not done:
        state_vec = get_joint_state_vector(pos_A, pos_B)
        with torch.no_grad():
            s_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            q_vals = model(s_t).squeeze(0).numpy()
            joint_action = int(np.argmax(q_vals))

        a_A, a_B = divmod(joint_action, 5)
        joint_actions.append((a_A, a_B))

        (next_A, next_B), r, term, trunc, info = env.step(joint_action)
        trajectory_A.append(next_A)
        trajectory_B.append(next_B)
        rewards.append(r)
        total_invalid += info.get("invalid_moves", 0)

        step_status = "OK"
        if info.get("collision", False):
            step_status = "COLLISION (" + ("Same-Cell" if info.get("same_cell") else "Swap") + ")"
        elif info.get("both_arrived", False):
            step_status = "BOTH ARRIVED (Success)"
        elif info.get("newly_arrived", 0) > 0:
            step_status = f"{info['newly_arrived']} Robot Arrived"
        elif trunc:
            step_status = "TIMEOUT"

        status_history.append(step_status)

        pos_A, pos_B = next_A, next_B
        done = term or trunc

    return {
        "trajectory_A": trajectory_A,
        "trajectory_B": trajectory_B,
        "joint_actions": joint_actions,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "both_arrived": info.get("both_arrived", False),
        "collision": info.get("collision", False),
        "timeout": (trunc and not term),
        "steps": env.step_count,
        "invalid_moves": total_invalid,
        "status_history": status_history,
    }


def print_two_robot_trajectory_table(eval_result: Dict[str, Any]):
    """Print step-by-step table of positions, actions, and rewards."""
    traj_A = eval_result["trajectory_A"]
    traj_B = eval_result["trajectory_B"]
    actions = eval_result["joint_actions"]
    rewards = eval_result["rewards"]
    statuses = eval_result["status_history"]

    print("\n" + "="*80)
    print(f"{'Step':<5} | {'Robot A (Pos)':<13} | {'Action A':<8} | {'Robot B (Pos)':<13} | {'Action B':<8} | {'Reward':<7} | {'Status':<20}")
    print("-" * 80)

    for t in range(len(actions)):
        pA = traj_A[t]
        pB = traj_B[t]
        aA_str = ACTION_NAMES[actions[t][0]]
        aB_str = ACTION_NAMES[actions[t][1]]
        r = rewards[t]
        status = statuses[t]
        print(f"{t+1:<5} | {str(pA):<13} | {aA_str:<8} | {str(pB):<13} | {aB_str:<8} | {r:<7.2f} | {status:<20}")

    print("-" * 80)
    print(f"Final Positions: Robot A at {traj_A[-1]}, Robot B at {traj_B[-1]}")
    print(f"Summary: Success={eval_result['both_arrived']}, Collision={eval_result['collision']}, Timeout={eval_result['timeout']}, Steps={eval_result['steps']}, Total Reward={eval_result['total_reward']:.2f}")
    print("="*80 + "\n")


def plot_two_robot_trajectories(eval_result: Dict[str, Any], output_path: str = "plots/dqn_multi_trajectory.png"):
    """
    Plot 2D grid with trajectories of Robot A and Robot B with timestep labels.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traj_A = eval_result["trajectory_A"]
    traj_B = eval_result["trajectory_B"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in OBSTACLES:
                rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='dimgray', edgecolor='black', hatch='//')
                ax.add_patch(rect)
            elif (r, c) == (2, 2):
                rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='lightgreen', edgecolor='black', alpha=0.4)
                ax.add_patch(rect)
                ax.text(c, 4 - r, 'Door', ha='center', va='center', fontsize=9, color='darkgreen', fontweight='bold')
            else:
                rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='white', edgecolor='lightgray')
                ax.add_patch(rect)

    # Plot Robot A trajectory (blue)
    pts_A = [(c, 4 - r) for (r, c) in traj_A]
    xA = [p[0] - 0.12 for p in pts_A]  # offset slightly to avoid overlap
    yA = [p[1] + 0.12 for p in pts_A]
    ax.plot(xA, yA, marker='o', markersize=8, color='blue', linewidth=2.5, label='Robot A (Start (2,0) -> Goal (2,4))')

    for t, (x, y) in enumerate(zip(xA, yA)):
        ax.text(x, y + 0.18, f"A:{t}", color='blue', fontsize=8, ha='center', fontweight='bold')

    # Plot Robot B trajectory (red)
    pts_B = [(c, 4 - r) for (r, c) in traj_B]
    xB = [p[0] + 0.12 for p in pts_B]  # offset slightly to avoid overlap
    yB = [p[1] - 0.12 for p in pts_B]
    ax.plot(xB, yB, marker='s', markersize=8, color='crimson', linewidth=2.5, linestyle='--', label='Robot B (Start (2,4) -> Goal (2,0))')

    for t, (x, y) in enumerate(zip(xB, yB)):
        ax.text(x, y - 0.22, f"B:{t}", color='crimson', fontsize=8, ha='center', fontweight='bold')

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"Col {c}" for c in range(5)])
    ax.set_yticklabels([f"Row {4-r}" for r in range(5)])
    ax.set_title("Two-Robot Collision-Avoidance Trajectories in 5x5 Warehouse", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved two-robot trajectory plot to {output_path}")


def plot_multi_training(history: Dict[str, Any], output_path: str = "plots/dqn_multi_training.png", window: int = 50):
    """Plot multi-agent training curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    returns = np.array(history["episode_returns"])
    arrivals = np.array(history["both_arrived"])
    collisions = np.array(history["collisions"])

    def moving_avg(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if len(returns) >= window:
        ma_ret = moving_avg(returns, window)
        x_axis = np.arange(window, len(returns) + 1)
        ax1.plot(returns, alpha=0.15, color='royalblue', label='Episode Return (Raw)')
        ax1.plot(x_axis, ma_ret, color='blue', linewidth=2, label=f'Return ({window}-ep MA)')
    else:
        ax1.plot(returns, color='blue', label='Episode Return')

    ax1.set_ylabel('Return')
    ax1.set_title('Two-Robot DQN: Training Performance over 60,000 Environment Steps')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    if len(arrivals) >= window:
        ma_arr = moving_avg(arrivals, window) * 100
        ma_col = moving_avg(collisions, window) * 100
        ax2.plot(x_axis, ma_arr, color='green', linewidth=2, label='Success Rate (% Both Arrived)')
        ax2.plot(x_axis, ma_col, color='red', linewidth=1.5, linestyle=':', label='Collision Rate (%)')
    ax2.plot(history["epsilons"], color='orange', linestyle='--', label='Epsilon')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rate (%) / Epsilon')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved multi-agent training plot to {output_path}")


if __name__ == "__main__":
    print("=== Training Two-Robot DQN (60,000 Environment Steps, Seed 7) ===")
    model, history = train_dqn_multi(total_steps=60000, seed=7)
    plot_multi_training(history, output_path="plots/dqn_multi_training.png")

    eval_result = evaluate_dqn_multi_greedy(model, start_A=(2, 0), start_B=(2, 4))
    print_two_robot_trajectory_table(eval_result)
    plot_two_robot_trajectories(eval_result, output_path="plots/dqn_multi_trajectory.png")
