"""
evaluate_all.py
Unified benchmark runner for CS F407 Reinforcement Learning Lab.
Executes:
1. All pre-training unit tests (Tasks 4, 6, 7).
2. Tabular Q-learning agent training and evaluation (Task 5).
3. DQN single-robot training and evaluation (Task 6).
4. DQN two-robot training and evaluation (Task 7).
5. Unseen start generalization testing (Task 8):
   - Single robot from (0, 0) without retraining.
   - Two robots from A at (0, 0) and B at (4, 4) without retraining.
6. Consolidated comparison tables and trajectory visualizations.
"""

from typing import Dict, Any, List, Tuple
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from warehouse_env import WarehouseEnv, WarehouseTwoRobotEnv, OBSTACLES, GRID_ROWS, GRID_COLS, ACTION_NAMES
from test_warehouse_env import test_task4_single_robot_env, test_task7_two_robot_env
from tabular_q import (
    run_synthetic_q_update_tests,
    train_tabular_q,
    evaluate_greedy as eval_tab_greedy,
    evaluate_random_policy,
    plot_tabular_training
)
from dqn_single import (
    run_synthetic_dqn_loss_tests,
    train_dqn_single,
    evaluate_dqn_greedy as eval_dqn1_greedy,
    plot_dqn_training
)
from dqn_multi import (
    train_dqn_multi,
    evaluate_dqn_multi_greedy as eval_dqn2_greedy,
    print_two_robot_trajectory_table,
    plot_two_robot_trajectories,
    plot_multi_training
)


def plot_single_robot_trajectories(
    tab_res: Dict[str, Any],
    dqn_res: Dict[str, Any],
    output_path: str = "plots/single_robot_trajectories.png"
):
    """Plot trajectories of Tabular Q-learning vs DQN on single robot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title, color in [
        (ax1, tab_res, "Tabular Q-Learning (Greedy Route)", "blue"),
        (ax2, dqn_res, "DQN Single Robot (Greedy Route)", "purple")
    ]:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if (r, c) in OBSTACLES:
                    rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='dimgray', edgecolor='black', hatch='//')
                    ax.add_patch(rect)
                elif (r, c) == (2, 2):
                    rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='lightgreen', edgecolor='black', alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(c, 4 - r, 'Door', ha='center', va='center', fontsize=10, color='darkgreen', fontweight='bold')
                elif (r, c) == (2, 0):
                    rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='wheat', edgecolor='gray')
                    ax.add_patch(rect)
                    ax.text(c, 4 - r, 'Start S\n(2,0)', ha='center', va='center', fontsize=9, fontweight='bold')
                elif (r, c) == (2, 4):
                    rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='gold', edgecolor='gray')
                    ax.add_patch(rect)
                    ax.text(c, 4 - r, 'Goal G\n(2,4)', ha='center', va='center', fontsize=9, fontweight='bold')
                else:
                    rect = patches.Rectangle((c - 0.5, 4.5 - r), 1, 1, facecolor='white', edgecolor='lightgray')
                    ax.add_patch(rect)

        traj = res["trajectory"]
        pts = [(c, 4 - r) for (r, c) in traj]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker='o', markersize=9, color=color, linewidth=3, label=f"Steps: {res['steps']}, Return: {res['total_reward']:.2f}")

        for t, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x + 0.15, y + 0.15, f"t={t}", color=color, fontsize=9, fontweight='bold')

        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"Col {c}" for c in range(5)])
        ax.set_yticklabels([f"Row {4-r}" for r in range(5)])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved single robot trajectories comparison plot to {output_path}")


def run_full_pipeline():
    print("=" * 80)
    print("        CS F407 REINFORCEMENT LEARNING LAB: COMPLETE PIPELINE")
    print("=" * 80)

    # 1. Unit Tests
    print("\n>>> STEP 1: Running All Formal Unit Tests (Tasks 4, 6, 7)")
    test_task4_single_robot_env()
    test_task7_two_robot_env()
    run_synthetic_q_update_tests()
    run_synthetic_dqn_loss_tests()
    print(">>> All unit tests verified successfully!\n")

    # 2. Part I: Tabular Q-learning
    print(">>> STEP 2: Training Tabular Q-learning Agent (Task 5)")
    q_table, tab_hist = train_tabular_q(episodes=1500, alpha=0.2, gamma=0.95, seed=7)
    plot_tabular_training(tab_hist, output_path="plots/tabular_q_training.png")

    env_tab = WarehouseEnv(start_pos=(2, 0), goal_pos=(2, 4))
    tab_eval_default = eval_tab_greedy(env_tab, q_table, start_pos=(2, 0))
    tab_eval_unseen = eval_tab_greedy(env_tab, q_table, start_pos=(0, 0))
    rand_eval = evaluate_random_policy(env_tab, n_episodes=100, seed=7)

    # 3. Part II: Single-Robot DQN
    print("\n>>> STEP 3: Training Single-Robot DQN Agent (Task 6)")
    dqn1_model, dqn1_hist = train_dqn_single(total_steps=12000, seed=7)
    plot_dqn_training(dqn1_hist, output_path="plots/dqn_single_training.png")

    dqn1_eval_default = eval_dqn1_greedy(dqn1_model, start_pos=(2, 0))
    dqn1_eval_unseen = eval_dqn1_greedy(dqn1_model, start_pos=(0, 0))
    plot_single_robot_trajectories(tab_eval_default, dqn1_eval_default, output_path="plots/single_robot_trajectories.png")

    # 4. Part III: Two-Robot Joint DQN
    print("\n>>> STEP 4: Training Two-Robot DQN Controller (Task 7)")
    dqn2_model, dqn2_hist = train_dqn_multi(total_steps=60000, seed=7)
    plot_multi_training(dqn2_hist, output_path="plots/dqn_multi_training.png")

    dqn2_eval_default = eval_dqn2_greedy(dqn2_model, start_A=(2, 0), start_B=(2, 4))
    print_two_robot_trajectory_table(dqn2_eval_default)
    plot_two_robot_trajectories(dqn2_eval_default, output_path="plots/dqn_multi_trajectory.png")

    # Unseen starts for two robots: A at (0, 0), B at (4, 4)
    dqn2_eval_unseen = eval_dqn2_greedy(dqn2_model, start_A=(0, 0), start_B=(4, 4))

    # 5. Summarize Results & Print Task 8 Comparison Table
    print("\n" + "=" * 85)
    print("           TASK 8: SUMMARY EVALUATION TABLE (DEFAULT STARTS)")
    print("=" * 85)

    def get_status_str(res: Dict[str, Any], is_two_robot: bool = False) -> str:
        if is_two_robot:
            if res.get("collision", False):
                return "Collision"
            elif res.get("both_arrived", False):
                return "Success"
            elif res.get("timeout", False):
                return "Timeout"
            return "Incomplete"
        else:
            if res.get("goal_reached", False):
                return "Success"
            elif res.get("timeout", False):
                return "Timeout"
            return "Failed"

    s_tab = get_status_str(tab_eval_default, False)
    s_dqn1 = get_status_str(dqn1_eval_default, False)
    s_dqn2 = get_status_str(dqn2_eval_default, True)

    steps_tab = str(tab_eval_default["steps"]) if tab_eval_default["goal_reached"] else "N/A"
    steps_dqn1 = str(dqn1_eval_default["steps"]) if dqn1_eval_default["goal_reached"] else "N/A"
    steps_dqn2 = str(dqn2_eval_default["steps"]) if dqn2_eval_default.get("both_arrived", False) else "N/A"

    print(f"{'Measure':<30} | {'Q-table, one robot':<18} | {'DQN, one robot':<18} | {'DQN, two robots':<18}")
    print("-" * 85)
    print(f"{'Budget and seed':<30} | {'1,500 ep, seed 7':<18} | {'12,000 steps, seed 7':<18} | {'60,000 steps, seed 7':<18}")
    print(f"{'Success / collision / timeout':<30} | {s_tab:<18} | {s_dqn1:<18} | {s_dqn2:<18}")
    print(f"{'Steps, if successful':<30} | {steps_tab:<18} | {steps_dqn1:<18} | {steps_dqn2:<18}")
    print(f"{'Invalid moves':<30} | {tab_eval_default['invalid_moves']:<18} | {dqn1_eval_default['invalid_moves']:<18} | {dqn2_eval_default['invalid_moves']:<18}")
    print(f"{'Deterministic Total Return':<30} | {tab_eval_default['total_reward']:<18.2f} | {dqn1_eval_default['total_reward']:<18.2f} | {dqn2_eval_default['total_reward']:<18.2f}")
    print("=" * 85)

    print("\n" + "=" * 85)
    print("        TASK 8: UNSEEN START GENERALIZATION TEST (NO RETRAINING)")
    print("=" * 85)
    s_tab_u = get_status_str(tab_eval_unseen, False)
    s_dqn1_u = get_status_str(dqn1_eval_unseen, False)
    s_dqn2_u = get_status_str(dqn2_eval_unseen, True)

    print(f"{'Controller':<25} | {'Unseen Start':<15} | {'Outcome':<12} | {'Steps':<7} | {'Invalid Moves':<14} | {'Return':<8}")
    print("-" * 85)
    print(f"{'Tabular Q-learning':<25} | {'(0, 0)':<15} | {s_tab_u:<12} | {tab_eval_unseen['steps']:<7} | {tab_eval_unseen['invalid_moves']:<14} | {tab_eval_unseen['total_reward']:<8.2f}")
    print(f"{'Single-Robot DQN':<25} | {'(0, 0)':<15} | {s_dqn1_u:<12} | {dqn1_eval_unseen['steps']:<7} | {dqn1_eval_unseen['invalid_moves']:<14} | {dqn1_eval_unseen['total_reward']:<8.2f}")
    print(f"{'Two-Robot Joint DQN':<25} | {'A:(0,0), B:(4,4)':<15} | {s_dqn2_u:<12} | {dqn2_eval_unseen['steps']:<7} | {dqn2_eval_unseen['invalid_moves']:<14} | {dqn2_eval_unseen['total_reward']:<8.2f}")
    print("=" * 85 + "\n")

    return {
        "tabular": (tab_eval_default, tab_eval_unseen, rand_eval),
        "dqn1": (dqn1_eval_default, dqn1_eval_unseen),
        "dqn2": (dqn2_eval_default, dqn2_eval_unseen),
    }


if __name__ == "__main__":
    run_full_pipeline()
