"""
compare_multi_seeds.py
Investigate Two-Robot DQN across different seeds and step budgets.
As noted in the lab handout:
"DQN training may stall or vary across seeds. Keep failed runs and inspect them.
If time permits, try a fresh run with another seed or a larger budget."
"""

import sys
import numpy as np
import torch
from dqn_multi import train_dqn_multi, evaluate_dqn_multi_greedy, print_two_robot_trajectory_table


def test_seed(seed: int, steps: int = 60000):
    print(f"\n==========================================")
    print(f"Testing Two-Robot DQN with Seed {seed}, Steps: {steps}")
    print(f"==========================================")
    model, history = train_dqn_multi(total_steps=steps, seed=seed)
    eval_res = evaluate_dqn_multi_greedy(model)
    print(f"Seed {seed} Result: Both Arrived={eval_res['both_arrived']}, Collision={eval_res['collision']}, Timeout={eval_res['timeout']}, Steps={eval_res['steps']}, Reward={eval_res['total_reward']:.2f}")
    if eval_res['both_arrived']:
        print(">>> SUCCESSFUL MUTUAL DELIVERY ACHIEVED! <<<")
        print_two_robot_trajectory_table(eval_res)
    return eval_res, model, history


if __name__ == "__main__":
    seeds_to_test = [7, 42, 1, 100]
    results = {}
    for s in seeds_to_test:
        res, model, hist = test_seed(s, steps=60000)
        results[s] = res
