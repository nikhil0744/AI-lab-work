# CS F407 Reinforcement Learning Lab - Warehouse Navigation

This directory contains the complete implementation, unit test suites, evaluation scripts, and comprehensive report for the CS F407 Reinforcement Learning and Deep RL Laboratory Exercise.

---

## Quickstart: How to Run and Check Results

All dependencies (`numpy`, `matplotlib`, `torch`) are configured. You can execute any script directly from this directory:

### 1. Run the Entire End-to-End Pipeline
To run all unit tests, train all three models (Tabular Q, Single DQN, Two-Robot DQN), run unseen start generalization tests, and generate all plots and tables in one command:
```bash
cd lab_rl
python3 evaluate_all.py
```

### 2. Run Individual Modules
* **Run Environment Unit Tests (Task 4 & Task 7)**:
  ```bash
  python3 test_warehouse_env.py
  ```
* **Run Tabular Q-Learning (Task 4 & Task 5)**:
  ```bash
  python3 tabular_q.py
  ```
* **Run Single-Robot DQN (Task 6)**:
  ```bash
  python3 dqn_single.py
  ```
* **Run Two-Robot DQN & View Trajectory Table (Task 7)**:
  ```bash
  python3 dqn_multi.py
  ```
* **Inspect Multi-Seed Stability (Seed 7 vs Seed 42)**:
  ```bash
  python3 compare_multi_seeds.py
  ```

---

## Generated Visualizations & Plots
All plots are automatically generated and saved to [`plots/`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/plots):
* [`plots/tabular_q_training.png`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/plots/tabular_q_training.png): Tabular Q-learning moving average return and epsilon decay.
* [`plots/dqn_single_training.png`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/plots/dqn_single_training.png): Single-robot DQN return and MSE TD error loss curves.
* [`plots/single_robot_trajectories.png`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/plots/single_robot_trajectories.png): Visual 2D grid routes showing optimal 4-step path through the warehouse door.
* [`plots/dqn_multi_training.png`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/plots/dqn_multi_training.png): Two-robot training performance and arrival rates.
* [`plots/dqn_multi_trajectory.png`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/plots/dqn_multi_trajectory.png): Trajectory map showing simultaneous movement of Robot A (blue) and Robot B (red).

---

## Detailed Lab Report
For full answers to all questions in Tasks 1–8, reflection prompts, and the RLHF discussion, see:
* [`REPORT.md`](file:///home/link4444/Downloads/Ai/lab_work_AI/lab_rl/REPORT.md)
