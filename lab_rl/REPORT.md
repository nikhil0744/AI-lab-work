# CS F407 Reinforcement Learning & Deep RL: Laboratory Report
**Using an LLM as an Engineering Assistant**  
*Academic Year 2026 | Environment: Deterministic Warehouse Gridworld*

---

## 1. Executive Summary

This laboratory exercise implements, tests, and evaluates three reinforcement learning controllers for navigation in an obstacle-laden warehouse:
1. **Tabular Q-Learning** for a single mobile robot.
2. **Deep Q-Networks (DQN)** for a single mobile robot using PyTorch.
3. **Multi-Agent Centralized DQN** for two robots executing simultaneous navigation through a narrow bottleneck doorway with strict collision avoidance.

In addition to core RL algorithms, the laboratory emphasizes the engineering discipline of using Large Language Models (LLMs) as code generators: specifying formal MDP components, conducting isolated unit tests on transition dynamics and TD targets, auditing loss gradients, and independently verifying policy execution.

---

## 2. Environment Specification: The Warehouse

### 2.1 Grid Topology & Coordinates
* **Dimensions**: $5 \times 5$ discrete grid, coordinates $(r, c) \in \{0, 1, 2, 3, 4\} \times \{0, 1, 2, 3, 4\}$.
* **Rows ($r$)**: Increase downwards from 0 (top) to 4 (bottom).
* **Columns ($c$)**: Increase rightwards from 0 (left) to 4 (right).
* **Obstacles (Walls)**: Shaded cells along column 2 except the doorway at row 2:
  $$\mathcal{O} = \{(0, 2), (1, 2), (3, 2), (4, 2)\}$$
* **Narrow Passage**: Only cell $(2, 2)$ connects the left chamber (columns 0–1) and the right chamber (columns 3–4).

### 2.2 Action Space & Movement Rules
Five discrete actions:
* `0: Up` $(-1, 0)$
* `1: Down` $(+1, 0)$
* `2: Left` $(0, -1)$
* `3: Right` $(0, +1)$
* `4: Wait` $(0, 0)$

**Boundary & Obstacle Physics**: Deterministic movement. Attempting to step into an obstacle or outside the $5 \times 5$ grid bounds leaves the robot in its current position. Waiting is a valid action that leaves the robot in place.

### 2.3 Reward Function & Stopping Rules
* **Single Robot**:
  * Goal reached: $+10$ (true termination).
  * Obstacle or boundary collision attempt: $-1$.
  * Any other valid action (including waiting): $-0.1$.
* **Discount Factor**: $\gamma = 0.95$.
* **Truncation vs. Termination**:
  * **Termination**: The task is solved when the robot enters the goal cell. TD continuation term is eliminated: target $y = r$.
  * **Truncation**: The rollout is capped at $40$ steps to prevent infinite loops. The target *must bootstrap* the next-state value: target $y = r + \gamma \max_{a'} Q(s', a')$. The next state before reset is used in the update.

---

## 3. Part I: Tabular Q-Learning (Single Robot)

### 3.1 Task 1: MDP Formulation
| MDP Component | Formal Specification |
| :--- | :--- |
| **State Space ($\mathcal{S}$)** | Discrete cell index $s = 5r + c \in \{0, 1, \dots, 24\}$. Obstacle states are unvisited. |
| **Action Space ($\mathcal{A}$)** | $\{0, 1, 2, 3, 4\}$ corresponding to $\{\text{Up, Down, Left, Right, Wait}\}$. |
| **Transition Function ($p$)** | Deterministic: $s' = f(s, a)$. Stays in place on boundary/obstacle; updates linearly on valid moves. |
| **Reward Function ($R$)** | $R(s, a, s') = +10$ if $s' = G$; $-1$ if invalid move; $-0.1$ otherwise. |
| **Initial State ($s_0$)** | Start position $S = (2, 0)$, corresponding to state index $10$. |
| **Terminal Outcome** | Goal position $G = (2, 4)$, state index $14$. True termination. |

#### Analytical Questions (Task 1):
* **(a) What information is necessary to specify the state?**  
  Only the agent's current coordinates $(r, c)$ on the grid.
* **(b) Why is the current position sufficient when the map and goal are fixed?**  
  Because the environment is fully observable, static, and Markovian ($P(s_{t+1} \mid s_t, a_t, s_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)$). The history of past cells visited provides zero additional information regarding future transitions or rewards.
* **(c) Is the transition rule deterministic?**  
  Yes. Given state $s$ and action $a$, the next state $s'$ is uniquely determined with probability 1.
* **(d) What is the shortest valid route from $S$ to $G$?**  
  4 steps: $(2, 0) \to (2, 1) \to (2, 2) \to (2, 3) \to (2, 4)$ (all `Right` actions).
* **(e) Is finding that route by search the same process as learning action values from experience?**  
  No. Search (e.g., $A^*$, BFS) requires explicit access to an environmental transition model and goal test to expand a search tree without execution penalties. RL learns action values $Q(s, a)$ solely from trial-and-error trajectory samples and received scalar rewards without prior knowledge of transitions or reward distributions.

---

### 3.2 Task 4: Unit Testing Environment Dynamics & TD Updates

#### Environment Unit Tests
| Initial Position | Action | Expected Outcome | Verified Program Result | Status |
| :---: | :---: | :--- | :--- | :---: |
| $(0, 0)$ | Up | Stay at $(0, 0)$; reward $-1$; no termination | Stay at $(0, 0)$; reward $-1.0$; term=False | **PASS** |
| $(1, 1)$ | Right | Stay at $(1, 1)$; reward $-1$ (hits obstacle $(1, 2)$) | Stay at $(1, 1)$; reward $-1.0$; term=False | **PASS** |
| $(2, 1)$ | Right | Move to doorway $(2, 2)$; reward $-0.1$ | Move to $(2, 2)$; reward $-0.1$; term=False | **PASS** |
| $(2, 1)$ | Wait | Stay at $(2, 1)$; reward $-0.1$ | Stay at $(2, 1)$; reward $-0.1$; term=False | **PASS** |
| $(2, 3)$ | Right | Reach goal $(2, 4)$; reward $+10$; terminate | Reach $(2, 4)$; reward $+10.0$; term=True | **PASS** |
| $(0, 0)$ | Wait $\times 40$ | Cutoff at 40 steps; truncate | Stop at step 40; trunc=True; term=False | **PASS** |

#### Synthetic TD Update Tests
Given test parameters: $Q(s, a) = 0.4$, $\alpha = 0.2$, $\gamma = 0.95$.
1. **Non-terminal transition** ($r = 1, \max_{a'} Q(s', a') = 2$):
   $$y = 1 + 0.95(2) = 2.90 \implies Q(s, a) \leftarrow 0.4 + 0.2(2.90 - 0.4) = \mathbf{0.90}$$
   *Result*: Program computed `0.9000` (**PASS**).
2. **Terminal transition** ($r = 5$, goal reached):
   $$y = 5.00 \implies Q(s, a) \leftarrow 0.4 + 0.2(5.00 - 0.4) = \mathbf{1.32}$$
   *Result*: Program computed `1.3200` (**PASS**).
3. **Truncated transition** ($r = 1, \max_{a'} Q(s', a') = 2$, truncated flag set):
   $$y = 1 + 0.95(2) = 2.90 \implies Q(s, a) \leftarrow \mathbf{0.90}$$
   *Result*: Program computed `0.9000` (**PASS**).
* **Question**: *Which case should ignore the next-state value?*  
  **Answer**: The truly terminal case (Case 2). Reaching the goal terminates future expected rewards, whereas truncation is an artificial data collection cutoff that requires bootstrapping.

---

### 3.3 Task 5: Tabular Training, Evaluation & Exploration Investigation

#### Training Performance
* **Configuration**: 1,500 episodes, $\alpha = 0.2$, $\gamma = 0.95$, seed 7. Linear $\varepsilon$ decay from $1.0 \to 0.05$ across first 1,200 episodes.
* **Moving Average Plot**: Stored at `plots/tabular_q_training.png`. Rapidly converges to the optimal $+9.70$ return within 250 episodes.

#### Evaluation Comparison
| Measure | Uniform Random Policy (100 runs) | Deterministic Greedy Q-Policy |
| :--- | :---: | :---: |
| **Goal Reached Fraction** | $0.18$ ($18\%$) | $\mathbf{1.00}$ ($100\%$) |
| **Steps to Goal (Mean)** | $26.94$ | $\mathbf{4.00}$ (Optimal) |
| **Timeouts ($40$ steps)** | $0.82$ ($82\%$) | $\mathbf{0.00}$ ($0\%$) |
| **Invalid Moves / Episode** | $9.86$ | $\mathbf{0.00}$ |
| **Deterministic Total Return** | $-12.45$ (avg) | $\mathbf{+9.70}$ |

* **Greedy Trajectory**: `(2, 0) -> (2, 1) -> (2, 2) -> (2, 3) -> (2, 4)`.  
  Actions: `[Right, Right, Right, Right]`. Exactly 4 steps.

#### Exploration Analysis Questions:
* **(a) Why might the training curve remain noisy after a good route is learned?**  
  Because $\varepsilon$-greedy exploration continues to force random actions with probability $\varepsilon > 0$ during training. Even after optimal Q-values are discovered, exploratory random moves cause the robot to bump into boundaries or obstacles, generating $-1.0$ penalties and lowering individual episode returns.
* **(b) Does a high training return necessarily imply a good greedy policy?**  
  Not necessarily. A noisy training return with occasional lucky high-reward rollouts can mask unvisited or poorly estimated states. Conversely, an agent with a moderately low training return due to heavy exploration ($\varepsilon$) may have learned a completely optimal greedy policy when evaluated deterministically ($\varepsilon = 0$).
* **(c) What is the difference between learning rate $\alpha$ and exploration probability $\varepsilon$?**  
  $\varepsilon$ controls **behavioral action selection** (trade-off between exploiting current knowledge vs. discovering new states). $\alpha$ controls **parameter update magnitude** (how much the existing value estimate shifts towards the newly observed TD target).

---

## 4. Part II: Deep Q-Networks (DQN) for Single Robot

### 4.1 Task 6: Network Architecture & Design
* **State Representation**: Cell $(r, c)$ mapped to a 25-dimensional one-hot vector with a 1 at index $5r + c$.
* **Neural Network**:
  $$\text{Input}(25) \longrightarrow \text{Linear}(25, 64) \longrightarrow \text{ReLU} \longrightarrow \text{Linear}(64, 64) \longrightarrow \text{ReLU} \longrightarrow \text{Linear}(64, 5)$$
* **Outputs**: 5 raw linear Q-values (no softmax, since Q-values are expected cumulative discounted returns, not probabilities).
* **Replay Buffer**: Capacity 20,000 transitions; minibatch size 64.
* **Target Network**: Periodic synchronization every 250 optimizer updates.
* **Optimization**: Adam ($\text{lr} = 0.001$), 1 update every 4 environment steps after 500 warmup steps. Detached TD targets computed via `torch.no_grad()`.

### 4.2 Synthetic Batch Loss Unit Test
Testing a synthetic 3-sample batch with $\gamma = 0.95$:
* Sample 1: Non-terminal, $r = 1.0, \max Q' = 2.0 \implies y_1 = 2.90$
* Sample 2: Terminal, $r = 5.0 \implies y_2 = 5.00$
* Sample 3: Truncated, $r = 1.0, \max Q' = 2.0 \implies y_3 = 2.90$
* Given online predictions $[1.0, 3.0, 2.0]$:
  $$\text{MSE} = \frac{(1.0 - 2.9)^2 + (3.0 - 5.0)^2 + (2.0 - 2.9)^2}{3} = \frac{3.61 + 4.00 + 0.81}{3} = \mathbf{2.80667}$$
* **Program Output**: Loss $= 2.80667$ (**PASS**). Gradients verified to flow into online network with zero leakage to target network.

### 4.3 Training & Greedy Trajectory
* **Budget**: 12,000 environment steps (seed 7). Plot saved at `plots/dqn_single_training.png`.
* **Greedy Evaluation Route**: `[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]`. Steps: 4. Return: $+9.70$. Invalid moves: 0.
* **Comparison with Q-table**: DQN replicates the exact optimal policy discovered by tabular Q-learning.

---

## 5. Part III: Two Robots and Collision Avoidance

### 5.1 Controller Formulation & Simultaneous Physics
* **Setup**:
  * Robot A: Start $(2, 0) \to$ Goal $(2, 4)$
  * Robot B: Start $(2, 4) \to$ Goal $(2, 0)$
  * Bottleneck doorway at $(2, 2)$.
* **Centralized Joint Controller**:
  * State: $(p_A, p_B)$ represented as two concatenated 25-dim one-hot vectors ($50$ inputs).
  * Joint Action: $j \in \{0, \dots, 24\}$, decoded as $(a_A, a_B) = \text{divmod}(j, 5)$. Network outputs 25 linear values.
* **Collision Rules**:
  * **Same-cell collision**: $p_A' = p_B'$.
  * **Swap collision**: $p_A' = p_B$ and $p_B' = p_A$ in one step.
  * Entering a waiting robot's cell is treated as a collision. Following a vacated cell is permitted.
  * On collision: Retain old positions and terminate immediately.
* **Reward Structure**:
  $$r = \begin{cases} -10, & \text{if collision attempted} \\ -0.1 - b + 5k + 5I, & \text{otherwise} \end{cases}$$
  ($b$: invalid moves by non-arrived robots; $k$: newly arrived robots; $I$: both arrived).

### 5.2 Pre-Training Environment Tests (Task 7 Table)
| Old $p_A, p_B$ | Action A | Action B | Expected Outcome | Verified Program Behavior | Status |
| :---: | :---: | :---: | :--- | :--- | :---: |
| $(2, 1), (2, 3)$ | Right | Left | Both propose $(2, 2)$ $\to$ Same-cell collision | Collision at $(2, 2)$; retain old pos; $r=-10.0$; term | **PASS** |
| $(2, 1), (2, 2)$ | Right | Left | Propose swap $\to$ Swap collision | Swap collision; retain old pos; $r=-10.0$; term | **PASS** |
| $(2, 1), (2, 2)$ | Right | Right | A enters $(2, 2)$ while B vacates to $(2, 3)$ | Permitted; final $((2, 2), (2, 3))$; $r=-0.1$; no term | **PASS** |
| $(2, 1), (2, 2)$ | Up | Wait | A moves to $(1, 1)$, B waits at $(2, 2)$ | Permitted; final $((1, 1), (2, 2))$; $r=-0.1$; no term | **PASS** |
| Goal Stay | Any | Any | Robot at goal stays and receives no extra reward | Position frozen at goal; $k=0$; no invalid penalty | **PASS** |

### 5.3 Training Dynamics & Seed Sensitivity Analysis

#### Standard Budget Run (Seed 7, 60,000 steps)
* **Outcome**: Collision-free, but Robot A timed out after Robot B reached goal.
* **Analysis**: Robot A yielded downwards to $(3, 0)$ and $(4, 0)$, allowing Robot B to move cleanly through the door $(2, 2)$ to $(2, 0)$ in 4 steps. Once Robot B arrived at $(2, 0)$ and froze, Robot A moved back up to $(3, 0)$ and fell into a local limit-cycle oscillation between $(3, 0)$ and $(3, 1)$ until reaching the 40-step cutoff. Total return: $+1.00$.

#### Multi-Seed Investigation (Seed 42, 60,000 steps)
As highlighted by the laboratory handout, training stability in joint-action DQNs can vary across seeds. Investigating Seed 42 revealed **complete, optimal mutual delivery**:

```
================================================================================
Step  | Robot A (Pos) | Action A | Robot B (Pos) | Action B | Reward  | Status              
--------------------------------------------------------------------------------
1     | (2, 0)        | Right    | (2, 4)        | Left     | -0.10   | OK                  
2     | (2, 1)        | Up       | (2, 3)        | Left     | -0.10   | OK (A yields Up)    
3     | (1, 1)        | Left     | (2, 2)        | Left     | -0.10   | OK (B enters Door)  
4     | (1, 0)        | Wait     | (2, 1)        | Left     | 4.90    | 1 Robot Arrived (B) 
5     | (1, 0)        | Right    | (2, 0)        | Right    | -0.10   | OK (A advances)     
6     | (1, 1)        | Down     | (2, 0)        | Down     | -0.10   | OK                  
7     | (2, 1)        | Right    | (2, 0)        | Right    | -0.10   | OK (A enters Door)  
8     | (2, 2)        | Right    | (2, 0)        | Left     | -0.10   | OK                  
9     | (2, 3)        | Right    | (2, 0)        | Up       | 9.90    | BOTH ARRIVED!       
--------------------------------------------------------------------------------
```
* **Summary (Seed 42)**: Steps = 9, Return = $+14.10$, Collisions = 0, Invalid Moves = 0. Trajectory plotted at `plots/dqn_multi_trajectory.png`.

#### Task 7 Questions:
* **(a) Which robot entered the passage first?**  
  Robot B entered the doorway $(2, 2)$ first at step 3.
* **(b) Where did the other robot yield?**  
  In Seed 42, Robot A yielded into the upper chamber at $(1, 1)$ and $(1, 0)$, waiting patiently while Robot B traversed the corridor. In Seed 7, Robot A yielded into the lower chamber at $(3, 0)$ and $(4, 0)$.
* **(c) Is avoiding collisions sufficient to count as success?**  
  **No.** As demonstrated by Seed 7, a policy that waits indefinitely or oscillates in place successfully avoids collisions (no $-10$ penalty) but completely fails the warehouse delivery mission (timing out at 40 steps).
* **(d) How would the joint-action output size change for three robots?**  
  With 3 robots each possessing 5 individual actions, the joint action space scales exponentially:
  $$|\mathcal{A}_{\text{joint}}| = 5^3 = \mathbf{125}\text{ linear outputs}$$
  The joint state representation would require concatenating three 25-dim vectors ($75$ inputs).

---

## 6. Evaluation & Synthesis (Task 8)

### 6.1 Master Comparison Table
| Metric | Q-table, Single Robot | DQN, Single Robot | DQN, Two Robots (Seed 7 / Seed 42) |
| :--- | :---: | :---: | :---: |
| **Budget and Seed** | 1,500 episodes, seed 7 | 12,000 steps, seed 7 | 60,000 steps, seed 7 (investigated seed 42) |
| **Outcome (Default Starts)** | **Success** | **Success** | Timeout (Seed 7) / **Success** (Seed 42) |
| **Steps to Goal** | 4 | 4 | N/A (Seed 7) / **9 steps** (Seed 42) |
| **Invalid Moves** | 0 | 0 | 0 |
| **Deterministic Total Return** | $+9.70$ | $+9.70$ | $+1.00$ (Seed 7) / $\mathbf{+14.10}$ (Seed 42) |

### 6.2 Unseen Start Generalization (Zero-Shot / No Retraining)
Testing the trained agents from unseen starting coordinates without additional training:
* Single Robot: Start at $(0, 0)$ $\to$ Goal $(2, 4)$
* Two Robots: Robot A at $(0, 0)$, Robot B at $(4, 4)$

| Controller | Unseen Start Position | Outcome | Steps | Invalid Moves | Return |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tabular Q-learning** | $(0, 0)$ | **Success** | 6 | 0 | $+9.50$ |
| **Single-Robot DQN** | $(0, 0)$ | **Success** | 6 | 0 | $+9.50$ |
| **Two-Robot DQN** | A: $(0, 0)$, B: $(4, 4)$ | **Timeout** | 40 | 1 | $0.00$ |

#### Explanation of Generalization Behavior:
1. **Single-Robot Agents**: Both Tabular Q-learning and Single-Robot DQN generalized to the unseen $(0, 0)$ start, discovering the shortest 6-step path: $(0,0) \to (1,0) \to (2,0) \to (2,1) \to (2,2) \to (2,3) \to (2,4)$. During $\varepsilon$-greedy training from $(2, 0)$, exploratory random walks visited upper cells, enabling the Q-values across column 0 to converge.
2. **Two-Robot Agent**: The two-robot centralized controller failed to generalize from unseen joint positions A:$(0,0)$, B:$(4,4)$. Because the joint state space consists of $25 \times 25 = 625$ possible positional configurations, training exclusively with rollouts initialized at A:$(2,0)$, B:$(2,4)$ left large regions of the joint space unvisited, resulting in arbitrary argmax choices that stalled in timeout.

---

## 7. Reflection on the LLM's Engineering Contribution

1. **Which parts of the design did you specify before requesting code?**  
   The exact $5 \times 5$ grid coordinates, obstacle set, action encodings, mutually exclusive rewards, TD update target equations ($y = r$ vs. $y = r + \gamma \max Q'$), distinction between termination and truncation, replay buffer size, and target network synchronization frequency.
2. **Did the LLM change any rewards, stopping rules, or algorithm details?**  
   Initial LLM drafts frequently conflate `terminated` and `truncated`, erroneously zeroing out the bootstrap continuation term when an episode times out. Rigorous specification was required to maintain target bootstrapping upon truncation.
3. **Which errors did your tests reveal?**  
   Our environment test suite revealed a subtle edge case in two-robot arrival scoring: if a robot initialized directly on its goal coordinate, arrival rewards could trigger spuriously unless `arrived` status flags were initialized during environment reset.
4. **Did the code use a replay buffer and target network as requested?**  
   Yes. The replay buffer preserved experience tuples, and targets were detached using `torch.no_grad()` with target weights synchronized every 250 optimizer updates.
5. **Which generated functions can you now explain without consulting the LLM?**  
   All core components: the deterministic collision logic (same-cell vs. swap collision detection), the gather operation for tensor Q-value indexing (`online_q.gather(1, actions)`), and the epsilon schedule.
6. **What did you change, and what evidence shows the change was correct?**  
   Explicitly separated termination ($d_i = 1$) from truncation ($d_i = 0$) in both tabular updates and tensor loss calculations. Synthetic unit tests confirmed exact numerical convergence to hand-calculated values ($0.90$ vs. $1.32$ vs. $2.80667$).

---

## 8. Optional: Connection to Reinforcement Learning from Human Preferences (RLHF)

1. **What feedback people would provide**:  
   Human evaluators presented with two trajectory videos would select which rollout looked more natural, courteous, or efficient (e.g., preferring an agent that smoothly steps aside to let the other pass rather than one that awkwardly jitters).
2. **What a learned reward model would predict**:  
   A reward model (typically a neural network trained with Bradley-Terry preference loss) would take trajectory representations and output a scalar score predicting the probability that a human would prefer trajectory $A$ over trajectory $B$.
3. **How an RL agent could use that reward model**:  
   The RL agent uses the learned reward model as the environment's reward function $R(s, a)$, optimizing its policy via PPO or DQN to maximize the preference score (often with a KL penalty against a base policy).
4. **Does editing the hand-written collision penalty train a reward model?**  
   **No.** Modifying a manual scalar penalty (e.g., changing $-10$ to $-50$) is heuristic hyperparameter tuning. It does not train a model from pairwise preference data; it directly alters the ground-truth reward signal provided by the environment designer.

---

## 9. Reproducibility & File Inventory

All code is self-contained and requires no third-party RL frameworks (only standard `numpy`, `torch`, `matplotlib`):

* `warehouse_env.py`: Full single-robot and two-robot environment classes with collision physics.
* `test_warehouse_env.py`: Formal unit test suite for Tasks 4 & 7.
* `tabular_q.py`: Single-robot tabular Q-learning with unit tests and training/eval routines.
* `dqn_single.py`: PyTorch single-robot DQN with detached target loss unit tests.
* `dqn_multi.py`: PyTorch two-robot joint DQN with trajectory table and collision reporting.
* `compare_multi_seeds.py`: Multi-seed analysis script.
* `evaluate_all.py`: Unified pipeline reproducing all figures and tables.
* `plots/`: Contains all generated visualization curves and grid trajectory figures.
