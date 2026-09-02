# Laboratory Report: Logical Reasoning for Planning
**Course:** Undergraduate Artificial Intelligence (CS-407)  
**Topic:** Using an LLM to Construct and Test a Simple Planning Agent  

---

## Task 0: Understand the Planning Problem

### Problem Formulation
- **(a) Initial State ($I$):**
  $$I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$$

- **(b) Goal State ($G$):**
  $$G = \{\text{At}(\text{Package}, C)\}$$

- **(c) Available Actions:**
  - $\text{Move}(X, Y)$ for connected pairs $(X, Y) \in \{(A, B), (B, A), (B, C), (C, B)\}$
  - $\text{PickUp}(\text{Package}, X)$ for $X \in \{A, B, C\}$
  - $\text{Drop}(\text{Package}, X)$ for $X \in \{A, B, C\}$

- **(d) Action Preconditions and Effects:**

| Action | Positive Preconditions | Negative Preconditions | Positive Effects | Negative Effects |
| :--- | :--- | :--- | :--- | :--- |
| $\text{Move}(X, Y)$ | $\{\text{At}(\text{Robot}, X)\}$ | $\emptyset$ | $\{\text{At}(\text{Robot}, Y)\}$ | $\{\text{At}(\text{Robot}, X)\}$ |
| $\text{PickUp}(\text{Package}, X)$ | $\{\text{At}(\text{Robot}, X), \text{At}(\text{Package}, X)\}$ | $\{\text{Holding}(\text{Package})\}$ | $\{\text{Holding}(\text{Package})\}$ | $\{\text{At}(\text{Package}, X)\}$ |
| $\text{Drop}(\text{Package}, X)$ | $\{\text{At}(\text{Robot}, X), \text{Holding}(\text{Package})\}$ | $\emptyset$ | $\{\text{At}(\text{Package}, X)\}$ | $\{\text{Holding}(\text{Package})\}$ |

---

### Task 0 Questions
**Starting from $I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$:**
1. **Is $\text{PickUp}(\text{Package}, A)$ applicable?**
   - **Yes.** Preconditions are $\text{At}(\text{Robot}, A)$ and $\text{At}(\text{Package}, A)$. Both facts are present in state $I$, and the robot is not holding the package. Therefore, $I \models \text{Preconditions}(\text{PickUp}(\text{Package}, A))$.
2. **Is $\text{Drop}(\text{Package}, C)$ applicable?**
   - **No.** Preconditions are $\text{At}(\text{Robot}, C)$ and $\text{Holding}(\text{Package})$. Neither proposition is present in state $I$.

---

## Task 1: Construct a Plan by Hand

### Action Sequence
1. $a_1 = \text{PickUp}(\text{Package}, A)$
2. $a_2 = \text{Move}(A, B)$
3. $a_3 = \text{Move}(B, C)$
4. $a_4 = \text{Drop}(\text{Package}, C)$

### State Transition Trace

| State | Action Applied | Facts in State |
| :--- | :--- | :--- |
| $S_0$ (Initial) | — | $\{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$ |
| $S_1$ | $\text{PickUp}(\text{Package}, A)$ | $\{\text{At}(\text{Robot}, A), \text{Holding}(\text{Package})\}$ |
| $S_2$ | $\text{Move}(A, B)$ | $\{\text{At}(\text{Robot}, B), \text{Holding}(\text{Package})\}$ |
| $S_3$ | $\text{Move}(B, C)$ | $\{\text{At}(\text{Robot}, C), \text{Holding}(\text{Package})\}$ |
| $S_4$ (Final) | $\text{Drop}(\text{Package}, C)$ | $\{\text{At}(\text{Robot}, C), \text{At}(\text{Package}, C)\}$ |

**Goal Verification:**  
$G = \{\text{At}(\text{Package}, C)\} \subseteq S_4 \implies S_4 \models G$. The plan is valid and minimal.

---

## Task 2: Implementation of the Planner

The planning program is implemented in Python (`planner.py`).

### Key Code Architecture
- **State Representation:** A Python `set` of proposition strings (e.g., `{"At(Robot, A)", "At(Package, A)"}`).
- **Action Class:** Stores positive/negative preconditions and positive/negative effects.
- **Applicability (`is_applicable`):**
  $$\text{pos\_preconditions} \subseteq S \quad \land \quad \text{neg\_preconditions} \cap S = \emptyset$$
- **State Transition (`apply`):**
  $$S' = (S \setminus \text{neg\_effects}) \cup \text{pos\_effects}$$
- **Search Algorithm:** Breadth-First Search (BFS) using a `collections.deque` and a `set` of `frozenset` to track visited states, guaranteeing the shortest sequence of actions.

---

## Task 3: Test the Generated Planner

### Test Results Summary

| Test Case | Initial State | Goal | Plan Found? | Resulting Plan | Valid? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test A: Solvable Problem** | `At(Robot, A)`, `At(Package, A)` | `At(Package, C)` | Yes | 1. `PickUp(Package, A)`<br>2. `Move(A, B)`<br>3. `Move(B, C)`<br>4. `Drop(Package, C)` | Yes |
| **Test B: Impossible Problem** | `At(Robot, A)`, `At(Package, A)` | `At(Package, C)` | No | `No plan found` | Yes |
| **Test C: Irrelevant Actions** | `At(Robot, A)`, `At(Package, A)` | `At(Package, C)` | Yes | 1. `PickUp(Package, A)`<br>2. `Move(A, B)`<br>3. `Move(B, C)`<br>4. `Drop(Package, C)` | Yes |

### Execution Output:
```text
============================================================
TEST: Test A (Solvable Warehouse Problem)
Initial State : ['At(Package, A)', 'At(Robot, A)']
Goal          : ['At(Package, C)']
------------------------------------------------------------
Result: Plan found (4 actions)

S0 (Initial) : ['At(Package, A)', 'At(Robot, A)']
  Step 1: Execute -> PickUp(Package, A)
  S1 State   : ['At(Robot, A)', 'Holding(Package)']
  Step 2: Execute -> Move(A, B)
  S2 State   : ['At(Robot, B)', 'Holding(Package)']
  Step 3: Execute -> Move(B, C)
  S3 State   : ['At(Robot, C)', 'Holding(Package)']
  Step 4: Execute -> Drop(Package, C)
  S4 State   : ['At(Package, C)', 'At(Robot, C)']
============================================================

============================================================
TEST: Test B (Impossible Problem - PickUp Removed)
Initial State : ['At(Package, A)', 'At(Robot, A)']
Goal          : ['At(Package, C)']
------------------------------------------------------------
Result: No plan found
============================================================

============================================================
TEST: Test C (Irrelevant Actions / Goal Discrimination)
Initial State : ['At(Package, A)', 'At(Robot, A)']
Goal          : ['At(Package, C)']
------------------------------------------------------------
Result: Plan found (4 actions)
...
============================================================
```

---

## Task 4: Logic and Search

### Roles of Logic and Search
- **Logic:** Evaluates local state validity and transitions. It decides whether an action $a$ can be legally executed in state $S$ ($S \models \text{Preconditions}(a)$) and computes the exact world update ($S' = \text{Apply}(S, a)$).
- **Search:** Manages global state exploration. It decides which states to explore and in what order (e.g. BFS for shortest plan) to reach a state where $S \models G$.

### Completed Flow Diagram
```
        Current state
              ↓
  Check action preconditions
              ↓
  [ Filter Applicable Actions ]  <--- (Missing step: Determine applicable actions)
              ↓
    Generate successor state
              ↓
    Search over alternatives
              ↓
             Goal?
```

---

## Task 5: LLM Plan Verification & Trust

### Question: Which should you trust more?
**(b) The independently executed state transitions.**

**Explanation:**  
An LLM generates natural language text probabilistically; it may invent missing transitions, hallucinate satisfied preconditions, or assume intermediate steps that violate formal domain constraints. In contrast, an independent deterministic execution engine mathematically checks $S \models \text{Preconditions}(a)$ against state representations. Independent verification guarantees soundness.

---

## Section 5: Reflection Questions

1. **Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**  
   It provides formal semantics and boundaries. Specifying precise STRIPS-like schemas prevents the LLM from making ambiguous assumptions about physical laws or world dynamics.

2. **Give an example of an error that could occur if the planner failed to check an action’s preconditions.**  
   The robot could execute `Drop(Package, C)` at location $C$ without ever moving to $A$ or executing `PickUp(Package, A)`, resulting in teleporting the package out of thin air.

3. **Why is a plan that “looks reasonable” not necessarily a valid plan?**  
   A sequence might sound intuitive in natural language (e.g., "Robot moves to C and drops package"), but subtle formal preconditions may be violated (e.g., robot never picked up the package at A, or locations are not directly connected).

4. **What did the LLM contribute to the implementation?**  
   The LLM provided boilerplate code, data structures, and the skeleton for state updates and BFS graph traversal.

5. **What did you have to verify independently?**  
   Precondition checking logic, state immutability during graph exploration, set difference/union correctness, termination conditions, and edge-case handling (e.g., unsolvable goals).

6. **In this laboratory, where is logical reasoning being used?**  
   - In verifying whether $S \models \text{Preconditions}(a)$.  
   - In verifying whether goal conditions hold in state $S_n$ ($S_n \models G$).  
   - In computing successor states using add/delete effect sets.

7. **How is planning related to the search algorithms studied in the previous module?**  
   Planning is state-space search where nodes are world states (sets of predicates), edges are applicable actions, and successor generation is governed by logical action rules rather than hardcoded adjacency matrices.

---

## Tasks 6–8: Prolog as a Logical Verifier (Optional Extension)

### Task 6: Prolog as a Plan Verifier (`planner.pl`)
- `connected(a, b). connected(b, a). connected(b, c). connected(c, b).`
- `can_move(X, Y) :- connected(X, Y).`

**Queries:**
- `?- can_move(a, b).` $\rightarrow$ `true.` (Direct fact in KB)
- `?- can_move(a, c).` $\rightarrow$ `false.` (No direct connection fact in KB)

**Relationship:** The Prolog rule represents the first-order implication $\forall X, Y (\text{Connected}(X, Y) \to \text{CanMove}(X, Y))$.

### Task 7: Checking Proposed Plans
- `?- valid_move(a, b).` $\rightarrow$ `true.`
- `?- valid_move(b, c).` $\rightarrow$ `true.`
- `?- valid_move(a, c).` $\rightarrow$ `false.`  
*Result:* Proposed action `Move(a, c)` is invalid because $A$ and $C$ are not directly connected.

### Task 8: Logical Reasoning Chain (`reasoning.pl`)
**Implication Sequence:**
$$\text{wet\_road} \implies (\text{wet\_road} \to \text{slippery}) \implies (\text{slippery} \to \text{reduce\_speed}) \implies \text{reduce\_speed}$$

---

## Section 7.2: Prolog Reflection Questions

1. **Difference between a Prolog fact and rule:**  
   A *fact* is an unconditional assertion (axiom, e.g. `connected(a, b).`). A *rule* is a conditional Horn clause with a head and body ($H \leftarrow B_1 \land \dots \land B_n$).

2. **Correspondence of query to knowledge base:**  
   A query $Q$ asks whether $KB \vdash Q$ (i.e. whether $Q$ is logically entailed by the axioms and rules in the knowledge base via SLD resolution).

3. **Why use Prolog to verify a Python-generated plan?**  
   Prolog provides a declarative logic engine completely separated from the imperative search code, enabling independent, formally verifiable rule validation.

4. **Advantage of an independent verifier for LLM outputs:**  
   It decouples *heuristic generation* from *formal verification* ($\text{Generate} \to \text{Verify}$). An LLM can be creative and rapid, while the independent verifier guarantees correctness, safety, and validity.
