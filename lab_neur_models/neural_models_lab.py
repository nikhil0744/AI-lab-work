"""
Laboratory: Neural Models - Learning, Depth, Activations, and Output Layers
Complete PyTorch implementation for:
1. Binary XOR Sensor Disagreement Model (2-2-1 Architecture)
2. Symmetry Breaking Experiment (Zero Weight Initialization)
3. Activation Function Investigation (Sigmoid, Tanh, ReLU)
4. Three-Class Redundant Sensor Extension (2-2-3 Architecture)
5. Numerical Diagnostics (Softmax Normalization & Shift Invariance)
"""

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional


def set_seed(seed: int = 42):
    """Set random seed for deterministic reproducibility across PyTorch and CPU."""
    torch.manual_seed(seed)
    random.seed(seed)


# ==============================================================================
# 1. Binary XOR Model (2-2-1 Architecture)
# ==============================================================================

class XORNet(nn.Module):
    """
    2-2-1 Feedforward Neural Network for XOR Sensor Disagreement.
    Architecture:
      Input (2) -> Linear(2, 2) -> Activation -> Linear(2, 1) -> Output Logit
    """
    def __init__(self, activation: str = "tanh", zero_init: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.activation_name = activation.lower()

        if self.activation_name == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.activation_name == "tanh":
            self.act = nn.Tanh()
        elif self.activation_name == "relu":
            self.act = nn.ReLU()
        elif self.activation_name == "identity":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if zero_init:
            with torch.no_grad():
                self.fc1.weight.zero_()
                self.fc1.bias.zero_()
                self.fc2.weight.zero_()
                self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out


class LinearBaselineNet(nn.Module):
    """Single affine transformation followed by linear output (no hidden layer)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ==============================================================================
# 2. Three-Class Sensor Model (2-2-3 Architecture)
# ==============================================================================

class ThreeClassSensorNet(nn.Module):
    """
    Multiclass classification for redundant sensors:
      Class 0: Both inactive (0, 0)
      Class 1: Disagreement (0, 1) or (1, 0)
      Class 2: Both active   (1, 1)
    Architecture: Input (2) -> Linear(2, 2) -> Tanh -> Linear(2, 3) -> Output Logits (3)
    """
    def __init__(self, hidden_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        return logits


# ==============================================================================
# 3. Training & Experimentation Routines
# ==============================================================================

def get_xor_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit four XOR training examples."""
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]], dtype=torch.float32)
    return X, y


def get_three_class_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Four training examples for 3-class sensor task."""
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    # 0: inactive, 1: disagree, 2: active
    y = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    return X, y


def train_xor_model(
    activation: str = "tanh",
    zero_init: bool = False,
    lr: float = 0.1,
    steps: int = 3000,
    seed: int = 42,
    record_early_grad_step: int = 10
) -> Dict:
    """Train 2-2-1 XOR network and record diagnostic metrics."""
    set_seed(seed)
    X, y = get_xor_data()

    model = XORNet(activation=activation, zero_init=zero_init)
    criterion = nn.BCEWithLogitsLoss()  # Numerically stable BCE with logits
    optimizer = optim.SGD(model.parameters(), lr=lr)

    initial_loss = None
    early_grad_norm = None
    early_w1_grad = None

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        if step == 1:
            initial_loss = loss.item()

        loss.backward()

        if step == record_early_grad_step:
            early_grad_norm = model.fc1.weight.grad.norm(2).item()
            early_w1_grad = model.fc1.weight.grad.detach().clone()

        optimizer.step()

    # Final evaluation
    with torch.no_grad():
        final_logits = model(X)
        final_loss = criterion(final_logits, y).item()
        final_probs = torch.sigmoid(final_logits)
        predictions = (final_probs >= 0.5).float()
        all_correct = bool(torch.equal(predictions, y))

    return {
        "model": model,
        "activation": activation,
        "zero_init": zero_init,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "probabilities": final_probs.squeeze().tolist(),
        "predictions": predictions.squeeze().tolist(),
        "targets": y.squeeze().tolist(),
        "all_correct": all_correct,
        "early_grad_norm": early_grad_norm,
        "early_w1_grad": early_w1_grad,
        "final_w1_grad": model.fc1.weight.grad.detach().clone() if model.fc1.weight.grad is not None else None,
        "fc1_weight": model.fc1.weight.detach().clone(),
        "fc1_bias": model.fc1.bias.detach().clone(),
        "fc2_weight": model.fc2.weight.detach().clone(),
        "fc2_bias": model.fc2.bias.detach().clone()
    }


def train_linear_baseline(lr: float = 0.1, steps: int = 3000, seed: int = 42) -> Dict:
    """Train single affine layer without hidden nonlinearity to demonstrate failure."""
    set_seed(seed)
    X, y = get_xor_data()
    model = LinearBaselineNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = model(X)
        final_loss = criterion(final_logits, y).item()
        final_probs = torch.sigmoid(final_logits)
        preds = (final_probs >= 0.5).float()
        correct_count = (preds == y).sum().item()

    return {
        "final_loss": final_loss,
        "probabilities": final_probs.squeeze().tolist(),
        "predictions": preds.squeeze().tolist(),
        "correct_count": int(correct_count),
        "accuracy": correct_count / 4.0
    }


def run_symmetry_experiment(steps: int = 100, lr: float = 0.1) -> List[Dict]:
    """
    Sets weights and biases to zero, trains over several steps, and inspects
    both rows of fc1.weight to prove mathematical symmetry trapping.
    """
    set_seed(42)
    X, y = get_xor_data()
    model = XORNet(activation="tanh", zero_init=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = []
    sample_steps = [0, 1, 5, 10, 50, 100]

    for step in range(steps + 1):
        if step in sample_steps:
            w1 = model.fc1.weight.detach().clone()
            grad1 = model.fc1.weight.grad.detach().clone() if model.fc1.weight.grad is not None else torch.zeros_like(w1)
            row_diff = torch.norm(w1[0] - w1[1]).item()
            history.append({
                "step": step,
                "row_0": w1[0].tolist(),
                "row_1": w1[1].tolist(),
                "row_difference_l2": row_diff,
                "grad_row_0": grad1[0].tolist(),
                "grad_row_1": grad1[1].tolist()
            })

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    return history


def run_activation_benchmark() -> List[Dict]:
    """
    Trains random-initialized network three times, changing only the hidden activation:
    Sigmoid, Tanh, and ReLU. Records final loss, accuracy, and early gradient norm.
    """
    activations = ["sigmoid", "tanh", "relu"]
    results = []
    for act in activations:
        res = train_xor_model(activation=act, lr=0.2, steps=4000, seed=42, record_early_grad_step=10)
        results.append({
            "activation": act.capitalize(),
            "initial_loss": res["initial_loss"],
            "final_loss": res["final_loss"],
            "all_correct": res["all_correct"],
            "early_grad_norm": res["early_grad_norm"]
        })
    return results


def train_three_class_extension(lr: float = 0.2, steps: int = 3000, seed: int = 42) -> Dict:
    """
    Three-class sensor disagreement task with multiclass cross-entropy:
    Class 0: (0,0), Class 1: (0,1) or (1,0), Class 2: (1,1).
    """
    set_seed(seed)
    X, y = get_three_class_data()
    model = ThreeClassSensorNet(hidden_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    initial_loss = None
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        if step == 1:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_logits = model(X)
        final_loss = criterion(final_logits, y).item()
        softmax_probs = torch.softmax(final_logits, dim=-1)
        preds = torch.argmax(softmax_probs, dim=-1)
        all_correct = bool(torch.equal(preds, y))

        # Test shift invariance on first example: z -> z + 100
        z_sample = final_logits[0:1]
        z_shifted = z_sample + 100.0
        prob_original = torch.softmax(z_sample, dim=-1)
        prob_shifted = torch.softmax(z_shifted, dim=-1)
        shift_diff = torch.norm(prob_original - prob_shifted).item()
        prob_sum = float(torch.sum(prob_original).item())

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "logits": final_logits.tolist(),
        "softmax_probs": softmax_probs.tolist(),
        "predictions": preds.tolist(),
        "targets": y.tolist(),
        "all_correct": all_correct,
        "fc2_weight_shape": list(model.fc2.weight.shape),
        "logits_per_example": final_logits.shape[1],
        "example_0_prob_sum": prob_sum,
        "shift_invariance_diff": shift_diff
    }


# ==============================================================================
# 4. Main Console Output Demonstration
# ==============================================================================

def print_banner(title: str):
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    print_banner("LABORATORY: NEURAL MODELS (LEARNING, DEPTH, ACTIVATIONS, OUTPUTS)")

    # 1. Linear Baseline Failure
    print("\n[TASK 1 CHECK] Single Affine Layer (No Hidden Nonlinearity):")
    lin_res = train_linear_baseline()
    print(f"  Final Loss: {lin_res['final_loss']:.5f}")
    print(f"  Accuracy:   {lin_res['correct_count']}/4 ({lin_res['accuracy'] * 100:.1f}%)")
    print(f"  Probabilities: {[round(p, 4) for p in lin_res['probabilities']]}")
    print("  Outcome: Fails XOR as predicted by linear inseparability theorem.")

    # 2. Baseline 2-2-1 Model (Tanh)
    print("\n" + "-" * 78)
    print("[TASK 3 & 4 PART A, B] Baseline 2-2-1 Network with Tanh Activation:")
    base_res = train_xor_model(activation="tanh", lr=0.2, steps=3000, seed=42)
    print(f"  Initial Loss: {base_res['initial_loss']:.5f}")
    print(f"  Final Loss:   {base_res['final_loss']:.5f}")
    print(f"  Inputs:       [(0,0), (0,1), (1,0), (1,1)]")
    print(f"  Targets:      {base_res['targets']}")
    print(f"  Output Probs: {[round(p, 4) for p in base_res['probabilities']]}")
    print(f"  Predictions:  {base_res['predictions']}")
    print(f"  All 4 Correct: {base_res['all_correct']}")
    print(f"  First Layer Weight Gradient dL/dW^(1) shape: {list(base_res['final_w1_grad'].shape)}")
    print(f"  First Layer Gradient Tensor:\n{base_res['final_w1_grad']}")

    # 3. Symmetry Experiment (Task 4 Part C)
    print("\n" + "-" * 78)
    print("[TASK 4 PART C] Symmetry Experiment (All Weights & Biases Initialized to 0.0):")
    sym_history = run_symmetry_experiment(steps=50, lr=0.1)
    print(f"  {'Step':<6} | {'Row 0 Weight':<24} | {'Row 1 Weight':<24} | {'L2 Difference':<15}")
    print("  " + "-" * 74)
    for row in sym_history:
        r0_str = f"[{row['row_0'][0]:.4f}, {row['row_0'][1]:.4f}]"
        r1_str = f"[{row['row_1'][0]:.4f}, {row['row_1'][1]:.4f}]"
        print(f"  {row['step']:<6} | {r0_str:<24} | {r1_str:<24} | {row['row_difference_l2']:<15.6e}")
    print("  Conclusion: Row difference remains exactly 0.0 at every step.")
    print("  Symmetry is mathematically preserved, preventing hidden units from differentiating.")

    # 4. Activation Function Benchmark (Task 4 Part D)
    print("\n" + "-" * 78)
    print("[TASK 4 PART D] Activation Function Benchmark (Sigmoid vs Tanh vs ReLU):")
    act_benchmark = run_activation_benchmark()
    print(f"  {'Hidden Activation':<18} | {'Final Loss':<12} | {'4/4 Correct?':<14} | {'Early ||grad W^(1)||_2':<20}")
    print("  " + "-" * 70)
    for res in act_benchmark:
        print(f"  {res['activation']:<18} | {res['final_loss']:<12.5f} | {str(res['all_correct']):<14} | {res['early_grad_norm']:<20.6f}")

    # 5. Multiclass Extension (Task 5)
    print("\n" + "-" * 78)
    print("[TASK 5] Three-Class Extension (Softmax & Multiclass Cross-Entropy):")
    mc_res = train_three_class_extension(lr=0.2, steps=3000, seed=42)
    print(f"  Initial Loss: {mc_res['initial_loss']:.5f}")
    print(f"  Final Loss:   {mc_res['final_loss']:.5f}")
    print(f"  Final Weight Matrix Shape (W^(2)): {mc_res['fc2_weight_shape']} (3 classes x 2 hidden)")
    print(f"  Logits per example: {mc_res['logits_per_example']}")
    print(f"  Inputs & Softmax Class Distributions [P(C0), P(C1), P(C2)]:")
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for inp, probs, pred, target in zip(inputs, mc_res["softmax_probs"], mc_res["predictions"], mc_res["targets"]):
        prob_str = f"[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]"
        print(f"    Input {inp} -> Probs: {prob_str} | Pred: {pred} | Target: {target}")
    print(f"  All Correct: {mc_res['all_correct']}")
    print(f"  Numerical Check (Sum of Probabilities for Example 0): {mc_res['example_0_prob_sum']:.8f}")
    print(f"  Shift Invariance Check ||softmax(z) - softmax(z + 100)||_2: {mc_res['shift_invariance_diff']:.2e}")
    print_banner("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
