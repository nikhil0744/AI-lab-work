"""
Unit tests for Laboratory: Neural Models
Validates:
1. Shape and Forward Pass of 2-2-1 XORNet and 2-2-3 ThreeClassSensorNet
2. Failure of Single Affine Layer (Linear Separability Limitation)
3. Convergence of 2-2-1 XORNet (Loss < 0.05, 100% Accuracy)
4. Mathematical Symmetry Preservation under Zero Initialization
5. Softmax Normalization (Sum = 1.0) and Logit Shift Invariance
"""

import unittest
import torch
import torch.nn as nn
from lab_neur_models.neural_models_lab import (
    XORNet,
    ThreeClassSensorNet,
    LinearBaselineNet,
    get_xor_data,
    get_three_class_data,
    train_xor_model,
    train_linear_baseline,
    run_symmetry_experiment,
    train_three_class_extension,
    set_seed
)


class TestNeuralModelsLab(unittest.TestCase):

    def setUp(self):
        set_seed(42)

    def test_xor_data_shapes(self):
        X, y = get_xor_data()
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4, 1))

    def test_three_class_data_shapes(self):
        X, y = get_three_class_data()
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4,))

    def test_linear_baseline_fails_xor(self):
        """A single affine model cannot achieve 100% accuracy on XOR."""
        res = train_linear_baseline(lr=0.1, steps=2000, seed=42)
        self.assertLess(res["correct_count"], 4, "Linear model should fail to separate XOR")
        self.assertGreaterEqual(res["final_loss"], 0.60)

    def test_xor_convergence_tanh(self):
        """2-2-1 Network with Tanh hidden activation solves XOR with loss < 0.05."""
        res = train_xor_model(activation="tanh", lr=0.2, steps=3000, seed=42)
        self.assertTrue(res["all_correct"], "XORNet with Tanh should achieve 4/4 accuracy")
        self.assertLess(res["final_loss"], 0.05, f"Expected loss < 0.05, got {res['final_loss']}")
        self.assertEqual(list(res["final_w1_grad"].shape), [2, 2])

    def test_symmetry_preservation(self):
        """Zero-initialized weights remain mathematically identical across rows."""
        history = run_symmetry_experiment(steps=50, lr=0.1)
        for entry in history:
            self.assertEqual(
                entry["row_difference_l2"],
                0.0,
                f"Row difference should be 0.0 at step {entry['step']}"
            )

    def test_three_class_model_convergence_and_invariants(self):
        """3-Class sensor net achieves 100% accuracy, sums to 1.0, and exhibits shift invariance."""
        res = train_three_class_extension(lr=0.2, steps=3000, seed=42)
        self.assertTrue(res["all_correct"], "Three-class model should classify all 4 inputs correctly")
        self.assertLess(res["final_loss"], 0.05)
        self.assertEqual(res["fc2_weight_shape"], [3, 2])
        self.assertAlmostEqual(res["example_0_prob_sum"], 1.0, places=6)
        self.assertLess(res["shift_invariance_diff"], 1e-6)


if __name__ == "__main__":
    unittest.main()
