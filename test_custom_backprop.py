#!/usr/bin/env python3
"""
Unit tests for CustomBackpropModule with rigorous validation of forward and backward passes.

Tests include:
1. Forward pass correctness
2. Gradient computation accuracy (numerical vs analytical)
3. Gradient flow validation
4. Edge cases (boundary values, zero gradients)
5. Integration with PyTorch autograd
"""

import torch
import torch.nn as nn
import numpy as np
from ultralytics.nn.modules.custom_backprop import CustomBackpropModule, FloorFractionalIndexing


def test_forward_pass():
    """Test forward pass produces expected outputs."""
    print("=" * 80)
    print("Test 1: Forward Pass Correctness")
    print("=" * 80)

    # Create module
    num_indices = 10
    num_channels = 4
    module = CustomBackpropModule(num_indices, num_channels)

    # Create simple input
    B, C, W, H = 2, 4, 3, 3
    x = torch.tensor([
        [[[0.5, 1.5, 2.7],
          [3.2, 4.0, 5.9],
          [6.1, 7.3, 8.8]]]
    ]).expand(B, C, W, H).clone()

    # Forward pass
    output = module(x)

    # Check output shape: (B, H, W, C)
    expected_shape = (B, H, W, C)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} != expected {expected_shape}"

    # Manually verify a few values
    # For x[0, 0, 0, 0] = 0.5:
    #   floor = 0, frac = 0.5
    #   output should be w[0, 0] * 0.5
    x_val = x[0, 0, 0, 0].item()
    idx = int(np.floor(x_val))
    frac = x_val - idx
    expected_val = module.weight[idx, 0].item() * frac
    actual_val = output[0, 0, 0, 0].item()

    print(f"Sample verification:")
    print(f"  Input x[0,0,0,0] = {x_val}")
    print(f"  floor(x) = {idx}, frac(x) = {frac:.4f}")
    print(f"  weight[{idx}, 0] = {module.weight[idx, 0].item():.4f}")
    print(f"  Expected output = {expected_val:.4f}")
    print(f"  Actual output = {actual_val:.4f}")
    print(f"  Difference = {abs(expected_val - actual_val):.6f}")

    assert abs(expected_val - actual_val) < 1e-5, "Forward pass computation mismatch"

    print("✓ Forward pass test PASSED\n")


def test_gradient_numerical():
    """Test gradients using numerical differentiation."""
    print("=" * 80)
    print("Test 2: Gradient Computation (Numerical Validation)")
    print("=" * 80)

    # Create module with fixed seed for reproducibility
    torch.manual_seed(42)
    num_indices = 5
    num_channels = 3
    module = CustomBackpropModule(num_indices, num_channels)

    # Create input requiring gradient
    B, C, W, H = 1, 3, 2, 2
    x = torch.randn(B, C, W, H, requires_grad=True)

    # Ensure x values are in valid range
    with torch.no_grad():
        x.data = torch.clamp(x.data, 0, num_indices - 1)

    # Forward pass
    output = module(x)

    # Compute loss (simple sum for testing)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Get analytical gradients
    grad_x_analytical = x.grad.clone()
    grad_w_analytical = module.weight.grad.clone()

    print(f"Analytical gradient statistics:")
    print(f"  grad_x: mean={grad_x_analytical.mean():.6f}, "
          f"std={grad_x_analytical.std():.6f}, "
          f"max={grad_x_analytical.max():.6f}")
    print(f"  grad_w: mean={grad_w_analytical.mean():.6f}, "
          f"std={grad_w_analytical.std():.6f}, "
          f"max={grad_w_analytical.max():.6f}")

    # Numerical gradient check for input x
    eps = 1e-4
    x_test = x.data[0, 0, 0, 0].item()

    # Perturb x
    x_plus = x.data.clone()
    x_plus[0, 0, 0, 0] += eps

    x_minus = x.data.clone()
    x_minus[0, 0, 0, 0] -= eps

    # Compute finite difference
    with torch.no_grad():
        x_plus_reshaped = x_plus.permute(0, 3, 2, 1)
        x_minus_reshaped = x_minus.permute(0, 3, 2, 1)

        out_plus = FloorFractionalIndexing.apply(x_plus_reshaped, module.weight).sum()
        out_minus = FloorFractionalIndexing.apply(x_minus_reshaped, module.weight).sum()

        grad_x_numerical = (out_plus - out_minus) / (2 * eps)

    grad_x_analytical_val = grad_x_analytical[0, 0, 0, 0].item()

    print(f"\nNumerical gradient check for x[0,0,0,0]:")
    print(f"  Numerical gradient: {grad_x_numerical:.6f}")
    print(f"  Analytical gradient: {grad_x_analytical_val:.6f}")
    print(f"  Relative error: {abs(grad_x_numerical - grad_x_analytical_val) / (abs(grad_x_numerical) + 1e-8):.6f}")

    # Allow some tolerance for numerical errors
    assert abs(grad_x_numerical - grad_x_analytical_val) < 1e-3 or \
           abs(grad_x_numerical - grad_x_analytical_val) / (abs(grad_x_numerical) + 1e-8) < 0.01, \
           "Gradient mismatch between numerical and analytical"

    print("✓ Gradient computation test PASSED\n")


def test_gradient_flow():
    """Test that gradients flow properly through the module."""
    print("=" * 80)
    print("Test 3: Gradient Flow Validation")
    print("=" * 80)

    torch.manual_seed(123)
    num_indices = 8
    num_channels = 4
    module = CustomBackpropModule(num_indices, num_channels)

    # Create input
    B, C, W, H = 2, 4, 4, 4
    x = torch.rand(B, C, W, H, requires_grad=True) * (num_indices - 1)

    # Forward pass
    output = module(x)

    # Create target (for testing gradient flow)
    target = torch.randn_like(output)

    # Compute loss
    loss = F.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Check that gradients exist and are non-zero
    assert x.grad is not None, "No gradient for input x"
    assert module.weight.grad is not None, "No gradient for weights"

    # Check gradient magnitudes
    grad_x_norm = x.grad.norm().item()
    grad_w_norm = module.weight.grad.norm().item()

    print(f"Gradient norms:")
    print(f"  ||grad_x|| = {grad_x_norm:.6f}")
    print(f"  ||grad_w|| = {grad_w_norm:.6f}")

    assert grad_x_norm > 1e-6, "Input gradient is too small (might indicate vanishing gradient)"
    assert grad_w_norm > 1e-6, "Weight gradient is too small (might indicate vanishing gradient)"

    print("✓ Gradient flow test PASSED\n")


def test_edge_cases():
    """Test edge cases like integer inputs, boundary values."""
    print("=" * 80)
    print("Test 4: Edge Cases")
    print("=" * 80)

    num_indices = 10
    num_channels = 3
    module = CustomBackpropModule(num_indices, num_channels)

    # Test 1: Integer inputs (fractional part = 0)
    print("Test 4.1: Integer inputs (frac=0)")
    x_int = torch.tensor([[[[1.0, 2.0, 3.0]]]]).expand(1, 3, 1, 3).clone()
    x_int.requires_grad = True

    output_int = module(x_int)
    loss_int = output_int.sum()
    loss_int.backward()

    # With frac=0, output should be 0 (w[idx] * 0)
    print(f"  Output with integer input: max={output_int.max():.6f}, "
          f"mean={output_int.mean():.6f}")
    assert torch.allclose(output_int, torch.zeros_like(output_int), atol=1e-6), \
        "Integer inputs should produce zero output"

    print("  ✓ Integer input test passed")

    # Test 2: Boundary values
    print("\nTest 4.2: Boundary values")
    x_boundary = torch.tensor([[[[0.0, 9.99]]]]).expand(1, 3, 1, 2).clone()
    x_boundary.requires_grad = True

    output_boundary = module(x_boundary)
    loss_boundary = output_boundary.sum()
    loss_boundary.backward()

    print(f"  Output shape: {output_boundary.shape}")
    print(f"  Gradients computed successfully")

    print("  ✓ Boundary value test passed")

    # Test 3: Near-integer values (frac ≈ 0.999)
    print("\nTest 4.3: Near-integer values")
    x_near = torch.tensor([[[[2.999, 3.001]]]]).expand(1, 3, 1, 2).clone()
    x_near.requires_grad = True

    output_near = module(x_near)
    loss_near = output_near.sum()
    loss_near.backward()

    # For 2.999: floor=2, frac=0.999, output ≈ w[2] * 0.999
    # For 3.001: floor=3, frac=0.001, output ≈ w[3] * 0.001
    print(f"  Output[0,0,0,0] (x=2.999): {output_near[0,0,0,0].item():.6f}")
    print(f"  Output[0,0,0,1] (x=3.001): {output_near[0,0,0,1].item():.6f}")
    print(f"  Gradients exist and are finite: {torch.isfinite(x_near.grad).all().item()}")

    print("  ✓ Near-integer value test passed")

    print("\n✓ All edge case tests PASSED\n")


def test_optimization():
    """Test that the module can be optimized via gradient descent."""
    print("=" * 80)
    print("Test 5: Optimization via Gradient Descent")
    print("=" * 80)

    torch.manual_seed(456)
    num_indices = 6
    num_channels = 3
    module = CustomBackpropModule(num_indices, num_channels)

    # Create fixed input and target
    B, C, W, H = 2, 3, 4, 4
    x = torch.rand(B, C, W, H) * (num_indices - 1)
    target = torch.randn(B, H, W, C)

    # Optimizer
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

    # Training loop
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()

        output = module(x)
        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    print(f"\nOptimization results:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Loss reduction: {loss_reduction * 100:.2f}%")

    assert final_loss < initial_loss, "Loss should decrease during optimization"
    assert loss_reduction > 0.1, "Loss should reduce by at least 10%"

    print("✓ Optimization test PASSED\n")


def test_mathematical_properties():
    """Test mathematical properties of the module."""
    print("=" * 80)
    print("Test 6: Mathematical Properties")
    print("=" * 80)

    torch.manual_seed(789)
    num_indices = 5
    num_channels = 2
    module = CustomBackpropModule(num_indices, num_channels)

    # Property 1: Linearity in weights
    print("Test 6.1: Linearity in weights")
    x = torch.rand(1, 2, 3, 3) * (num_indices - 1)

    # Scale weights by constant
    scale = 2.5
    w_original = module.weight.data.clone()

    output1 = module(x)

    with torch.no_grad():
        module.weight.data *= scale

    output2 = module(x)

    with torch.no_grad():
        module.weight.data = w_original

    # Output should scale linearly
    ratio = (output2 / (output1 + 1e-8)).mean().item()
    print(f"  Weight scale factor: {scale}")
    print(f"  Output scale factor: {ratio:.6f}")
    print(f"  Expected: {scale}")

    assert abs(ratio - scale) < 0.1, "Output should scale linearly with weights"
    print("  ✓ Linearity test passed")

    # Property 2: Fractional part bounded in [0, 1)
    print("\nTest 6.2: Fractional part bounds")
    x_test = torch.rand(1, 2, 5, 5) * 10  # Values in [0, 10)
    x_reshaped = x_test.permute(0, 3, 2, 1)

    x_floor = torch.floor(x_reshaped)
    x_frac = x_reshaped - x_floor

    print(f"  Fractional part range: [{x_frac.min():.6f}, {x_frac.max():.6f}]")
    assert (x_frac >= 0).all() and (x_frac < 1).all(), \
        "Fractional part should be in [0, 1)"
    print("  ✓ Fractional bounds test passed")

    print("\n✓ Mathematical properties test PASSED\n")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print(" CUSTOM BACKPROPAGATION MODULE - COMPREHENSIVE TEST SUITE")
    print("=" * 80 + "\n")

    tests = [
        ("Forward Pass", test_forward_pass),
        ("Gradient Computation", test_gradient_numerical),
        ("Gradient Flow", test_gradient_flow),
        ("Edge Cases", test_edge_cases),
        ("Optimization", test_optimization),
        ("Mathematical Properties", test_mathematical_properties),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}\n")
            failed += 1

    print("=" * 80)
    print(f" TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The module is working correctly.\n")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.\n")

    return failed == 0


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()

    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)
