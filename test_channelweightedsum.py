"""Test script for ChannelWeightedSum module."""

import torch
from ultralytics.nn.modules import ChannelWeightedSum


def test_channel_weighted_sum():
    """Test ChannelWeightedSum module with various input shapes."""
    print("Testing ChannelWeightedSum module...")

    # Test cases with different configurations
    test_cases = [
        {"B": 1, "C": 64, "W": 32, "H": 32, "name": "Single batch, 64 channels"},
        {"B": 2, "C": 128, "W": 16, "H": 16, "name": "Batch of 2, 128 channels"},
        {"B": 4, "C": 256, "W": 8, "H": 8, "name": "Batch of 4, 256 channels"},
        {"B": 1, "C": 3, "W": 224, "H": 224, "name": "RGB-like input"},
    ]

    for test_case in test_cases:
        B = test_case["B"]
        C = test_case["C"]
        W = test_case["W"]
        H = test_case["H"]
        name = test_case["name"]

        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"Input shape: (B={B}, C={C}, W={W}, H={H})")
        print(f"{'='*60}")

        # Create module
        module = ChannelWeightedSum(c1=C)

        # Create input tensor
        x = torch.randn(B, C, W, H)

        # Forward pass
        output = module(x)

        # Validate output shape
        expected_shape = (B, C, W, H)
        assert output.shape == expected_shape, (
            f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
        )

        # Check parameter count
        param_count = sum(p.numel() for p in module.parameters())
        expected_param_count = C * C  # C×C weight matrix
        assert param_count == expected_param_count, (
            f"Parameter count mismatch! Expected {expected_param_count}, got {param_count}"
        )

        print(f"✓ Output shape: {output.shape} - CORRECT")
        print(f"✓ Parameter count: {param_count} (C×C = {C}×{C}) - CORRECT")
        print(f"✓ Weight matrix shape: {module.weight.shape} - CORRECT")

        # Test gradient flow
        loss = output.sum()
        loss.backward()

        assert module.weight.grad is not None, "Gradient not computed for weight!"
        print(f"✓ Gradient computed successfully")

        # Test that output is different from input (transformation is happening)
        # This would fail if the module was just passing through the input
        with torch.no_grad():
            identity_module = ChannelWeightedSum(c1=C)
            identity_module.weight.data = torch.eye(C)
            identity_output = identity_module(x)

            # With identity matrix, output should equal input
            assert torch.allclose(identity_output, x, atol=1e-5), (
                "Identity transformation failed!"
            )
            print(f"✓ Identity transformation works correctly")

    print(f"\n{'='*60}")
    print("All tests PASSED! ✓")
    print(f"{'='*60}\n")


def test_channel_weighted_sum_computation():
    """Test the mathematical correctness of ChannelWeightedSum."""
    print("\nTesting mathematical correctness...")
    print(f"{'='*60}")

    # Simple test case for manual verification
    B, C, W, H = 1, 2, 2, 2

    # Create module
    module = ChannelWeightedSum(c1=C)

    # Set known weights for verification
    with torch.no_grad():
        module.weight.data = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

    # Create simple input
    x = torch.ones(B, C, W, H)
    x[0, 0, :, :] = 1.0  # First channel all 1s
    x[0, 1, :, :] = 2.0  # Second channel all 2s

    print(f"Input shape: {x.shape}")
    print(f"Input tensor:\n  Channel 0: all 1.0\n  Channel 1: all 2.0")
    print(f"Weight matrix:\n{module.weight.data}")

    # Forward pass
    output = module(x)

    # Expected output computation:
    # For each spatial location (h, w):
    #   out_channel_0 = 1.0 * 1.0 + 2.0 * 0.5 = 1.0 + 1.0 = 2.0
    #   out_channel_1 = 1.0 * 0.5 + 2.0 * 1.0 = 0.5 + 2.0 = 2.5
    expected_ch0 = 2.0
    expected_ch1 = 2.5

    print(f"\nExpected output:\n  Channel 0: all {expected_ch0}\n  Channel 1: all {expected_ch1}")
    print(f"\nActual output:\n  Channel 0: {output[0, 0, 0, 0].item():.4f}\n  Channel 1: {output[0, 1, 0, 0].item():.4f}")

    assert torch.allclose(output[0, 0, :, :], torch.full((W, H), expected_ch0), atol=1e-5), (
        f"Channel 0 computation incorrect! Expected {expected_ch0}, got {output[0, 0, 0, 0].item()}"
    )
    assert torch.allclose(output[0, 1, :, :], torch.full((W, H), expected_ch1), atol=1e-5), (
        f"Channel 1 computation incorrect! Expected {expected_ch1}, got {output[0, 1, 0, 0].item()}"
    )

    print(f"✓ Mathematical computation is CORRECT")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run tests
    test_channel_weighted_sum()
    test_channel_weighted_sum_computation()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✓✓✓")
    print("="*60 + "\n")
