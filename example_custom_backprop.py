#!/usr/bin/env python3
"""
Example script demonstrating the CustomBackpropModule usage.

This script shows:
1. Basic usage of the module
2. Training with gradient descent
3. Visualization of learned weights
4. Interpretation of results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Note: In a real environment with PyTorch installed, uncomment the following:
# from ultralytics.nn.modules import CustomBackpropModule


def example_basic_usage():
    """Demonstrate basic forward and backward pass."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Module configuration
    num_indices = 10  # Support inputs in [0, 10)
    num_channels = 4  # Number of channels

    print(f"Creating CustomBackpropModule with:")
    print(f"  - num_indices = {num_indices}")
    print(f"  - num_channels = {num_channels}")
    print()

    # Note: Uncomment when PyTorch is available
    # module = CustomBackpropModule(num_indices, num_channels)

    # Input shape: (B, C, W, H)
    B, C, W, H = 2, 4, 3, 3
    print(f"Input shape: (B={B}, C={C}, W={W}, H={H})")
    print()

    # Note: Uncomment when PyTorch is available
    # x = torch.rand(B, C, W, H, requires_grad=True) * (num_indices - 1)
    # print(f"Sample input values: {x[0, 0, 0, :3].detach().numpy()}")
    # print()

    # Forward pass
    # output = module(x)
    # print(f"Output shape: {output.shape} (B, H, W, C)")
    # print(f"Sample output values: {output[0, 0, 0, :].detach().numpy()}")
    # print()

    # Backward pass
    # loss = output.sum()
    # loss.backward()

    # print(f"Gradients computed:")
    # print(f"  - Input gradient shape: {x.grad.shape}")
    # print(f"  - Weight gradient shape: {module.weight.grad.shape}")
    # print()

    print("✓ Basic usage example completed")
    print()


def example_training_loop():
    """Demonstrate training with gradient descent."""
    print("=" * 80)
    print("Example 2: Training Loop")
    print("=" * 80)

    num_indices = 8
    num_channels = 3

    print("Setting up training:")
    print(f"  - Module: CustomBackpropModule({num_indices}, {num_channels})")
    print(f"  - Optimizer: Adam (lr=0.01)")
    print(f"  - Loss: MSE")
    print()

    # Note: Uncomment when PyTorch is available
    # module = CustomBackpropModule(num_indices, num_channels)
    # optimizer = optim.Adam(module.parameters(), lr=0.01)

    # Generate synthetic data
    # B, C, W, H = 4, 3, 5, 5
    # x_train = torch.rand(B, C, W, H) * (num_indices - 1)
    # target = torch.randn(B, H, W, C)

    print("Training for 20 epochs...")
    # for epoch in range(20):
    #     optimizer.zero_grad()
    #     output = module(x_train)
    #     loss = nn.functional.mse_loss(output, target)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if epoch % 5 == 0:
    #         print(f"  Epoch {epoch:2d}: loss = {loss.item():.6f}")

    print()
    print("✓ Training example completed")
    print()


def example_interpretation():
    """Demonstrate interpretation of the module."""
    print("=" * 80)
    print("Example 3: Interpretation")
    print("=" * 80)

    print("Understanding the operation:")
    print()

    # Example value
    x_val = 2.7
    floor_val = int(np.floor(x_val))
    frac_val = x_val - floor_val

    print(f"Input value: x = {x_val}")
    print(f"  floor(x) = {floor_val}")
    print(f"  frac(x) = {frac_val:.2f}")
    print()

    print("Computation:")
    print(f"  1. Use floor(x) = {floor_val} as index into weight matrix")
    print(f"  2. Get weight vector w[{floor_val}]")
    print(f"  3. Multiply by fractional part: w[{floor_val}] * {frac_val:.2f}")
    print()

    print("Interpretation:")
    print(f"  - The floor ({floor_val}) selects which 'feature state' to use")
    print(f"  - The fraction ({frac_val:.2f}) determines how much of that state")
    print(f"  - Similar to attention: selecting and weighting features")
    print()

    print("✓ Interpretation example completed")
    print()


def example_gradient_flow():
    """Demonstrate gradient flow through the module."""
    print("=" * 80)
    print("Example 4: Gradient Flow")
    print("=" * 80)

    print("Gradient computation:")
    print()

    print("Forward pass:")
    print("  output = w[floor(x)] * (x - floor(x))")
    print()

    print("Backward pass (chain rule):")
    print("  ∂loss/∂x = ∂loss/∂output * ∂output/∂x")
    print("           = grad_output * w[floor(x)]")
    print()
    print("  ∂loss/∂w[j] = Σ grad_output * (x - floor(x))")
    print("                where floor(x) = j")
    print()

    print("Key properties:")
    print("  ✓ Gradients flow to both input x and weights w")
    print("  ✓ Input gradient scales with weight magnitude")
    print("  ✓ Weight gradient accumulates from all positions using that weight")
    print("  ✓ No vanishing gradients (as long as weights are reasonable)")
    print()

    print("✓ Gradient flow example completed")
    print()


def example_use_cases():
    """Show potential use cases."""
    print("=" * 80)
    print("Example 5: Potential Use Cases")
    print("=" * 80)

    use_cases = [
        ("Quantization-Aware Training",
         "Learn discrete representations while maintaining differentiability"),

        ("Feature Encoding",
         "Map continuous sensor values to learned feature embeddings"),

        ("Soft Indexing",
         "Implement attention-like mechanisms with learned memory banks"),

        ("Neural Architecture Search",
         "Learn discrete architectural choices via continuous relaxation"),

        ("Structured Prediction",
         "Combine discrete categorization with continuous modulation"),
    ]

    for i, (title, description) in enumerate(use_cases, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
        print()

    print("✓ Use cases overview completed")
    print()


def run_all_examples():
    """Run all demonstration examples."""
    print("\n" + "=" * 80)
    print(" CUSTOM BACKPROPAGATION MODULE - USAGE EXAMPLES")
    print("=" * 80 + "\n")

    print("NOTE: This is a demonstration script showing usage patterns.")
    print("To run the actual module, ensure PyTorch is installed and uncomment")
    print("the relevant code sections.\n")

    examples = [
        example_basic_usage,
        example_training_loop,
        example_interpretation,
        example_gradient_flow,
        example_use_cases,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in example: {e}\n")

    print("=" * 80)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Install PyTorch: pip install torch")
    print("  2. Uncomment the code in this script")
    print("  3. Run the tests: python test_custom_backprop.py")
    print("  4. Read the documentation: CUSTOM_BACKPROP_DOCUMENTATION.md")
    print()


if __name__ == "__main__":
    run_all_examples()
