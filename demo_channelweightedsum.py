"""
Demonstration of ChannelWeightedSum module usage.

This script shows how to use the ChannelWeightedSum module in different contexts.
"""

import torch
from ultralytics.nn.modules import ChannelWeightedSum
from ultralytics.nn.tasks import DetectionModel


def demo_standalone_usage():
    """Demonstrate using ChannelWeightedSum as a standalone module."""
    print("\n" + "="*70)
    print("DEMO 1: Standalone Module Usage")
    print("="*70)

    # Create module for 128 channels
    module = ChannelWeightedSum(c1=128)

    # Create sample input
    batch_size = 4
    channels = 128
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width)

    print(f"\nInput shape: {x.shape}")
    print(f"Module parameters: {sum(p.numel() for p in module.parameters())} (128×128={128*128})")
    print(f"Weight matrix shape: {module.weight.shape}")

    # Forward pass
    output = module(x)
    print(f"Output shape: {output.shape}")
    print(f"✓ Shape preserved: {x.shape == output.shape}")


def demo_yaml_model():
    """Demonstrate using ChannelWeightedSum in a YAML model."""
    print("\n" + "="*70)
    print("DEMO 2: YAML Model Integration")
    print("="*70)

    # Load model with ChannelWeightedSum modules
    model = DetectionModel("test_channelweightedsum_yaml.yaml", ch=3, nc=80, verbose=False)

    # Find all ChannelWeightedSum modules
    cws_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "ChannelWeightedSum":
            cws_modules.append((name, module))

    print(f"\nFound {len(cws_modules)} ChannelWeightedSum module(s) in the model:")
    for name, module in cws_modules:
        print(f"  - {name}: {module.channels} channels, {sum(p.numel() for p in module.parameters())} parameters")

    # Test with sample input
    x = torch.randn(1, 3, 640, 640)
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"✓ Model forward pass successful")
    print(f"Output type: {type(output)}")


def demo_channel_mixing():
    """Demonstrate channel mixing behavior."""
    print("\n" + "="*70)
    print("DEMO 3: Channel Mixing Behavior")
    print("="*70)

    # Create simple 3-channel module
    module = ChannelWeightedSum(c1=3)

    # Set specific weights for demonstration
    with torch.no_grad():
        # Each output channel is a weighted sum of input channels
        module.weight.data = torch.tensor([
            [0.8, 0.1, 0.1],  # Output channel 0: mostly input channel 0
            [0.1, 0.8, 0.1],  # Output channel 1: mostly input channel 1
            [0.1, 0.1, 0.8],  # Output channel 2: mostly input channel 2
        ])

    print("\nWeight matrix:")
    print(module.weight.data)

    # Create input with distinct values per channel
    x = torch.zeros(1, 3, 4, 4)
    x[0, 0, :, :] = 1.0  # Channel 0: all 1s
    x[0, 1, :, :] = 2.0  # Channel 1: all 2s
    x[0, 2, :, :] = 3.0  # Channel 2: all 3s

    print("\nInput:")
    print(f"  Channel 0: all 1.0")
    print(f"  Channel 1: all 2.0")
    print(f"  Channel 2: all 3.0")

    # Forward pass
    output = module(x)

    print("\nOutput (each channel is weighted mix of inputs):")
    print(f"  Channel 0: {output[0, 0, 0, 0].item():.2f} (expected: 0.8*1.0 + 0.1*2.0 + 0.1*3.0 = 1.3)")
    print(f"  Channel 1: {output[0, 1, 0, 0].item():.2f} (expected: 0.1*1.0 + 0.8*2.0 + 0.1*3.0 = 2.0)")
    print(f"  Channel 2: {output[0, 2, 0, 0].item():.2f} (expected: 0.1*1.0 + 0.1*2.0 + 0.8*3.0 = 2.7)")


def demo_training_behavior():
    """Demonstrate gradient flow during training."""
    print("\n" + "="*70)
    print("DEMO 4: Training Behavior (Gradient Flow)")
    print("="*70)

    # Create module
    module = ChannelWeightedSum(c1=64)

    # Create optimizer
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)

    print(f"\nInitial weight statistics:")
    print(f"  Mean: {module.weight.data.mean().item():.6f}")
    print(f"  Std:  {module.weight.data.std().item():.6f}")

    # Simulate training for a few iterations
    for i in range(5):
        # Create random input and target
        x = torch.randn(2, 64, 16, 16)
        target = torch.randn(2, 64, 16, 16)

        # Forward pass
        output = module(x)

        # Compute simple MSE loss
        loss = ((output - target) ** 2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradient
        grad_norm = module.weight.grad.norm().item()

        # Update weights
        optimizer.step()

        if i == 0 or i == 4:
            print(f"\nIteration {i+1}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Gradient norm: {grad_norm:.6f}")

    print(f"\nFinal weight statistics:")
    print(f"  Mean: {module.weight.data.mean().item():.6f}")
    print(f"  Std:  {module.weight.data.std().item():.6f}")
    print(f"✓ Weights updated successfully during training")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ChannelWeightedSum Module Demonstration")
    print("="*70)

    demo_standalone_usage()
    demo_yaml_model()
    demo_channel_mixing()
    demo_training_behavior()

    print("\n" + "="*70)
    print("All Demonstrations Completed Successfully! ✓")
    print("="*70 + "\n")
