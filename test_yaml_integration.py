"""Test YAML integration for ChannelWeightedSum module."""

import torch
from ultralytics.nn.tasks import DetectionModel


def test_yaml_integration():
    """Test that ChannelWeightedSum can be loaded from YAML and works correctly."""
    print("Testing YAML integration for ChannelWeightedSum...")
    print("="*60)

    # Load model from YAML
    yaml_path = "test_channelweightedsum_yaml.yaml"

    try:
        # Create model
        model = DetectionModel(yaml_path, ch=3, nc=80, verbose=True)
        print(f"\n✓ Model loaded successfully from {yaml_path}")

        # Print model structure
        print(f"\nModel structure:")
        print(model)

        # Create a test input
        batch_size = 2
        channels = 3
        height = 640
        width = 640

        x = torch.randn(batch_size, channels, height, width)
        print(f"\n✓ Created test input with shape: {x.shape}")

        # Forward pass
        print(f"\nPerforming forward pass...")
        with torch.no_grad():
            output = model(x)

        print(f"✓ Forward pass completed successfully")
        print(f"✓ Output type: {type(output)}")

        if isinstance(output, (list, tuple)):
            print(f"✓ Number of outputs: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"  Output {i} shape: {out.shape}")

        # Test that ChannelWeightedSum modules exist and have correct parameters
        channelweightedsum_modules = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "ChannelWeightedSum":
                channelweightedsum_modules.append((name, module))

        print(f"\n✓ Found {len(channelweightedsum_modules)} ChannelWeightedSum module(s)")

        for name, module in channelweightedsum_modules:
            channels = module.channels
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  Module: {name}")
            print(f"    Channels: {channels}")
            print(f"    Parameters: {param_count} (C×C = {channels}×{channels})")
            print(f"    Weight shape: {module.weight.shape}")

        # Test backward pass (gradient flow)
        print(f"\nTesting gradient flow...")
        x_train = torch.randn(1, 3, 640, 640, requires_grad=True)
        output_train = model(x_train)

        # Simple loss for testing - handle dict, tuple, or list output
        if isinstance(output_train, dict):
            # Extract tensors from dict
            loss = sum(v.sum() if isinstance(v, torch.Tensor) else 0 for v in output_train.values())
        elif isinstance(output_train, (list, tuple)):
            loss = sum(o.sum() if isinstance(o, torch.Tensor) else 0 for o in output_train)
        else:
            loss = output_train.sum()

        loss.backward()

        # Check that ChannelWeightedSum modules have gradients
        for name, module in channelweightedsum_modules:
            assert module.weight.grad is not None, f"No gradient for {name}"
            print(f"✓ Gradient computed for {name}")

        print(f"\n{'='*60}")
        print("YAML Integration Test PASSED! ✓✓✓")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n✗ Error during YAML integration test:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_yaml_integration()
    exit(0 if success else 1)
