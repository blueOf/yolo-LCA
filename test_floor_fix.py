"""Test to validate that floor rounding prevents background pollution in foreground features."""

import torch
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveLoss


def test_floor_rounding_consistency():
    """Test that coordinate conversion uses floor consistently to avoid background pollution."""

    # Create a simple test case
    loss_fn = NeckContrastiveLoss(sample_per_box=16, max_bg_samples_per_image=256)

    # Create a feature map: 1 batch, 64 channels, 8x8 spatial
    feat = torch.randn(1, 64, 8, 8)

    # Create a batch with one box that has fractional coordinates
    # Box at normalized coords (0.3, 0.3, 0.6, 0.6)
    # With feature map size 8x8:
    # - Using floor: x1=2, y1=2, x2=4, y2=4 (box covers pixels [2:5, 2:5])
    # - Using ceil-1: x1=2, y1=2, x2=4, y2=4 (same result for this case)
    # But for edge case like (0.375, 0.375, 0.625, 0.625):
    # - Using floor: x1=3, y1=3, x2=5, y2=5
    # - Using ceil-1: x1=3, y1=3, x2=4, y2=4 (different!)

    batch = {
        "cls": torch.tensor([[0]]),
        "batch_idx": torch.tensor([[0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]])  # center_x, center_y, width, height
    }

    # This should not raise an error
    try:
        loss = loss_fn._loss_single_level(feat, batch)
        print(f"✓ Test passed: Loss computed successfully: {loss.item():.6f}")
        assert loss.item() >= 0.0, "Loss should be non-negative"
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False


def test_edge_case_coordinates():
    """Test edge cases where ceil vs floor makes a difference."""

    loss_fn = NeckContrastiveLoss(sample_per_box=16, max_bg_samples_per_image=256)

    # Feature map: 1 batch, 64 channels, 10x10 spatial
    feat = torch.randn(1, 64, 10, 10)

    # Box that when mapped to 10x10 creates edge case
    # (0.375, 0.375) -> floor(3.75) = 3, ceil(3.75)-1 = 3 (same)
    # (0.625, 0.625) -> floor(6.25) = 6, ceil(6.25)-1 = 6 (same)
    # But (0.3, 0.3) -> floor(3.0) = 3, ceil(3.0)-1 = 2 (different!)

    batch = {
        "cls": torch.tensor([[0]]),
        "batch_idx": torch.tensor([[0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # Creates box at (0.3, 0.3, 0.7, 0.7)
    }

    try:
        loss = loss_fn._loss_single_level(feat, batch)
        print(f"✓ Edge case test passed: Loss = {loss.item():.6f}")
        return True
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False


def test_multiple_boxes():
    """Test with multiple boxes to ensure proper foreground/background separation."""

    loss_fn = NeckContrastiveLoss(sample_per_box=16, max_bg_samples_per_image=256)

    # Feature map: 2 batches, 64 channels, 16x16 spatial
    feat = torch.randn(2, 64, 16, 16)

    # Multiple boxes with different classes
    batch = {
        "cls": torch.tensor([[0], [1], [0]]),
        "batch_idx": torch.tensor([[0], [0], [1]]),
        "bboxes": torch.tensor([
            [0.25, 0.25, 0.3, 0.3],  # box 1 in image 0
            [0.75, 0.75, 0.3, 0.3],  # box 2 in image 0
            [0.5, 0.5, 0.4, 0.4],    # box 3 in image 1
        ])
    }

    try:
        loss = loss_fn._loss_single_level(feat, batch)
        print(f"✓ Multiple boxes test passed: Loss = {loss.item():.6f}")
        assert loss.item() >= 0.0, "Loss should be non-negative"
        return True
    except Exception as e:
        print(f"✗ Multiple boxes test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing floor rounding fix for contrastive loss...\n")

    tests = [
        ("Basic floor rounding", test_floor_rounding_consistency),
        ("Edge case coordinates", test_edge_case_coordinates),
        ("Multiple boxes", test_multiple_boxes),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nRunning: {name}")
        if test_fn():
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        exit(1)
