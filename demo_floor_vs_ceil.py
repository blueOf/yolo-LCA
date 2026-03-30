#!/usr/bin/env python3
"""
Demonstration of the difference between ceil and floor rounding for bounding boxes.
This script shows why using floor consistently prevents background contamination.
"""

import math


def convert_box_old(x_norm, y_norm, width, height):
    """OLD method: floor for top-left, ceil-1 for bottom-right."""
    x1 = math.floor(x_norm * width)
    y1 = math.floor(y_norm * height)
    x2 = math.ceil(x_norm * width) - 1
    y2 = math.ceil(y_norm * height) - 1
    return x1, y1, x2, y2


def convert_box_new(x_norm, y_norm, width, height):
    """NEW method: floor for all coordinates."""
    x1 = math.floor(x_norm * width)
    y1 = math.floor(y_norm * height)
    x2 = math.floor(x_norm * width)
    y2 = math.floor(y_norm * height)
    return x1, y1, x2, y2


def demo_cases():
    """Demonstrate various test cases."""

    print("=" * 80)
    print("Comparing CEIL vs FLOOR for Bounding Box Coordinate Conversion")
    print("=" * 80)
    print()

    # Test cases: (normalized_coord, feature_map_size, description)
    test_cases = [
        (0.3, 10, "Clean case - 0.3 * 10 = 3.0"),
        (0.35, 10, "Edge case - 0.35 * 10 = 3.5"),
        (0.375, 10, "Edge case - 0.375 * 10 = 3.75"),
        (0.71, 10, "Edge case - 0.71 * 10 = 7.1"),
        (0.625, 16, "Edge case - 0.625 * 16 = 10.0"),
        (0.6875, 16, "Edge case - 0.6875 * 16 = 11.0"),
    ]

    for norm_coord, size, description in test_cases:
        print(f"\n{description}")
        print(f"  Normalized coord: {norm_coord}, Feature map size: {size}")
        print(f"  Actual pixel coord: {norm_coord * size}")

        # Old method (ceil - 1)
        old_val = math.ceil(norm_coord * size) - 1
        # New method (floor)
        new_val = math.floor(norm_coord * size)

        print(f"  OLD (ceil-1): {old_val}")
        print(f"  NEW (floor):  {new_val}")

        if old_val != new_val:
            print(f"  ⚠️  DIFFERENCE: {old_val - new_val} pixel(s)")
            if old_val > new_val:
                print(f"      Old method includes {old_val - new_val} extra pixel(s) beyond boundary!")
        else:
            print(f"  ✓ Same result")

    print("\n" + "=" * 80)
    print("\nExample: Box (0.3, 0.3) to (0.71, 0.71) on 10x10 feature map")
    print("=" * 80)

    # Full box example
    x1_norm, y1_norm = 0.3, 0.3
    x2_norm, y2_norm = 0.71, 0.71
    width, height = 10, 10

    print(f"\nNormalized box: ({x1_norm}, {y1_norm}) to ({x2_norm}, {y2_norm})")
    print(f"Feature map: {width}x{height}")

    # Top-left (both methods use floor, so same)
    x1_old = math.floor(x1_norm * width)
    y1_old = math.floor(y1_norm * height)
    x1_new = math.floor(x1_norm * width)
    y1_new = math.floor(y1_norm * height)

    # Bottom-right (different methods)
    x2_old = math.ceil(x2_norm * width) - 1
    y2_old = math.ceil(y2_norm * height) - 1
    x2_new = math.floor(x2_norm * width)
    y2_new = math.floor(y2_norm * height)

    print(f"\nOLD method (ceil-1):")
    print(f"  Pixel coords: ({x1_old}, {y1_old}) to ({x2_old}, {y2_old})")
    print(f"  Box size: {x2_old - x1_old + 1} x {y2_old - y1_old + 1} pixels")
    print(f"  Range: [{x1_old}:{x2_old + 1}, {y1_old}:{y2_old + 1}]")

    print(f"\nNEW method (floor):")
    print(f"  Pixel coords: ({x1_new}, {y1_new}) to ({x2_new}, {y2_new})")
    print(f"  Box size: {x2_new - x1_new + 1} x {y2_new - y1_new + 1} pixels")
    print(f"  Range: [{x1_new}:{x2_new + 1}, {y1_new}:{y2_new + 1}]")

    if x2_old != x2_new or y2_old != y2_new:
        print(f"\n⚠️  OLD method includes extra pixels at ({x2_old}, {y2_old})")
        print(f"    Actual normalized coordinate: {x2_norm * width} = {x2_norm * width:.1f}")
        print(f"    OLD includes pixel {x2_old}, which represents [{x2_old}, {x2_old + 1})")
        print(f"    Since {x2_norm * width:.1f} < {x2_old + 1}, pixel {x2_old} contains background!")

    print("\n" + "=" * 80)
    print("Conclusion:")
    print("=" * 80)
    print("Using FLOOR consistently ensures we never include pixels beyond the box boundary.")
    print("This prevents background contamination in the foreground feature pool.")
    print("=" * 80)


if __name__ == "__main__":
    demo_cases()
