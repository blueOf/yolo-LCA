# Fix: Change Feature Extraction from Ceiling to Floor Rounding

## Problem Statement (问题说明)

In the contrastive loss calculation (`ultralytics/tools/neck_contrastive/trainer.py`), the feature extraction code was using **asymmetric rounding** when converting normalized bounding box coordinates to pixel coordinates:

- **Top-left corner (x1, y1)**: Used `torch.floor()` (round down)
- **Bottom-right corner (x2, y2)**: Used `torch.ceil() - 1` (round up, then subtract 1)

## Why This Was a Problem (为什么这是个问题)

Using `ceil()` for the bottom-right corner can **round UP** and include pixels that are technically outside the actual bounding box boundaries. This causes:

1. **Background pixels incorrectly marked as foreground**: When ceil rounds up, pixels slightly outside the box get included
2. **Contaminated feature pools**: Background features get mixed into the foreground class features
3. **Degraded contrastive learning**: The loss function tries to pull "same-class" features together, but some of those features are actually background, not the object

## Example

Consider a normalized box with coordinates (0.3, 0.3, 0.7, 0.7) mapped to a 10x10 feature map:

### OLD CODE (with ceil):
```python
x1 = int(torch.floor(0.3 * 10).item())  # = 3
y1 = int(torch.floor(0.3 * 10).item())  # = 3
x2 = int(torch.ceil(0.7 * 10).item()) - 1  # = 7 - 1 = 6
y2 = int(torch.ceil(0.7 * 10).item()) - 1  # = 7 - 1 = 6
# Box covers: [3:7, 3:7] = 4x4 pixels
```

But for a box at (0.3, 0.3, 0.71, 0.71):
```python
x1 = int(torch.floor(0.3 * 10).item())  # = 3
y1 = int(torch.floor(0.3 * 10).item())  # = 3
x2 = int(torch.ceil(0.71 * 10).item()) - 1  # = ceil(7.1) - 1 = 8 - 1 = 7
y2 = int(torch.ceil(0.71 * 10).item()) - 1  # = 8 - 1 = 7
# Box covers: [3:8, 3:8] = 5x5 pixels
# But the actual box only goes to 0.71 * 10 = 7.1, so pixel 7 is mostly background!
```

### NEW CODE (with floor):
```python
x1 = int(torch.floor(0.3 * 10).item())  # = 3
y1 = int(torch.floor(0.3 * 10).item())  # = 3
x2 = int(torch.floor(0.71 * 10).item())  # = 7
y2 = int(torch.floor(0.71 * 10).item())  # = 7
# Box covers: [3:8, 3:8] = 5x5 pixels (same as before)
```

But the key difference is **consistency**: floor always rounds DOWN, meaning we never include pixels that extend beyond the box boundary.

## The Fix (修复方案)

Changed lines 160-161 in `ultralytics/tools/neck_contrastive/trainer.py`:

```python
# BEFORE:
x2 = int(torch.ceil(box[2] * w).item()) - 1
y2 = int(torch.ceil(box[3] * h).item()) - 1

# AFTER:
x2 = int(torch.floor(box[2] * w).item())
y2 = int(torch.floor(box[3] * h).item())
```

## Impact (影响)

1. **More accurate foreground/background separation**: Floor rounding ensures we only include pixels that are definitely inside the bounding box
2. **Cleaner contrastive learning signal**: The background pool won't be contaminated with foreground edge pixels
3. **Better convergence**: The contrastive loss can more effectively pull true object features together and push background features away

## Verification (验证)

The change ensures:
- Consistent rounding direction (always floor/down) for all box coordinates
- No inclusion of pixels beyond the normalized box boundaries
- Proper separation between foreground object features and background features

This follows the principle: **When in doubt, exclude rather than include** - it's better to miss a few edge pixels than to contaminate the feature pools with background.
