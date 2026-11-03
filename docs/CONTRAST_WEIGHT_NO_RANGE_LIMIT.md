# Contrast Weight Parameters - No Range Restrictions

## Issue

After the initial fix for DDP training, users encountered a new error when trying to use `contrast_weight_init` and `contrast_weight_final` with values > 1.0:

```
ValueError: 'contrast_weight_init=2' is an invalid value. Valid 'contrast_weight_init' values are between 0.0 and 1.0.
```

## Root Cause

The parameters were initially placed in `CFG_FRACTION_KEYS`, which enforces a 0.0-1.0 range restriction. This was based on misleading comments in `default.yaml` that described them as "ratios".

However, examining the actual implementation in `ultralytics/utils/loss.py` reveals:

```python
# Line 1401-1407
current_weight = (
    self.contrast_weight_init
    + (self.contrast_weight_final - self.contrast_weight_init) * progress
)

# Store the current weight - this will be applied directly to the contrast loss
self.contrast_weight_dynamic = current_weight
```

These parameters are used as **direct weight multipliers** in linear interpolation, not as ratios. There's no mathematical requirement for them to be in the 0.0-1.0 range.

## Solution

Move `contrast_weight_init` and `contrast_weight_final` from `CFG_FRACTION_KEYS` to `CFG_FLOAT_KEYS`:

### Before (Incorrect)
```python
CFG_FLOAT_KEYS = frozenset({
    "contrast_weight",  # Only this one had no range restriction
})

CFG_FRACTION_KEYS = frozenset({
    "contrast_weight_init",   # ✗ Incorrectly restricted to 0.0-1.0
    "contrast_weight_final",  # ✗ Incorrectly restricted to 0.0-1.0
})
```

### After (Correct)
```python
CFG_FLOAT_KEYS = frozenset({
    "contrast_weight",        # ✓ No range restriction
    "contrast_weight_init",   # ✓ No range restriction
    "contrast_weight_final",  # ✓ No range restriction
})

CFG_FRACTION_KEYS = frozenset({
    # contrast_weight_init and contrast_weight_final removed
})
```

## Usage Examples

All contrast weight parameters now accept any float value:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Example 1: High initial weight, low final weight (decreasing schedule)
model.train(
    data="coco.yaml",
    epochs=100,
    contrast_weight=2.0,
    contrast_weight_init=3.0,
    contrast_weight_final=1.0,
)

# Example 2: Low initial weight, high final weight (increasing schedule)
model.train(
    data="coco.yaml",
    epochs=100,
    contrast_weight=2.0,
    contrast_weight_init=0.5,
    contrast_weight_final=4.0,
)

# Example 3: Static high weight (no dynamic scheduling)
model.train(
    data="coco.yaml",
    epochs=100,
    contrast_weight=5.0,
    contrast_weight_init=5.0,
    contrast_weight_final=5.0,
)
```

## Dynamic Weight Scheduling

The dynamic scheduling works as follows:

1. **Epoch 0**: Uses `contrast_weight_init` as the multiplier
2. **During Training**: Linear interpolation based on progress
   ```python
   progress = current_epoch / (total_epochs - 1)
   current_weight = init + (final - init) * progress
   ```
3. **Final Epoch**: Uses `contrast_weight_final` as the multiplier

### Example Schedule

For `contrast_weight_init=2.0`, `contrast_weight_final=4.0`, over 100 epochs:
- Epoch 0: weight = 2.0
- Epoch 25: weight = 2.5
- Epoch 50: weight = 3.0
- Epoch 75: weight = 3.5
- Epoch 99: weight = 4.0

## Validation Behavior

```python
# ✓ All valid - no range restrictions
contrast_weight=10.0
contrast_weight_init=2.0
contrast_weight_final=8.0

# ✓ Decreasing schedule is valid
contrast_weight_init=5.0
contrast_weight_final=1.0

# ✓ Fractional values still work
contrast_weight_init=0.1
contrast_weight_final=0.5

# ✓ Negative values are technically accepted (though may not make sense semantically)
contrast_weight_init=-1.0  # No validation error, but use with caution
```

## Configuration File Comments

The comments in `ultralytics/cfg/default.yaml` have been updated:

### Before (Misleading)
```yaml
contrast_weight_init: 0.1 # (float) initial contrast weight ratio (contrast / (contrast + other))
contrast_weight_final: 0.1 # (float) final contrast weight ratio (contrast / (contrast + other))
```

### After (Accurate)
```yaml
contrast_weight_init: 0.1 # (float) initial contrast weight for dynamic scheduling (contrastive learning tasks)
contrast_weight_final: 0.1 # (float) final contrast weight for dynamic scheduling (contrastive learning tasks)
```

## Summary of Changes

1. **ultralytics/cfg/__init__.py**:
   - Added `contrast_weight_init` and `contrast_weight_final` to `CFG_FLOAT_KEYS`
   - Removed them from `CFG_FRACTION_KEYS`

2. **ultralytics/cfg/default.yaml**:
   - Updated comments to remove "ratio" terminology
   - Clarified they are weights for dynamic scheduling

3. **docs/CONTRAST_WEIGHT_PARAMS_FIX.md**:
   - Updated to reflect no range restrictions
   - Clarified usage and validation behavior

## Backward Compatibility

This change is **backward compatible**:
- Old configurations with values in 0.0-1.0 range still work
- New configurations can now use values > 1.0
- No existing functionality is broken

## Related Issues

- Initial DDP error: Parameters not registered → Fixed by adding to CFG_*_KEYS
- Range restriction error: Parameters in wrong validation set → Fixed by moving to CFG_FLOAT_KEYS

Both issues are now resolved, and all three contrast weight parameters work consistently with no range restrictions.
