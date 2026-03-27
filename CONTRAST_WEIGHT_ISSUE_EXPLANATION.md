# Contrast Weight Issue: Why Losses are Different Even with Same 0.1 Values

## The Problem

用户报告：即使所有参数都是 0.1，训练时损失却不一样。

Even though all parameters are 0.1:
- `contrast_weight = 0.1`
- `contrast_weight_init = 0.1`
- `contrast_weight_final = 0.1`

The training losses are **different** between the old and new implementations.

## Root Cause Analysis

### Old Implementation (Before Dynamic Adjustment)

```python
def loss(self, preds, batch):
    # ... compute contrast_loss ...
    loss[3] = contrast_loss * self.contrast_weight  # Simply: contrast_loss * 0.1
    return loss * batch_size, loss_detach_full
```

**Behavior:** Contrast loss is simply multiplied by 0.1.

### New Implementation (With Dynamic Adjustment)

```python
def update(self):
    # Called after EACH epoch by trainer
    self.contrast_ratio = 0.1  # Sets this attribute even when init==final

def loss(self, preds, batch):
    # ... compute contrast_loss ...
    if hasattr(self, 'contrast_ratio'):  # TRUE after first epoch!
        # Dynamic weighting formula
        other_loss_sum = det_loss.sum()  # e.g., 10.0
        target_contrast_weight = 0.1 / (1 - 0.1) = 0.111...
        dynamic_weight = 0.111... * (other_loss_sum / contrast_loss)
        loss[3] = contrast_loss * dynamic_weight  # DIFFERENT from contrast_loss * 0.1
    else:
        loss[3] = contrast_loss * self.contrast_weight  # Only for epoch 0
```

**Behavior:**
- **Epoch 0**: Uses simple multiplication `contrast_loss * 0.1` (same as old)
- **Epoch 1+**: Uses dynamic formula (different from old!)

## Mathematical Difference

### Old Method (Static)
```
weighted_contrast = contrast_loss * 0.1
```

### New Method (Dynamic)
```
target_ratio = 0.1 / (1 - 0.1) = 0.111...
dynamic_weight = 0.111... * (other_loss_sum / contrast_loss)
weighted_contrast = contrast_loss * dynamic_weight
```

### Example Calculation

Assume:
- `contrast_loss = 5.0`
- `other_loss_sum = 10.0` (box + cls + dfl already weighted)

**Old method:**
```
weighted_contrast = 5.0 * 0.1 = 0.5
total_loss = 0.5 + 10.0 = 10.5
ratio = 0.5 / 10.5 = 0.0476 (4.76%)
```

**New method (dynamic):**
```
target_weight = 0.1 / 0.9 = 0.111...
dynamic_weight = 0.111... * (10.0 / 5.0) = 0.222...
weighted_contrast = 5.0 * 0.222... = 1.111...
total_loss = 1.111... + 10.0 = 11.111...
ratio = 1.111... / 11.111... = 0.1 (10%)
```

**The dynamic method achieves a 10% ratio, while the old method only achieves 4.76%!**

## Why This Happens

The dynamic adjustment feature was designed to maintain a **target ratio** of contrast loss relative to total loss:

```
ratio = contrast_loss / (contrast_loss + other_losses)
```

When you set `contrast_weight_init = 0.1`, it means "I want contrast loss to be 10% of the total loss", NOT "multiply contrast loss by 0.1".

The problem is that even when `init == final == 0.1`, the code **still applies dynamic adjustment** after the first epoch, because:

1. `update()` is called after epoch 0
2. `update()` sets `self.contrast_ratio = 0.1`
3. Once `contrast_ratio` exists, the loss calculation uses the dynamic formula

## The Solution

We need to **skip dynamic adjustment when init == final**, because there's no need to adjust a constant weight dynamically, and users expect the same behavior as the old static implementation.

### Option 1: Don't Set contrast_ratio When No Change Needed

In `update()`:
```python
def update(self):
    # Only set contrast_ratio if there's actual dynamic adjustment needed
    if self.contrast_weight_init != self.contrast_weight_final:
        self.updates += 1
        # ... calculate and set self.contrast_ratio ...
```

### Option 2: Add a Flag to Disable Dynamic Adjustment

In `__init__()`:
```python
self.use_dynamic_adjustment = (
    self.contrast_weight_init != self.contrast_weight_final
)
```

Then in `loss()`:
```python
if hasattr(self, 'contrast_ratio') and self.use_dynamic_adjustment:
    # Use dynamic weighting
else:
    # Use static weighting
```

### Option 3: Don't Call update() When No Change

Modify the trainer to check before calling update(), but this is less clean.

## Recommended Fix

**Option 1** is the cleanest: modify `update()` to only set `contrast_ratio` when there's actual dynamic adjustment needed.

```python
def update(self) -> None:
    """Update contrast weight based on dynamic adjustment schedule."""
    # Only apply dynamic adjustment if init != final
    if self.contrast_weight_init == self.contrast_weight_final:
        return  # No dynamic adjustment needed

    self.updates += 1
    total_epochs = max(self.hyp.epochs - 1, 1)
    progress = min(self.updates / total_epochs, 1.0)
    current_ratio = (
        self.contrast_weight_init
        + (self.contrast_weight_final - self.contrast_weight_init) * progress
    )
    self.contrast_ratio = current_ratio
```

This ensures:
- When `init == final`: Uses static weighting (like old implementation)
- When `init != final`: Uses dynamic weighting (new feature)

## Backward Compatibility

With this fix:
- Existing users with default config (all 0.1) get the same behavior as before ✓
- New users who set different init/final values get dynamic adjustment ✓
- No breaking changes ✓
