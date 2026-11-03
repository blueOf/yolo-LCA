# Contrast Weight Parameters Fix for DDP Training

## Problem

When using custom contrastive loss parameters (`contrast_weight`, `contrast_weight_init`, `contrast_weight_final`) in multi-GPU DDP (Distributed Data Parallel) training, the following error occurred:

```
SyntaxError: 'contrast_weight' is not a valid YOLO argument.
'contrast_weight_final' is not a valid YOLO argument.
'contrast_weight_init' is not a valid YOLO argument.
```

This error occurred specifically during DDP training because:
1. These parameters were defined in `ultralytics/cfg/default.yaml`
2. But they were NOT registered in the configuration validation sets in `ultralytics/cfg/__init__.py`
3. During DDP training, worker processes receive these parameters as overrides
4. The `check_dict_alignment()` function validates overrides against base configuration
5. Since the parameters weren't in the validation sets, they were rejected as invalid

## Solution

The fix registers these parameters in the appropriate validation sets in `ultralytics/cfg/__init__.py`:

### 1. `contrast_weight` → Added to `CFG_FLOAT_KEYS`
- **Type**: Float (can be any positive float)
- **Range**: No restriction (can be > 1.0)
- **Usage**: Similar to `box`, `cls`, `dfl` loss weights
- **Example**: `contrast_weight=2.0` ✓

### 2. `contrast_weight_init` → Added to `CFG_FLOAT_KEYS`
- **Type**: Float (can be any positive float)
- **Range**: No restriction (can be > 1.0)
- **Usage**: Initial weight for dynamic weight scheduling
- **Example**: `contrast_weight_init=2.0` ✓

### 3. `contrast_weight_final` → Added to `CFG_FLOAT_KEYS`
- **Type**: Float (can be any positive float)
- **Range**: No restriction (can be > 1.0)
- **Usage**: Final weight for dynamic weight scheduling
- **Example**: `contrast_weight_final=3.0` ✓

## Usage

Now you can use these parameters in both single-GPU and multi-GPU training:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Single GPU training
model.train(
    data="coco.yaml",
    epochs=100,
    device=0,
    contrast_weight=2.0,
    contrast_weight_init=0.5,
    contrast_weight_final=0.8,
)

# Multi-GPU DDP training
model.train(
    data="coco.yaml",
    epochs=100,
    device=[0, 1, 2],  # or device="0,1,2"
    contrast_weight=2.0,
    contrast_weight_init=0.5,
    contrast_weight_final=0.8,
)
```

## Parameter Semantics

### `contrast_weight`
Loss multiplier for the contrastive loss component. Works like other loss weights (`box=7.5`, `cls=0.5`, `dfl=1.5`).

**Formula**: `total_loss = box_loss * box + cls_loss * cls + dfl_loss * dfl + contrast_loss * contrast_weight`

### `contrast_weight_init` and `contrast_weight_final`
These control dynamic weight scheduling during training:
- Training starts with `contrast_weight_init` weight value
- Gradually interpolates to `contrast_weight_final` weight value over training epochs
- When `init == final`, static weighting is used (no dynamic adjustment)
- These are direct weight multipliers, not ratios, so they can be > 1.0

## Validation Behavior

The fix ensures proper validation:

```python
# ✓ Valid usage - All three parameters accept any float value
contrast_weight=5.0          # OK: can be any float
contrast_weight_init=2.0     # OK: can be any float (no range restriction)
contrast_weight_final=3.0    # OK: can be any float (no range restriction)

# All contrast weight parameters are in CFG_FLOAT_KEYS - no range restrictions
```

## Technical Details

### DDP Training Flow
1. **Main Process**: Creates trainer with overrides (including contrast parameters)
2. **Script Generation**: `generate_ddp_file()` captures `vars(trainer.args)` in temporary file
3. **Worker Processes**: Each worker loads temporary file and recreates trainer
4. **Validation**: `get_cfg()` calls `check_dict_alignment()` to validate parameters
5. **Fix**: Parameters are now in validation sets, so they pass validation

### Files Modified
- `ultralytics/cfg/__init__.py`: Added parameters to `CFG_FLOAT_KEYS` and `CFG_FRACTION_KEYS`

### Code Changes
```python
# CFG_FLOAT_KEYS - allows any float value
CFG_FLOAT_KEYS = frozenset({
    "warmup_epochs", "box", "cls", "dfl", "degrees", "shear",
    "time", "workspace", "batch",
    "contrast_weight",        # ← Added
    "contrast_weight_init",   # ← Added (moved from CFG_FRACTION_KEYS)
    "contrast_weight_final",  # ← Added (moved from CFG_FRACTION_KEYS)
})

# CFG_FRACTION_KEYS - enforces 0.0 ≤ value ≤ 1.0
# contrast_weight_init and contrast_weight_final were removed from here
# as they don't need range restrictions
CFG_FRACTION_KEYS = frozenset({
    "dropout", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_momentum", "warmup_bias_lr", "hsv_h", "hsv_s", "hsv_v",
    "translate", "scale", "perspective", "flipud", "fliplr",
    "bgr", "mosaic", "mixup", "cutmix", "copy_paste",
    "conf", "iou", "fraction", "multi_scale",
})
```

## Testing

To verify the fix works:

1. **Single-GPU Test**:
```bash
python train.py --contrast_weight=2.0 --device=0
```

2. **Multi-GPU DDP Test**:
```bash
python train.py --contrast_weight=2.0 --device=0,1,2
```

Both should now work without the `SyntaxError`.

## Related Files
- `ultralytics/cfg/__init__.py` - Configuration validation
- `ultralytics/cfg/default.yaml` - Default parameter values
- `ultralytics/utils/dist.py` - DDP script generation
- `ultralytics/utils/loss.py` - Contrastive loss implementation
- `ultralytics/engine/trainer.py` - Training orchestration

## References
- Default values: `ultralytics/cfg/default.yaml:105-107`
- Validation logic: `ultralytics/cfg/__init__.py:150-194, 327-385, 464-506`
- DDP setup: `ultralytics/utils/dist.py:34-130`
