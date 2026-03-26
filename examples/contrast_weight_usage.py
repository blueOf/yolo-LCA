#!/usr/bin/env python3
"""
Example: Using dynamic contrast weight adjustment

This example demonstrates how to use the new contrast weight parameters:
- contrast_weight: Base weight for contrastive loss
- contrast_weight_init: Initial ratio (contrast / total)
- contrast_weight_final: Final ratio (contrast / total)

The weight automatically adjusts during training from init to final over epochs.
"""

# Example 1: Training with default parameters (no dynamic adjustment)
print("=" * 70)
print("Example 1: Training with default parameters")
print("=" * 70)
print("""
# Uses default values from ultralytics/cfg/default.yaml
# contrast_weight: 0.1
# contrast_weight_init: 0.1 (10% contrast, 90% other)
# contrast_weight_final: 0.1 (10% contrast, 90% other)
# No dynamic adjustment since init == final

yolo detect train \\
    data=coco.yaml \\
    model=yolov8-contrast.yaml \\
    epochs=100
""")

# Example 2: Dynamic adjustment from 10% to 50%
print("\n" + "=" * 70)
print("Example 2: Dynamic adjustment from 10% to 50%")
print("=" * 70)
print("""
# Start with 10% contrast loss, gradually increase to 50%
# This emphasizes detection losses early, then balances with contrastive learning

yolo detect train \\
    data=coco.yaml \\
    model=yolov8-contrast.yaml \\
    epochs=100 \\
    contrast_weight=1.0 \\
    contrast_weight_init=0.1 \\
    contrast_weight_final=0.5

Training progression:
- Epoch 0:   10% contrast, 90% other (detection-focused)
- Epoch 25:  20% contrast, 80% other
- Epoch 50:  30% contrast, 70% other
- Epoch 75:  40% contrast, 60% other
- Epoch 100: 50% contrast, 50% other (balanced)
""")

# Example 3: Aggressive contrastive learning
print("\n" + "=" * 70)
print("Example 3: Aggressive contrastive learning")
print("=" * 70)
print("""
# Start with balanced weights, gradually increase contrast emphasis
# Use when you want to strongly emphasize feature learning

yolo detect train \\
    data=coco.yaml \\
    model=yolov8-contrast.yaml \\
    epochs=100 \\
    contrast_weight=2.0 \\
    contrast_weight_init=0.3 \\
    contrast_weight_final=0.7

Training progression:
- Epoch 0:   30% contrast, 70% other (moderate contrast)
- Epoch 50:  50% contrast, 50% other (balanced)
- Epoch 100: 70% contrast, 30% other (contrast-focused)
""")

# Example 4: Using custom config file
print("\n" + "=" * 70)
print("Example 4: Using custom config file")
print("=" * 70)
print("""
Create a custom config file: custom_contrast_config.yaml

---
# Custom configuration for contrast weight adjustment
contrast_weight: 1.5
contrast_weight_init: 0.15
contrast_weight_final: 0.45
epochs: 150
batch: 16
---

Then train with:

yolo detect train \\
    data=coco.yaml \\
    model=yolov8-contrast.yaml \\
    cfg=custom_contrast_config.yaml
""")

# Example 5: Python API usage
print("\n" + "=" * 70)
print("Example 5: Python API usage")
print("=" * 70)
print("""
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Train with dynamic contrast weight adjustment
results = model.train(
    data='coco.yaml',
    model='yolov8-contrast.yaml',
    epochs=100,
    contrast_weight=1.0,
    contrast_weight_init=0.1,
    contrast_weight_final=0.5,
    imgsz=640,
    batch=16
)

# The contrast weight ratio is automatically updated at the end of each epoch
# You can monitor it during training via the trainer's criterion object
""")

# Example 6: Monitoring during training
print("\n" + "=" * 70)
print("Example 6: Monitoring weight changes during training")
print("=" * 70)
print("""
To monitor the current contrast ratio during training, you can check the
criterion object:

from ultralytics import YOLO
from ultralytics.utils import callbacks

def on_train_epoch_end(trainer):
    if hasattr(trainer.model.criterion, 'contrast_ratio'):
        epoch = trainer.epoch
        ratio = trainer.model.criterion.contrast_ratio
        print(f"Epoch {epoch}: Contrast ratio = {ratio:.4f}")

# Register callback
callbacks.default_callbacks['on_train_epoch_end'].append(on_train_epoch_end)

# Train
model = YOLO('yolov8-contrast.yaml')
model.train(data='coco.yaml', epochs=100, contrast_weight_init=0.1, contrast_weight_final=0.5)
""")

# Understanding the parameters
print("\n" + "=" * 70)
print("Understanding the parameters")
print("=" * 70)
print("""
Parameter explanations:

1. contrast_weight (float, default: 0.1)
   - Base weight multiplier for the contrast loss
   - Used as fallback when no dynamic adjustment is configured
   - Higher values give more importance to contrast loss

2. contrast_weight_init (float, default: 0.1)
   - Initial ratio at epoch 0
   - Represents: contrast_loss / (contrast_loss + other_losses)
   - Value of 0.1 means 10% contrast, 90% other losses
   - Value of 0.5 means 50% contrast, 50% other losses

3. contrast_weight_final (float, default: 0.1)
   - Final ratio at the last epoch
   - Same interpretation as contrast_weight_init
   - When init == final, weight stays constant (no dynamic adjustment)
   - When init < final, contrast importance increases over training
   - When init > final, contrast importance decreases over training

Dynamic adjustment formula:
   current_ratio = init + (final - init) * (current_epoch / total_epochs)

The actual weight applied to contrast_loss is calculated to achieve this ratio
relative to the weighted sum of other losses (box + cls + dfl).
""")

# Best practices
print("\n" + "=" * 70)
print("Best practices")
print("=" * 70)
print("""
Recommended configurations:

1. **Conservative approach** (safe default):
   contrast_weight_init: 0.1
   contrast_weight_final: 0.2
   → Gradually increase contrast importance slightly

2. **Balanced approach** (for most cases):
   contrast_weight_init: 0.1
   contrast_weight_final: 0.5
   → Start detection-focused, end balanced

3. **Aggressive approach** (strong feature learning):
   contrast_weight_init: 0.2
   contrast_weight_final: 0.7
   → Emphasize contrastive learning more

4. **Reverse approach** (useful for fine-tuning):
   contrast_weight_init: 0.5
   contrast_weight_final: 0.1
   → Start with strong contrast, then focus on detection

Tips:
- Keep init low (0.1-0.2) to establish good detection first
- Adjust final based on your task: higher for feature quality, lower for speed
- Monitor validation metrics to find optimal ratios for your dataset
- Use longer training (more epochs) when using higher final ratios
""")

print("\n" + "=" * 70)
print("For more details, see: docs/contrast_weight_dynamic_adjustment.md")
print("=" * 70)
