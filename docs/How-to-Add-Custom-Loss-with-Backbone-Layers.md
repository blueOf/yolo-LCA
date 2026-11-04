# How to Add a Custom Loss Using Backbone Layers in YOLO Framework

This document provides a comprehensive guide on implementing a custom loss function that uses intermediate backbone layer features in the Ultralytics YOLO framework.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Implementation Steps](#implementation-steps)
4. [Complete Example: Neck Contrastive Loss](#complete-example-neck-contrastive-loss)
5. [Best Practices and FAQ](#best-practices-and-faq)

---

## Overview

In the YOLO framework, loss computation typically relies on final model outputs (bounding boxes, class probabilities). However, some advanced training techniques require access to intermediate backbone or neck features, such as:

- **Contrastive Learning**: Pull same-class features together, push different-class features apart
- **Feature Consistency Loss**: Enforce consistency across different scales
- **Knowledge Distillation**: Match student intermediate features to teacher features
- **Regularization Loss**: Apply sparsity or smoothness constraints on feature maps

This guide demonstrates how to implement such custom losses.

---

## Core Concepts

### 1. YOLO Model Structure

YOLO models consist of three main components:

```
Backbone → Neck → Head
   ↓        ↓      ↓
Feature   Multi-  Detection
Extraction Scale   Output
          Fusion
```

Model access in code:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Model structure
model.model           # Outer YOLO wrapper
model.model.model     # Underlying nn.Sequential with all layers
model.model.model[-1] # Last layer (typically Detect head)
model.model.model[5]  # Access layer 5 (e.g., a backbone layer)
```

### 2. Layer Indexing and Connections

Each layer has three important attributes:

- **`m.i`**: Layer index (position in Sequential)
- **`m.f`**: Input source index (`-1` = previous layer, specific number = specified layer)
- **`m.type`**: Layer type string (e.g., `"Conv"`, `"C3k2"`, `"Concat"`)

Example YAML configuration:

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]      # 0: P1/2
  - [-1, 1, Conv, [128, 3, 2]]     # 1: P2/4
  - [-1, 2, C3k2, [256, False]]    # 2
  - [-1, 1, Conv, [256, 3, 2]]     # 3: P3/8
  - [-1, 2, C3k2, [512, False]]    # 4
  - [-1, 1, Conv, [512, 3, 2]]     # 5: P4/16
  - [-1, 2, C3k2, [512]]           # 6
  - [-1, 1, Conv, [1024, 3, 2]]    # 7: P5/32
  - [-1, 2, C3k2, [1024]]          # 8
  - [-1, 1, SPPF, [1024, 5]]       # 9

head:
  - [-1, 1, Upsample, [None, 2]]   # 10
  - [[-1, 6], 1, Concat, [1]]      # 11: Concatenate layer 10 and 6
  # ... more head layers
  - [[16, 19, 22], 1, Detect, [nc]] # 23: Detect head, inputs from layers 16,19,22
```

### 3. Feature Extraction Methods

Three main approaches to access intermediate backbone features:

#### Method 1: Forward Hooks (Recommended)

Use PyTorch's forward hooks to capture intermediate layer outputs:

```python
features = {}

def hook_fn(layer_id):
    def fn(module, input, output):
        features[layer_id] = output
    return fn

# Register hooks to specified layers
handles = []
for layer_id in [3, 5, 8]:  # Layers to extract
    handle = model.model.model[layer_id].register_forward_hook(
        hook_fn(layer_id)
    )
    handles.append(handle)

# Forward pass
output = model(image)

# Now features dict contains outputs from layers 3, 5, 8
# features[3]: [batch, channels, height, width]
```

#### Method 2: Modify Forward Function

Directly save intermediate outputs in model's forward function (not recommended, requires framework modification).

#### Method 3: Using embed Parameter

Use model's embed parameter to get pooled features from specific layers:

```python
# Returns globally average-pooled features from specified layers
embeddings = model.predict(image, embed=[10, 20, 27])
```

Note: embed method applies adaptive_avg_pool2d(1,1), suitable only for global feature vectors.

---

## Implementation Steps

### Step 1: Define Loss Class

Create a loss class inheriting from `torch.nn.Module`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomLoss(nn.Module):
    """Custom loss using features from specified backbone layers."""

    def __init__(self, margin: float = 0.2, weight: float = 1.0):
        """
        Initialize custom loss.

        Args:
            margin: Margin parameter for loss calculation
            weight: Loss weight
        """
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        features: list[torch.Tensor],
        batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            features: Feature list from specified backbone layers,
                     each element shape [B, C, H, W]
            batch: Training batch containing 'img', 'cls', 'bboxes', 'batch_idx'

        Returns:
            Scalar loss tensor
        """
        total_loss = 0.0

        # Compute loss for each scale separately
        for feat in features:
            # Implement your loss computation logic
            # Example: L2 regularization
            loss_i = (feat ** 2).mean()
            total_loss += loss_i

        return total_loss * self.weight
```

### Step 2: Create Feature Extractor

Use forward hooks to extract features from specified layers:

```python
class FeatureExtractor:
    """Extract features from specified layers using forward hooks."""

    def __init__(self, model: torch.nn.Module, feature_layers: list[int]):
        """
        Args:
            model: YOLO model (can be YOLO wrapper or underlying nn.Module)
            feature_layers: List of layer indices to extract, e.g., [3, 5, 8]
        """
        self.model = model
        self.feature_layers = list(feature_layers)
        self._feature_dict = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to specified layers."""
        # Get underlying nn.Sequential model
        modules = getattr(self.model, "model", None)
        if modules is None:
            raise AttributeError("Model must have .model attribute")

        # Handle nested model.model structure (YOLO wrapper case)
        if hasattr(modules, "model"):
            modules = modules.model

        # Register hook for each specified layer
        for layer_idx in self.feature_layers:
            handle = modules[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        """Create hook function to capture output from specified layer."""
        def hook_fn(module, input, output):
            self._feature_dict[layer_idx] = output
        return hook_fn

    def clear(self):
        """Clear captured features (call before each batch)."""
        self._feature_dict.clear()

    def get_features(self) -> list[torch.Tensor]:
        """Get features in order of feature_layers."""
        features = []
        for layer_idx in self.feature_layers:
            if layer_idx not in self._feature_dict:
                raise RuntimeError(f"Layer {layer_idx} feature not captured")
            features.append(self._feature_dict[layer_idx])
        return features

    def close(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
```

### Step 3: Create Configuration Class

Define loss configuration parameters using `dataclass`:

```python
from dataclasses import dataclass

@dataclass
class CustomLossConfig:
    """Configuration parameters for custom loss."""
    feature_layers: list[int]        # Layer indices to extract, e.g., [3, 5, 8]
    loss_weight: float = 0.1         # Loss weight
    margin: float = 0.2              # Loss hyperparameter
    use_detection_loss: bool = True  # Whether to use detection loss
    freeze_head: bool = False        # Whether to freeze detection head
```

### Step 4: Create Trainer

Create a trainer class to integrate custom loss with original detection loss:

```python
class CustomLossTrainer:
    """Trainer that combines detection loss with custom loss."""

    def __init__(self, model: torch.nn.Module, config: CustomLossConfig):
        self.model = model
        self.config = config

        # Get trainable underlying model
        self.train_model = self._resolve_train_model(model)

        # Create feature extractor
        self.extractor = FeatureExtractor(
            self.train_model,
            config.feature_layers
        )

        # Create custom loss
        self.criterion = MyCustomLoss(
            margin=config.margin,
            weight=config.loss_weight
        )

        # Optional: freeze detection head
        if config.freeze_head:
            self._freeze_head()

    @staticmethod
    def _resolve_train_model(model):
        """Resolve YOLO wrapper to get underlying nn.Module."""
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, torch.nn.Module) and hasattr(inner_model, "model"):
            return inner_model
        return model

    def _freeze_head(self):
        """Freeze detection head parameters."""
        modules = self.train_model.model
        if hasattr(modules, "__len__") and len(modules) > 0:
            head = modules[-1]
            for p in head.parameters():
                p.requires_grad_(False)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> dict[str, torch.Tensor]:
        """
        Execute one training step.

        Args:
            batch: Dict containing 'img', 'cls', 'bboxes', 'batch_idx'
            optimizer: Optimizer

        Returns:
            Dict containing various losses
        """
        self.train_model.train()
        optimizer.zero_grad()
        self.extractor.clear()

        # Compute detection loss (optional)
        det_loss = None
        if self.config.use_detection_loss:
            det_loss, det_items = self.train_model(batch)
            if not isinstance(det_loss, torch.Tensor):
                raise TypeError("Detection loss must be a tensor")
            det_loss = det_loss.sum() if det_loss.ndim != 0 else det_loss
        else:
            # Only forward pass to trigger hooks
            _ = self.train_model(batch["img"])
            det_loss = torch.tensor(0.0, device=batch["img"].device)

        # Get extracted features
        features = self.extractor.get_features()

        # Compute custom loss
        custom_loss = self.criterion(features, batch)

        # Combine total loss
        total_loss = det_loss + self.config.loss_weight * custom_loss

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.detach(),
            "det_loss": det_loss.detach(),
            "custom_loss": custom_loss.detach(),
        }

    def close(self):
        """Clean up resources."""
        self.extractor.close()
```

### Step 5: Write Training Script

```python
import os
import torch
from ultralytics import YOLO

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. Load pretrained model
model = YOLO("yolo11n.pt")

# 2. Determine feature layers to extract
# Method A: Manually specify layer indices (based on YAML config)
feature_layers = [3, 5, 8]  # Corresponding to P3/8, P4/16, P5/32 outputs

# Method B: Automatically get Detect head input layers
# feature_layers = list(model.model.model[-1].f)

# 3. Create configuration
config = CustomLossConfig(
    feature_layers=feature_layers,
    loss_weight=0.3,           # Custom loss weight
    use_detection_loss=True,   # Use detection loss
    freeze_head=False          # Don't freeze detection head
)

# 4. Create trainer
trainer = CustomLossTrainer(model=model, config=config)

# 5. Prepare data loader (using standard YOLO dataset)
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset

data_dict = check_det_dataset("coco128.yaml")
train_loader = build_dataloader(
    path=data_dict["train"],
    imgsz=640,
    batch_size=16,
    stride=32,
    hyp=model.model.args,
    augment=True,
    cache=False,
    workers=8,
    shuffle=True,
)[0]

# 6. Setup optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(
    model.model.parameters(),
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=0.0001
)

# 7. Training loop
epochs = 100
print(f"Starting training for {epochs} epochs...")

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")

    epoch_metrics = {
        "total_loss": 0.0,
        "det_loss": 0.0,
        "custom_loss": 0.0
    }
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Execute training step
        metrics = trainer.training_step(batch, optimizer)

        # Accumulate metrics
        epoch_metrics["total_loss"] += metrics["total_loss"].item()
        epoch_metrics["det_loss"] += metrics["det_loss"].item()
        epoch_metrics["custom_loss"] += metrics["custom_loss"].item()
        num_batches += 1

        # Print progress
        if batch_idx % 10 == 0:
            print(
                f"  Batch {batch_idx}/{len(train_loader)} | "
                f"Total: {metrics['total_loss']:.4f} | "
                f"Det: {metrics['det_loss']:.4f} | "
                f"Custom: {metrics['custom_loss']:.4f}"
            )

    # Update learning rate
    scheduler.step()

    # Calculate epoch averages
    for key in epoch_metrics:
        epoch_metrics[key] /= max(num_batches, 1)

    print(
        f"Epoch {epoch+1} completed | "
        f"Avg Total: {epoch_metrics['total_loss']:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        save_path = f"runs/custom_loss/epoch_{epoch+1}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model": model.model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")

# 8. Cleanup
trainer.close()
print("Training completed!")
```

---

## Complete Example: Neck Contrastive Loss

The codebase includes a complete implementation example: **NeckContrastiveLoss**, located at `ultralytics/tools/neck_contrastive/trainer.py`.

### Example Overview

NeckContrastiveLoss aims to:
- **Pull**: Bring features of same-class objects closer together
- **Push**: Push object features away from background features

### Core Implementation

The implementation consists of three main components:

#### 1. NeckFeatureExtractor Class

Located at `ultralytics/tools/neck_contrastive/trainer.py:31-86`:

```python
class NeckFeatureExtractor:
    """Extract feature maps from selected model layers via forward hooks."""

    def __init__(self, model: torch.nn.Module, feature_layers: list[int]):
        self.model = model
        self.feature_layers = list(feature_layers)
        self._feature_dict = {}
        self._handles = []
        self._register()

    def _register(self):
        """Register forward hooks to model layers."""
        modules = getattr(self.model, "model", None)
        if hasattr(modules, "model"):
            modules = modules.model

        for layer in self.feature_layers:
            handle = modules[layer].register_forward_hook(self._hook(layer))
            self._handles.append(handle)

    def _hook(self, layer: int):
        """Create hook function for capturing layer output."""
        def fn(_, __, output):
            self._feature_dict[layer] = output
        return fn

    def get_features(self) -> list[torch.Tensor]:
        """Return features in order of feature_layers."""
        return [self._feature_dict[layer] for layer in self.feature_layers]

    def close(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
```

#### 2. NeckContrastiveLoss Class

Located at `ultralytics/tools/neck_contrastive/trainer.py:88-208`:

```python
class NeckContrastiveLoss(torch.nn.Module):
    """Pull same-class features together, push background features apart."""

    def __init__(
        self,
        sample_per_box: int = 16,
        max_bg_samples_per_image: int = 256,
        max_pos_samples_per_class: int = 256,
        margin: float = 0.2,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sample_per_box = sample_per_box
        self.max_bg_samples_per_image = max_bg_samples_per_image
        self.max_pos_samples_per_class = max_pos_samples_per_class
        self.margin = margin
        self.eps = eps

    def forward(
        self,
        features: list[torch.Tensor],
        batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Steps:
        1. Get bboxes and class labels from batch
        2. Sample feature vectors at corresponding positions in feature maps
        3. Compute similarity of same-class features (pull loss)
        4. Compute similarity of foreground-background features (push loss)
        """
        losses = [self._loss_single_level(feat, batch) for feat in features]
        return torch.stack(losses).mean()

    def _loss_single_level(
        self,
        feat: torch.Tensor,
        batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute contrastive loss for a single scale."""
        # Implementation details in the actual code file
        # Key steps:
        # 1. Convert normalized bboxes to feature map coordinates
        # 2. Sample vectors from bbox regions (foreground)
        # 3. Sample vectors from non-bbox regions (background)
        # 4. Compute pull term: 1 - cosine_similarity(same_class_vectors)
        # 5. Compute push term: relu(cosine_similarity(fg, bg) - margin)
        # 6. Return combined loss
        pass
```

#### 3. NeckContrastiveTrainer Class

Located at `ultralytics/tools/neck_contrastive/trainer.py:211-321`:

```python
class NeckContrastiveTrainer:
    """Training helper combining detection loss with neck contrastive loss."""

    def __init__(self, model: torch.nn.Module, config: NeckContrastiveConfig):
        self.model = model
        self.config = config
        self.train_model = self._resolve_train_model(model)
        self.extractor = NeckFeatureExtractor(
            self.train_model,
            config.feature_layers
        )
        self.criterion = NeckContrastiveLoss(
            sample_per_box=config.sample_per_box,
            max_bg_samples_per_image=config.max_bg_samples_per_image,
            max_pos_samples_per_class=config.max_pos_samples_per_class,
            margin=config.margin,
        )
        if config.freeze_head:
            self._freeze_head()

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> dict[str, torch.Tensor]:
        """Execute one training step with combined losses."""
        self.train_model.train()
        optimizer.zero_grad()
        self.extractor.clear()

        # Compute detection loss
        det_loss = torch.tensor(0.0)
        if self.config.use_detection_loss:
            det_loss, _ = self.train_model(batch)

        # Get extracted features
        features = self.extractor.get_features()

        # Compute custom loss
        custom_loss = self.criterion(features, batch)

        # Combine losses
        total_loss = det_loss + self.config.contrastive_weight * custom_loss

        # Backprop and optimize
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.detach(),
            "det_loss": det_loss.detach(),
            "contrastive_loss": custom_loss.detach(),
        }
```

### Usage Example

Reference from `train_neck_contrastive.py`:

```python
from ultralytics import YOLO
from ultralytics.tools.neck_contrastive import (
    NeckContrastiveTrainer,
    NeckContrastiveYOLOData,
)
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig

# Load model
model = YOLO("yolo11n.pt")

# Get feature layers (Detect head input layers)
feature_layers = list(model.model.model[-1].f)  # Typically [16, 19, 22]

# Create data loader
data = NeckContrastiveYOLOData(
    data="coco128.yaml",
    imgsz=640,
    batch=16,
    workers=8,
)
train_loader = data.build_dataloader(mode="train", stride=32)

# Create trainer
trainer = NeckContrastiveTrainer(
    model=model,
    config=NeckContrastiveConfig(
        feature_layers=feature_layers,
        contrastive_weight=0.3,
    ),
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(
    model.model.parameters(),
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005
)

for epoch in range(100):
    for batch in train_loader:
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Execute training step
        metrics = trainer.training_step(batch, optimizer)

        print(f"Loss: {metrics['total_loss']:.4f}")

trainer.close()
```

---

## Best Practices and FAQ

### Q1: How to determine which layers to use?

**A:** Depends on your loss objective:

1. **Backbone output layers**: Key downsampling points
   - YOLO11/YOLO26: Typically around layers 3, 5, 8 (corresponding to P3/8, P4/16, P5/32)
   - Check YAML config to determine

2. **Neck output layers**: Multi-scale features after fusion
   - Typically input layers to Detect head
   - Get via: `feature_layers = list(model.model.model[-1].f)`

3. **Custom layers**: Based on requirements
   - Use `cheak_output.py` script to test different layer output shapes
   - Recommend choosing semantically meaningful layers (e.g., C3k2 outputs, not intermediate Conv)

### Q2: How to handle features at different scales?

**A:** Several strategies:

1. **Compute separately and average** (recommended):
```python
losses = [compute_loss(feat) for feat in features]
total_loss = torch.stack(losses).mean()
```

2. **Weighted average**:
```python
weights = [1.0, 0.5, 0.25]  # Higher weight for larger scales
losses = [w * compute_loss(feat) for w, feat in zip(weights, features)]
total_loss = sum(losses)
```

3. **Upsample to unified scale**:
```python
target_size = features[0].shape[-2:]
upsampled = [
    F.interpolate(feat, size=target_size, mode='bilinear')
    for feat in features
]
```

### Q3: How to efficiently sample feature vectors?

**A:** For high-resolution feature maps, computing all pixels leads to memory and computation issues. Recommendations:

1. **Random sampling**: Sample fixed number of pixels per box region
2. **Limit total samples**: Set maximum samples per class/per image
3. **Use mini-batches**: Compute similarity matrices in batches

See `NeckContrastiveLoss._sample_rows` method for example code.

### Q4: How to debug if feature extraction works correctly?

**A:** Use the following debug script:

```python
import torch
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.eval()

features = {}

def hook_fn(layer_id):
    def fn(module, input, output):
        features[layer_id] = output
        print(f"Layer {layer_id} output shape: {output.shape}")
    return fn

# Register hooks
for layer_id in [3, 5, 8]:
    model.model.model[layer_id].register_forward_hook(hook_fn(layer_id))

# Test forward pass
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    _ = model.model(dummy_input)

# Check features
for layer_id, feat in features.items():
    print(f"Layer {layer_id}: {feat.shape}, dtype={feat.dtype}")
```

### Q5: How to set custom loss weight?

**A:** Weight setting requires balancing:

- **Detection loss** typically ranges 10-100
- **Custom loss** ranges vary greatly depending on implementation
- **Recommended strategy**:
  1. Train separately to observe value ranges of both losses
  2. Use weight to make both losses similar in magnitude
  3. Typical weight: 0.1-1.0

Example:
```python
# Assuming det_loss ≈ 5.0, custom_loss ≈ 0.5
# Use weight=0.3 to make custom loss contribution ≈ 0.15, similar to det_loss
config = CustomLossConfig(
    loss_weight=0.3,
    use_detection_loss=True
)
```

### Q6: Should I freeze the detection head?

**A:** Depends on training strategy:

- **Freeze detection head (freeze_head=True)**:
  - Suitable for: Training only backbone/neck feature representation
  - Pros: More stable training, prevents detection performance degradation
  - Cons: Cannot improve detection performance

- **Don't freeze (freeze_head=False)**:
  - Suitable for: Want to optimize both detection and feature representation
  - Pros: End-to-end optimization
  - Cons: May require smaller learning rate and more epochs

### Q7: How to integrate with YOLO's native training pipeline?

**A:** Current implementation uses independent training loop. To integrate with YOLO's standard training:

1. **Modify model's `init_criterion` method**:
```python
# In ultralytics/nn/tasks.py DetectionModel class
def init_criterion(self):
    if self.use_custom_loss:
        return CombinedLoss(self)  # Return combined loss
    return v8DetectionLoss(self)
```

2. **Modify `loss` method**:
```python
def loss(self, batch, preds=None):
    if not hasattr(self, "criterion"):
        self.criterion = self.init_criterion()
    if preds is None:
        preds = self.forward(batch["img"])
    return self.criterion(preds, batch)
```

However, this requires modifying framework core code (not recommended). Suggest using independent trainer approach (as shown in this document).

### Q8: Importance of Feature Normalization

**A:** When computing cosine similarity or feature distance, L2 normalization is strongly recommended:

```python
# Normalize to unit vectors
vectors = F.normalize(vectors, dim=1, eps=1e-8)

# Now dot product equals cosine similarity
similarity = vectors @ vectors.T  # Range [-1, 1]
```

Benefits:
- Numerical stability
- Smoother gradients
- More controllable loss magnitude

### Q9: How to view complete model structure?

**A:** Use these methods:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Method 1: Print model
print(model.model)

# Method 2: Iterate through all layers
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: type={layer.type}, from={layer.f}")

# Method 3: Use verbose parameter when loading
from ultralytics.nn.tasks import DetectionModel
model = DetectionModel("yolo11n.yaml", nc=80, verbose=True)
```

### Q10: Memory Optimization Techniques

**A:** Memory optimization when handling high-resolution feature maps:

1. **Limit sampling quantity**:
```python
max_samples_per_box = 16
max_samples_per_class = 256
max_bg_samples = 256
```

2. **Use gradient accumulation**:
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = trainer.compute_loss(batch)
    (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    metrics = trainer.training_step(batch, optimizer)
    loss = metrics["total_loss"]

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Summary

Key steps for implementing custom backbone layer loss:

1. **Define Loss class**: Inherit from `nn.Module`, implement `forward` method
2. **Feature extractor**: Use forward hooks to extract features from specified layers
3. **Trainer wrapper**: Combine detection loss with custom loss
4. **Training loop**: Write standard PyTorch training code
5. **Debug and validate**: Ensure feature extraction is correct, loss values are reasonable

Reference existing implementation:
- **Code location**: `ultralytics/tools/neck_contrastive/trainer.py`
- **Training script**: `train_neck_contrastive.py`
- **Test script**: `test_neck_contrastive.py`

With this documentation, you should be able to implement any custom loss based on backbone intermediate layer features!

---

## Quick Reference: Layer Index Cheat Sheet

### YOLO11n/YOLO26 Typical Layer Structure

```
Backbone Layers:
  Layer 0:  Conv [64, 3, 2]     → P1/2   (320x320)
  Layer 1:  Conv [128, 3, 2]    → P2/4   (160x160)
  Layer 2:  C3k2 [256]
  Layer 3:  Conv [256, 3, 2]    → P3/8   (80x80)   ← Common extraction point
  Layer 4:  C3k2 [512]
  Layer 5:  Conv [512, 3, 2]    → P4/16  (40x40)   ← Common extraction point
  Layer 6:  C3k2 [512]
  Layer 7:  Conv [1024, 3, 2]   → P5/32  (20x20)
  Layer 8:  C3k2 [1024]                            ← Common extraction point
  Layer 9:  SPPF [1024]
  Layer 10: C2PSA [1024]

Neck Layers:
  Layer 11-15: Upsample + Concat + C3k2 processing
  Layer 16: Output for P3/8   ← Detect head input 1
  Layer 17-18: Downsample + Concat
  Layer 19: Output for P4/16  ← Detect head input 2
  Layer 20-21: Downsample + Concat
  Layer 22: Output for P5/32  ← Detect head input 3

Head Layer:
  Layer 23: Detect head (inputs: [16, 19, 22])
```

### Common Extraction Patterns

```python
# Pattern 1: Backbone outputs (early features)
feature_layers = [3, 5, 8]  # P3, P4, P5 before neck

# Pattern 2: Neck outputs (fused features)
feature_layers = list(model.model.model[-1].f)  # e.g., [16, 19, 22]

# Pattern 3: Mixed backbone and neck
feature_layers = [5, 8, 16, 19]  # Backbone P4/P5 + Neck P3/P4

# Pattern 4: Single high-level feature
feature_layers = [10]  # After SPPF/C2PSA
```
