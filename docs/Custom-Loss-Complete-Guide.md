# YOLO自定义Loss完整指南 / Complete Guide to YOLO Custom Loss

## 架构概览 / Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOLO Model                              │
│                                                                 │
│  Input Image (3, 640, 640)                                     │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────┐      │
│  │ BACKBONE (特征提取 / Feature Extraction)            │      │
│  │                                                      │      │
│  │  Layer 0: Conv [64]      → P1/2  (320×320)         │      │
│  │  Layer 1: Conv [128]     → P2/4  (160×160)         │      │
│  │  Layer 2: C3k2 [256]                                │      │
│  │  Layer 3: Conv [256]     → P3/8  (80×80)   ★      │      │
│  │  Layer 4: C3k2 [512]                                │      │
│  │  Layer 5: Conv [512]     → P4/16 (40×40)   ★      │      │
│  │  Layer 6: C3k2 [512]                                │      │
│  │  Layer 7: Conv [1024]    → P5/32 (20×20)          │      │
│  │  Layer 8: C3k2 [1024]                       ★      │      │
│  │  Layer 9: SPPF [1024]                              │      │
│  │  Layer 10: C2PSA [1024]                            │      │
│  └──────────────────────────────────────────────────────┘      │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────┐      │
│  │ NECK (多尺度特征融合 / Multi-scale Fusion)          │      │
│  │                                                      │      │
│  │  Layer 11-15: Upsample + Concat + C3k2              │      │
│  │  Layer 16: P3/8 output  (80×80)            ★      │      │
│  │  Layer 17-18: Downsample + Concat                   │      │
│  │  Layer 19: P4/16 output (40×40)            ★      │      │
│  │  Layer 20-21: Downsample + Concat                   │      │
│  │  Layer 22: P5/32 output (20×20)            ★      │      │
│  └──────────────────────────────────────────────────────┘      │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────┐      │
│  │ HEAD (检测输出 / Detection Output)                  │      │
│  │                                                      │      │
│  │  Layer 23: Detect [input: 16, 19, 22]               │      │
│  │    Output: Bboxes, Scores, Classes                  │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

标注★的层是常用的特征提取点
Layers marked with ★ are common feature extraction points
```

## 特征提取策略 / Feature Extraction Strategies

### 策略1: Backbone特征提取 / Strategy 1: Backbone Feature Extraction

**适用场景 / Use Cases:**
- 需要底层、语义较少的特征 / Need low-level, less semantic features
- 实现特征金字塔Loss / Implement feature pyramid loss
- 早期特征约束 / Early feature constraints

**推荐层 / Recommended Layers:**
```python
feature_layers = [3, 5, 8]  # P3/8, P4/16, P5/32 backbone outputs
```

**特征特点 / Feature Characteristics:**
- Layer 3 (P3/8):  256 channels, 80×80,  细粒度特征 / Fine-grained
- Layer 5 (P4/16): 512 channels, 40×40,  中等语义 / Medium semantic
- Layer 8 (P5/32): 1024 channels, 20×20, 高语义特征 / High-level semantic

### 策略2: Neck特征提取 / Strategy 2: Neck Feature Extraction

**适用场景 / Use Cases:**
- 需要多尺度融合后的特征 / Need multi-scale fused features
- 实现对比学习 / Implement contrastive learning
- 特征蒸馏 / Feature distillation

**推荐层 / Recommended Layers:**
```python
# 自动获取Detect头的输入层
# Auto-get Detect head input layers
feature_layers = list(model.model.model[-1].f)  # Typically [16, 19, 22]
```

**特征特点 / Feature Characteristics:**
- Layer 16 (P3): 融合后的细粒度特征 / Fused fine-grained features
- Layer 19 (P4): 融合后的中等特征 / Fused medium features
- Layer 22 (P5): 融合后的高层特征 / Fused high-level features

### 策略3: 混合提取 / Strategy 3: Mixed Extraction

**适用场景 / Use Cases:**
- 需要backbone和neck的特征 / Need both backbone and neck features
- 跨阶段特征约束 / Cross-stage feature constraints

```python
feature_layers = [5, 8, 16, 19]  # Mix of backbone and neck
```

## 实现模式对比 / Implementation Pattern Comparison

### 模式A: 独立Loss计算 / Pattern A: Independent Loss Computation

**特点 / Characteristics:**
- Loss独立于检测loss / Independent from detection loss
- 适合特征正则化 / Suitable for feature regularization
- 实现简单 / Simple implementation

**示例 / Example:**
```python
class MyLoss(nn.Module):
    def forward(self, features, batch):
        # 只关注特征本身，不需要标注信息
        # Only focus on features, no annotation needed
        loss = 0.0
        for feat in features:
            # 例如：特征稀疏性约束
            # Example: Feature sparsity constraint
            loss += torch.abs(feat).mean()
        return loss
```

### 模式B: 标注驱动Loss / Pattern B: Annotation-Driven Loss

**特点 / Characteristics:**
- 需要使用bbox和类别标注 / Requires bbox and class annotations
- 适合对比学习、度量学习 / Suitable for contrastive/metric learning
- 实现复杂但效果更好 / Complex but more effective

**示例 / Example:**
```python
class MyLoss(nn.Module):
    def forward(self, features, batch):
        boxes = batch["bboxes"]
        classes = batch["cls"]
        batch_idx = batch["batch_idx"]

        # 在特征图上定位bbox区域
        # Locate bbox regions in feature map
        # 提取同类/异类特征进行对比
        # Extract same-class/different-class features for contrast
        # ...
```

### 模式C: 双模型Loss（蒸馏）/ Pattern C: Dual-Model Loss (Distillation)

**特点 / Characteristics:**
- 需要教师和学生两个模型 / Requires teacher and student models
- 适合模型压缩和知识迁移 / Suitable for model compression and knowledge transfer
- 训练时教师模型冻结 / Teacher model frozen during training

**示例 / Example:**
```python
class DistillationLoss(nn.Module):
    def forward(self, student_features, teacher_features, batch):
        loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 使学生特征接近教师特征
            # Make student features close to teacher features
            loss += F.mse_loss(s_feat, t_feat)
        return loss
```

## 完整工作流程图 / Complete Workflow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 定义Loss类 / Define Loss Class                            │
│    - 继承nn.Module / Inherit nn.Module                        │
│    - 实现forward(features, batch) / Implement forward()       │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 创建特征提取器 / Create Feature Extractor                 │
│    - 使用forward hooks / Use forward hooks                   │
│    - 注册到指定层 / Register to specified layers             │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. 创建训练器 / Create Trainer                               │
│    - 整合特征提取器和Loss / Integrate extractor and loss     │
│    - 实现training_step / Implement training_step             │
└────────────────┬─────────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. 训练循环 / Training Loop                                  │
│                                                               │
│   for epoch in epochs:                                        │
│       for batch in dataloader:                                │
│           ┌──────────────────────────────────────┐           │
│           │ a) 前向传播（触发hooks）              │           │
│           │    Forward pass (trigger hooks)      │           │
│           └──────────────┬───────────────────────┘           │
│                          ↓                                    │
│           ┌──────────────────────────────────────┐           │
│           │ b) 提取特征                          │           │
│           │    Extract features                  │           │
│           └──────────────┬───────────────────────┘           │
│                          ↓                                    │
│           ┌──────────────────────────────────────┐           │
│           │ c) 计算自定义Loss                    │           │
│           │    Compute custom loss               │           │
│           └──────────────┬───────────────────────┘           │
│                          ↓                                    │
│           ┌──────────────────────────────────────┐           │
│           │ d) 组合总Loss                        │           │
│           │    Combine total loss                │           │
│           └──────────────┬───────────────────────┘           │
│                          ↓                                    │
│           ┌──────────────────────────────────────┐           │
│           │ e) 反向传播 + 优化                   │           │
│           │    Backward + Optimize               │           │
│           └──────────────────────────────────────┘           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Loss权重调优指南 / Loss Weight Tuning Guide

### 权重设置原则 / Weight Setting Principles

1. **观察数值范围 / Observe Value Ranges**

```python
# 第一步：先单独打印两个loss的数值
# Step 1: Print values of both losses separately
print(f"Detection Loss: {det_loss.item():.4f}")
print(f"Custom Loss: {custom_loss.item():.4f}")

# 示例输出 / Example output:
# Detection Loss: 5.2341
# Custom Loss: 0.4521

# 目标：使两个loss的贡献相当
# Goal: Make both losses contribute equally
# 如果custom_loss * weight ≈ det_loss，则权重合理
# If custom_loss * weight ≈ det_loss, weight is reasonable
```

2. **逐步调整权重 / Gradually Adjust Weight**

```python
# 开始时使用较小权重 / Start with small weight
initial_weight = 0.1

# 如果custom loss太小，增大权重 / If custom loss too small, increase
# 如果custom loss太大，减小权重 / If custom loss too large, decrease

# 典型范围 / Typical range: 0.1 - 1.0
```

3. **动态权重（高级）/ Dynamic Weight (Advanced)**

```python
# 随训练进度调整权重 / Adjust weight with training progress
def get_dynamic_weight(epoch, max_epochs):
    # 线性增长 / Linear growth
    return 0.1 + 0.9 * (epoch / max_epochs)

# 或余弦调度 / Or cosine schedule
import math
def get_cosine_weight(epoch, max_epochs):
    return 0.1 + 0.9 * (1 - math.cos(math.pi * epoch / max_epochs)) / 2
```

## 常见Loss类型实现模板 / Common Loss Type Templates

### 模板1: 特征一致性Loss / Template 1: Feature Consistency Loss

**目标 / Goal**: 不同尺度的特征保持一致

```python
class FeatureConsistencyLoss(nn.Module):
    def forward(self, features, batch):
        total_loss = 0.0
        # 对所有特征对计算MSE / Compute MSE for all feature pairs
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                # 对齐尺寸 / Align sizes
                f1 = features[i]
                f2 = F.interpolate(features[j], size=f1.shape[-2:])
                # MSE loss
                total_loss += F.mse_loss(f1, f2)
        return total_loss / (len(features) * (len(features)-1) / 2)
```

### 模板2: 对比学习Loss / Template 2: Contrastive Learning Loss

**目标 / Goal**: Pull同类特征，Push异类特征

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, features, batch):
        # 1. 根据bbox提取特征向量
        # Extract feature vectors based on bboxes
        class_vectors = {}  # {class_id: [vectors]}

        # 2. 计算同类相似度（pull）
        # Compute same-class similarity (pull)
        pull_loss = 0.0
        for vectors in class_vectors.values():
            vectors = F.normalize(vectors, dim=1)
            sim = vectors @ vectors.T
            pull_loss += (1.0 - sim).mean()

        # 3. 计算异类相似度（push）
        # Compute different-class similarity (push)
        push_loss = 0.0
        # ... (compare different classes)

        return pull_loss + push_loss
```

### 模板3: 注意力Loss / Template 3: Attention Loss

**目标 / Goal**: 强制模型关注重要区域

```python
class AttentionLoss(nn.Module):
    def forward(self, features, batch):
        boxes = batch["bboxes"]  # Ground truth boxes
        loss = 0.0

        for feat in features:
            # 在特征图上生成注意力权重
            # Generate attention weights on feature map
            attention_map = feat.pow(2).mean(dim=1)  # [B, H, W]

            # 计算bbox区域的注意力应该更高
            # Bbox regions should have higher attention
            # ... (implementation details)

        return loss
```

### 模板4: 正则化Loss / Template 4: Regularization Loss

**目标 / Goal**: 约束特征的分布或性质

```python
class FeatureRegularizationLoss(nn.Module):
    def forward(self, features, batch):
        loss = 0.0

        for feat in features:
            # 稀疏性正则 / Sparsity regularization
            sparsity_loss = torch.abs(feat).mean()

            # 平滑性正则 / Smoothness regularization
            diff_h = torch.abs(feat[:, :, 1:, :] - feat[:, :, :-1, :]).mean()
            diff_w = torch.abs(feat[:, :, :, 1:] - feat[:, :, :, :-1]).mean()
            smoothness_loss = diff_h + diff_w

            loss += sparsity_loss + smoothness_loss

        return loss
```

## 性能优化技巧 / Performance Optimization Tips

### 1. 内存优化 / Memory Optimization

```python
# ❌ 不好：计算所有像素的相似度矩阵
# Bad: Compute similarity matrix for all pixels
all_vectors = feat.reshape(C, -1).T  # [H*W, C] - 可能很大！/ Can be huge!
sim_matrix = all_vectors @ all_vectors.T  # [H*W, H*W] - 内存爆炸！/ Memory explosion!

# ✓ 好：随机采样固定数量的向量
# Good: Randomly sample fixed number of vectors
max_samples = 256
if all_vectors.shape[0] > max_samples:
    idx = torch.randperm(all_vectors.shape[0])[:max_samples]
    sampled_vectors = all_vectors[idx]
```

### 2. 计算优化 / Computation Optimization

```python
# ❌ 不好：逐个计算loss
# Bad: Compute loss one by one
for i in range(len(features)):
    for j in range(len(features)):
        loss += some_loss(features[i], features[j])

# ✓ 好：向量化计算
# Good: Vectorized computation
stacked_features = torch.stack(features)  # [N, B, C, H, W]
# 使用广播机制一次性计算
# Use broadcasting to compute all at once
loss = compute_batch_loss(stacked_features)
```

### 3. 数值稳定性 / Numerical Stability

```python
# ✓ 总是使用eps避免除零
# Always use eps to avoid division by zero
eps = 1e-8
normalized = F.normalize(vectors, dim=1, eps=eps)

# ✓ 使用log_softmax而不是log(softmax)
# Use log_softmax instead of log(softmax)
log_probs = F.log_softmax(logits, dim=1)

# ✓ 裁剪梯度
# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

## 调试检查清单 / Debug Checklist

### 阶段1: 特征提取验证 / Stage 1: Feature Extraction Validation

```python
# ✓ 检查项 / Checklist:
# 1. 层索引是否在有效范围内？/ Are layer indices in valid range?
print(f"Model has {len(model.model.model)} layers")
print(f"Extracting from layers: {feature_layers}")

# 2. Hook是否成功注册？/ Are hooks registered successfully?
print(f"Registered {len(extractor._handles)} hooks")

# 3. 特征是否被捕获？/ Are features captured?
features = extractor.get_features()
print(f"Captured {len(features)} features")

# 4. 特征形状是否正确？/ Are feature shapes correct?
for i, feat in enumerate(features):
    print(f"Feature {i}: {feat.shape}")
```

### 阶段2: Loss计算验证 / Stage 2: Loss Computation Validation

```python
# ✓ 检查项 / Checklist:
# 1. Loss是否为标量？/ Is loss a scalar?
assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"

# 2. Loss是否需要梯度？/ Does loss require gradient?
assert loss.requires_grad, "Loss must require gradient"

# 3. Loss数值是否合理？/ Is loss value reasonable?
assert not torch.isnan(loss), "Loss is NaN"
assert not torch.isinf(loss), "Loss is Inf"
print(f"Loss value: {loss.item():.6f}")

# 4. 梯度是否正常？/ Are gradients normal?
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 1000 or grad_norm != grad_norm:  # Check for large or NaN
            print(f"Warning: {name} gradient norm: {grad_norm}")
```

### 阶段3: 训练效果验证 / Stage 3: Training Effect Validation

```python
# ✓ 检查项 / Checklist:
# 1. Loss是否下降？/ Is loss decreasing?
# 绘制loss曲线 / Plot loss curve

# 2. 是否过拟合？/ Is it overfitting?
# 同时监控训练和验证loss / Monitor both train and val loss

# 3. 检测性能是否提升？/ Is detection performance improved?
# 在验证集上评估mAP / Evaluate mAP on validation set

# 4. 训练是否稳定？/ Is training stable?
# 检查loss方差 / Check loss variance
```

## 代码组织建议 / Code Organization Suggestions

### 方案1: 工具模块方式 / Approach 1: Tool Module

```
ultralytics/
  tools/
    my_custom_loss/
      __init__.py         # 导出主要类 / Export main classes
      loss.py             # Loss类定义 / Loss class definition
      extractor.py        # 特征提取器 / Feature extractor
      trainer.py          # 训练器 / Trainer
      config.py           # 配置类 / Config class
      data.py             # 数据加载器（可选）/ Data loader (optional)
```

### 方案2: 独立脚本方式 / Approach 2: Standalone Script

```
project/
  custom_loss.py          # 所有类定义 / All class definitions
  train_custom.py         # 训练脚本 / Training script
  test_custom.py          # 测试脚本 / Test script
```

## 参考实现对照表 / Reference Implementation Comparison

| 组件 / Component | 位置 / Location | 行数 / Lines | 说明 / Description |
|------------------|----------------|--------------|-------------------|
| Loss类 / Loss Class | `trainer.py` | 88-208 | NeckContrastiveLoss |
| 特征提取器 / Extractor | `trainer.py` | 31-86 | NeckFeatureExtractor |
| 训练器 / Trainer | `trainer.py` | 211-321 | NeckContrastiveTrainer |
| 配置类 / Config | `trainer.py` | 17-29 | NeckContrastiveConfig |
| 训练脚本 / Train Script | `train_neck_contrastive.py` | 1-97 | 完整训练循环 / Full loop |
| 测试脚本 / Test Script | `test_neck_contrastive.py` | - | 单元测试 / Unit tests |

## 示例代码索引 / Example Code Index

### 基础示例 / Basic Examples

1. **特征一致性Loss** / Feature Consistency Loss
   - 文件 / File: `example_custom_loss.py`
   - 功能 / Function: 多尺度特征一致性约束
   - 难度 / Difficulty: ⭐⭐☆☆☆

2. **特征蒸馏Loss** / Feature Distillation Loss
   - 文件 / File: `example_distillation_loss.py`
   - 功能 / Function: 教师-学生模型知识蒸馏
   - 难度 / Difficulty: ⭐⭐⭐☆☆

3. **对比学习Loss** / Contrastive Learning Loss
   - 文件 / File: `ultralytics/tools/neck_contrastive/trainer.py`
   - 功能 / Function: Pull同类，Push异类特征
   - 难度 / Difficulty: ⭐⭐⭐⭐☆

## 快速参考卡片 / Quick Reference Card

### 获取特征层索引 / Get Feature Layer Indices

```python
# Backbone层 / Backbone layers
feature_layers = [3, 5, 8]

# Neck层 / Neck layers
feature_layers = list(model.model.model[-1].f)

# 查看所有层 / View all layers
for i, layer in enumerate(model.model.model):
    print(f"{i}: {layer.type}")
```

### 注册Hook模板 / Register Hook Template

```python
features = {}
handles = []

for layer_id in [3, 5, 8]:
    handle = model.model.model[layer_id].register_forward_hook(
        lambda m, i, o, lid=layer_id: features.update({lid: o})
    )
    handles.append(handle)

# ... training ...

# 清理 / Cleanup
for handle in handles:
    handle.remove()
```

### Loss计算模板 / Loss Computation Template

```python
class MyLoss(nn.Module):
    def forward(self, features, batch):
        # 模式1: 特征级loss / Pattern 1: Feature-level loss
        loss = sum((feat ** 2).mean() for feat in features)

        # 模式2: 基于标注的loss / Pattern 2: Annotation-based loss
        boxes = batch["bboxes"]
        classes = batch["cls"]
        # ... extract region features and compute loss

        return loss
```

### 训练循环模板 / Training Loop Template

```python
for epoch in range(epochs):
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        extractor.clear()

        # Forward
        _ = model(batch["img"])
        features = extractor.get_features()

        # Loss
        loss = criterion(features, batch)

        # Backward
        loss.backward()
        optimizer.step()
```

## 故障排除 / Troubleshooting

### 问题：RuntimeError: "Layer X feature not captured"

**原因 / Cause:**
- 层索引错误 / Wrong layer index
- Hook注册失败 / Hook registration failed

**解决 / Solution:**
```python
# 检查模型层数 / Check model layer count
print(f"Model layers: {len(model.model.model)}")

# 检查hook是否注册 / Check if hooks registered
print(f"Hooks: {len(extractor._handles)}")
```

### 问题：Loss是NaN或Inf / Loss is NaN or Inf

**原因 / Cause:**
- 除零错误 / Division by zero
- 梯度爆炸 / Gradient explosion
- 学习率过大 / Learning rate too high

**解决 / Solution:**
```python
# 1. 使用eps / Use eps
normalized = F.normalize(x, dim=1, eps=1e-8)

# 2. 裁剪梯度 / Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

# 3. 降低学习率 / Lower learning rate
lr = 0.0001
```

### 问题：训练很慢 / Training is slow

**原因 / Cause:**
- 采样数过多 / Too many samples
- 特征图分辨率过高 / Feature map resolution too high

**解决 / Solution:**
```python
# 1. 限制采样数 / Limit sampling
max_samples = 256

# 2. 使用梯度累积 / Use gradient accumulation
accumulation_steps = 4

# 3. 减小batch size / Reduce batch size
batch_size = 8
```

### 问题：Out of Memory (OOM)

**原因 / Cause:**
- 相似度矩阵过大 / Similarity matrix too large
- Batch size过大 / Batch size too large

**解决 / Solution:**
```python
# 1. 分批计算相似度 / Compute similarity in batches
chunk_size = 128
for i in range(0, len(vectors), chunk_size):
    chunk = vectors[i:i+chunk_size]
    sim = chunk @ all_vectors.T

# 2. 使用混合精度 / Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    loss = criterion(features, batch)

# 3. 减小特征图 / Reduce feature map size
feat_small = F.adaptive_avg_pool2d(feat, output_size=(20, 20))
```

## 性能评估指标 / Performance Evaluation Metrics

### 训练阶段监控 / Training Phase Monitoring

```python
metrics = {
    "epoch": epoch,
    "total_loss": total_loss.item(),
    "det_loss": det_loss.item(),
    "custom_loss": custom_loss.item(),
    "lr": scheduler.get_last_lr()[0],
    "gpu_mem": torch.cuda.max_memory_allocated() / 1024**3,  # GB
}

# 记录到tensorboard / Log to tensorboard
# 或保存到CSV / Or save to CSV
```

### 验证阶段评估 / Validation Phase Evaluation

```python
# 1. 检测性能 / Detection performance
results = model.val(data="coco128.yaml")
print(f"mAP50: {results.box.map50}")
print(f"mAP50-95: {results.box.map}")

# 2. 特征质量（可选）/ Feature quality (optional)
# - 特征可分性 / Feature separability
# - 类内方差 vs 类间方差 / Intra-class vs inter-class variance
# - t-SNE可视化 / t-SNE visualization
```

## 完整项目结构示例 / Complete Project Structure Example

```
yolo-custom-loss/
│
├── config/
│   └── custom_loss_config.yaml    # 超参数配置 / Hyperparameter config
│
├── losses/
│   ├── __init__.py
│   ├── base_loss.py               # 基础Loss类 / Base loss class
│   ├── consistency_loss.py        # 一致性Loss / Consistency loss
│   ├── contrastive_loss.py        # 对比Loss / Contrastive loss
│   └── distillation_loss.py       # 蒸馏Loss / Distillation loss
│
├── extractors/
│   ├── __init__.py
│   └── feature_extractor.py       # 特征提取器 / Feature extractor
│
├── trainers/
│   ├── __init__.py
│   └── custom_trainer.py          # 自定义训练器 / Custom trainer
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py           # 特征可视化 / Feature visualization
│   └── metrics.py                 # 评估指标 / Evaluation metrics
│
├── train.py                        # 训练脚本 / Training script
├── test.py                         # 测试脚本 / Test script
├── eval.py                         # 评估脚本 / Evaluation script
└── README.md                       # 说明文档 / Documentation
```

## 关键代码片段速查 / Key Code Snippets Quick Reference

### 获取模型信息 / Get Model Info

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# 总层数 / Total layers
print(len(model.model.model))

# Detect头输入 / Detect head inputs
print(model.model.model[-1].f)

# 查看stride / View strides
print(model.model.model[-1].stride)
```

### 处理Batch数据 / Process Batch Data

```python
# Batch字典结构 / Batch dict structure
batch = {
    "img": Tensor[B, 3, H, W],           # 图像 / Images
    "cls": Tensor[N, 1],                 # 类别 / Classes
    "bboxes": Tensor[N, 4],              # 归一化xywh / Normalized xywh
    "batch_idx": Tensor[N],              # 所属图像索引 / Image index
    "ori_shape": List[Tuple],            # 原始尺寸 / Original size
    "resized_shape": List[Tuple],        # 调整后尺寸 / Resized size
}

# 移动到GPU / Move to GPU
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}
```

### 特征归一化 / Feature Normalization

```python
# L2归一化（推荐）/ L2 normalization (recommended)
normalized = F.normalize(features, dim=1, eps=1e-8)

# 标准化 / Standardization
mean = features.mean(dim=(2, 3), keepdim=True)
std = features.std(dim=(2, 3), keepdim=True)
standardized = (features - mean) / (std + 1e-8)
```

## 学习路径建议 / Learning Path Suggestion

1. **初级 / Beginner**:
   - 运行 `example_custom_loss.py`
   - 理解特征提取和简单loss计算
   - Understand feature extraction and simple loss

2. **中级 / Intermediate**:
   - 运行 `example_distillation_loss.py`
   - 学习双模型训练
   - Learn dual-model training

3. **高级 / Advanced**:
   - 研究 `ultralytics/tools/neck_contrastive/trainer.py`
   - 实现复杂的基于标注的loss
   - Implement complex annotation-based loss

4. **专家 / Expert**:
   - 整合到YOLO标准训练流程
   - 优化性能和内存使用
   - Integrate into standard YOLO training pipeline

## 相关资源 / Related Resources

- **本代码库参考 / This Repo Reference**: `ultralytics/tools/neck_contrastive/`
- **训练脚本 / Training Script**: `train_neck_contrastive.py`
- **测试脚本 / Test Script**: `test_neck_contrastive.py`
- **YOLO文档 / YOLO Docs**: https://docs.ultralytics.com/

---

**最后更新 / Last Updated**: 2026-03-25

**作者 / Author**: Based on Ultralytics YOLO Framework

**许可 / License**: AGPL-3.0
