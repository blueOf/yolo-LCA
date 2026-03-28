# Per-Class Weighted Contrastive Loss

## 概述 (Overview)

对比损失(Contrastive Loss)现在支持基于类别实例数量的内部权重计算，用于平衡类别样本多少之间的损失贡献。

The contrastive loss now supports internal per-class weighting based on class instance counts to balance the loss contribution between classes with many vs. few samples.

## 问题 (Problem)

在训练过程中，数据集中某些类别的实例数量可能远多于其他类别。在标准的对比学习中，这会导致：
- 常见类别主导损失函数
- 稀有类别的特征学习不足
- 类别不平衡影响整体性能

In training, some classes may have far more instances than others in the dataset. In standard contrastive learning, this leads to:
- Common classes dominating the loss function
- Insufficient feature learning for rare classes
- Class imbalance affecting overall performance

## 解决方案 (Solution)

### 1. 逆频率权重 (Inverse Frequency Weighting)

为每个类别计算权重：

```python
# 统计每个类别的实例数量
class_counts = torch.bincount(gt_classes, minlength=num_classes)

# 逆频率权重：稀有类别获得更高权重
class_weights = total_objects / (num_classes * class_counts)

# 归一化使平均权重为1.0
class_weights = class_weights / class_weights.mean()
```

For each class, calculate weights:
- Count instances per class
- Apply inverse frequency: rare classes get higher weights
- Normalize so mean weight = 1.0

### 2. 按对象加权 (Per-Object Weighting)

对比损失计算时，每个对象的损失乘以其类别权重：

```python
# 获取对象的类别权重
object_weight = class_weights[object_class]

# 应用权重
weighted_loss = loss * object_weight
total_contrast_loss += weighted_loss
weighted_object_count += object_weight
```

During contrastive loss computation, each object's loss is multiplied by its class weight:
- Get object's class weight
- Apply weight to loss
- Accumulate weighted loss and count

### 3. 加权归一化 (Weighted Normalization)

最终损失按加权对象数量归一化：

```python
final_loss = total_contrast_loss / max(weighted_object_count, 1)
```

Final loss is normalized by weighted object count (not raw count).

## 实现细节 (Implementation Details)

### 函数签名 (Function Signature)

```python
def compute_contrastive_loss(
    self,
    contrast_feats: list[torch.Tensor],
    fg_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    gt_bboxes: torch.Tensor,
    batch_idx_tensor: torch.Tensor,
    gt_classes: torch.Tensor,  # 新增：类别标签
) -> torch.Tensor:
```

### 关键步骤 (Key Steps)

**步骤1：计算类别权重**
```python
class_counts = torch.bincount(gt_classes.squeeze().long(), minlength=self.nc)
class_counts = torch.clamp(class_counts, min=1)  # 避免除零
total_objects = gt_classes.shape[0]
class_weights = total_objects / (self.nc * class_counts.float())
class_weights = class_weights / class_weights.mean()
```

**步骤2：跟踪对象类别**
```python
gt_object_classes = []  # 记录每个GT对象的类别
for gt_idx in gt_indices_b:
    # ... 提取特征 ...
    gt_object_classes.append(gt_classes_b[gt_idx.item()])

gt_object_classes = torch.stack(gt_object_classes)
object_weights = class_weights[gt_object_classes]
```

**步骤3：应用权重**
```python
for i in range(num_gt):
    # ... 计算InfoNCE loss ...
    loss = F.cross_entropy(logits, labels)

    # 应用类别权重
    object_weight = object_weights[i]
    weighted_loss = loss * object_weight
    total_contrast_loss += weighted_loss
    weighted_object_count += object_weight
```

**步骤4：加权归一化**
```python
return total_contrast_loss / max(weighted_object_count, 1)
```

## 示例 (Example)

### 场景：类别不平衡 (Scenario: Class Imbalance)

假设一个batch有10个对象：
- 类别0（常见）：5个实例
- 类别1（稀有）：2个实例
- 类别2（稀有）：3个实例

Assume a batch has 10 objects:
- Class 0 (common): 5 instances
- Class 1 (rare): 2 instances
- Class 2 (rare): 3 instances

**权重计算 (Weight Calculation)**:

```python
total_objects = 10
num_classes = 80

# 类别计数
class_counts[0] = 5
class_counts[1] = 2
class_counts[2] = 3
# 其他类别 = 0 (被clamp到1)

# 原始权重
raw_weight[0] = 10 / (80 * 5) = 0.025
raw_weight[1] = 10 / (80 * 2) = 0.0625
raw_weight[2] = 10 / (80 * 3) = 0.0417

# 归一化后
mean_weight = mean([0.025, 0.0625, 0.0417, ...])
normalized_weight[0] ≈ 0.5  (较低，因为常见)
normalized_weight[1] ≈ 1.25 (较高，因为稀有)
normalized_weight[2] ≈ 0.83 (中等)
```

**损失计算 (Loss Calculation)**:

```python
# 假设原始损失都是0.5
for obj in objects:
    raw_loss = 0.5
    if obj.class == 0:
        weighted_loss = 0.5 * 0.5 = 0.25   # 降低
    elif obj.class == 1:
        weighted_loss = 0.5 * 1.25 = 0.625 # 增加
    elif obj.class == 2:
        weighted_loss = 0.5 * 0.83 = 0.415 # 中等
```

**结果 (Result)**:
- 稀有类别对总损失的贡献增加
- 常见类别的主导地位被抑制
- 模型更平衡地学习所有类别的特征

Rare classes contribute more to total loss, common classes' dominance is suppressed, and the model learns features for all classes more evenly.

## 效果 (Benefits)

### 1. 平衡训练 (Balanced Training)
- 所有类别获得相对公平的训练关注
- 稀有类别不会被忽视
- 常见类别不会过度主导

All classes receive relatively fair training attention, rare classes are not ignored, and common classes don't over-dominate.

### 2. 改善稀有类别性能 (Improved Rare Class Performance)
- 稀有类别的特征表示质量提高
- 更好的类间区分能力
- 潜在的mAP提升

Better feature representation quality for rare classes, improved inter-class discrimination, potential mAP improvement.

### 3. 自适应调整 (Adaptive Adjustment)
- 每个batch自动计算权重
- 适应不同的数据分布
- 无需手动调参

Weights automatically calculated per batch, adapts to different data distributions, no manual tuning needed.

### 4. 保持尺度 (Scale Preservation)
- 归一化确保损失值保持在合理范围
- 不影响与其他损失组件的平衡
- 动态权重调整仍然有效

Normalization ensures loss values remain in reasonable range, doesn't affect balance with other loss components, dynamic weight adjustment still works.

## 数学原理 (Mathematical Formulation)

### 标准对比损失 (Standard Contrastive Loss)
```
L_contrast = (1/N) * Σ_i L_i
```
其中N是对象数量，L_i是第i个对象的InfoNCE损失。

Where N is the number of objects, L_i is the InfoNCE loss for object i.

### 加权对比损失 (Weighted Contrastive Loss)
```
L_contrast_weighted = (Σ_i w_i * L_i) / (Σ_i w_i)
```
其中：
- w_i = weight[class_i] 是第i个对象的类别权重
- weight[c] = (N / (C * count[c])) / mean_weight
- N是总对象数
- C是类别总数
- count[c]是类别c的实例数

Where:
- w_i = weight[class_i] is the class weight for object i
- weight[c] = (N / (C * count[c])) / mean_weight
- N is total number of objects
- C is total number of classes
- count[c] is instance count for class c

### 权重性质 (Weight Properties)

1. **逆频率 (Inverse Frequency)**:
   - count[c] ↑ → weight[c] ↓
   - 更频繁的类别获得更低权重

2. **归一化 (Normalization)**:
   - mean(weight) = 1.0
   - 保持损失值的整体尺度

3. **平衡性 (Balance)**:
   - Σ_i w_i ≈ N
   - 加权对象数接近原始对象数

## 配置和使用 (Configuration and Usage)

### 自动启用 (Automatically Enabled)

该功能在使用 `v8ContrastiveDetectionLoss` 时自动启用，无需额外配置。

This feature is automatically enabled when using `v8ContrastiveDetectionLoss`, no additional configuration needed.

### 训练命令 (Training Command)

```python
from ultralytics import YOLO

model = YOLO('yolov8-contrast.yaml')
model.train(
    data='dataset.yaml',
    epochs=300,
    batch=16,
    # 对比学习权重仍可调整
    contrast_weight=0.5,
    contrast_weight_init=0.5,
    contrast_weight_final=1.5,
)
```

### 监控效果 (Monitoring)

在训练日志中查看：
- `train/contrast_loss`: 整体对比损失值
- 稀有类别的AP应该有所提升
- 类别间的性能差异应该缩小

Check in training logs:
- `train/contrast_loss`: Overall contrastive loss value
- AP for rare classes should improve
- Performance gap between classes should narrow

## 技术细节 (Technical Details)

### 处理边界情况 (Handling Edge Cases)

1. **空类别 (Empty Classes)**:
   ```python
   class_counts = torch.clamp(class_counts, min=1)
   ```
   避免除零错误

2. **单个对象图像 (Single Object Images)**:
   ```python
   if len(gt_features) < 2:
       continue
   ```
   需要至少2个对象才能计算对比损失

3. **批次大小变化 (Variable Batch Size)**:
   权重在每个batch重新计算，自动适应

### 性能影响 (Performance Impact)

- **计算开销 (Computational Overhead)**: 最小（仅增加bincount和权重查找）
- **内存开销 (Memory Overhead)**: 可忽略（存储class_weights: O(num_classes)）
- **训练速度 (Training Speed)**: 几乎无影响

Minimal computational overhead (only bincount and weight lookup), negligible memory overhead (storing class_weights: O(num_classes)), almost no impact on training speed.

## 兼容性 (Compatibility)

### 向后兼容 (Backward Compatible)
- 保持原有接口不变（除了新增gt_classes参数）
- 不影响标准YOLO检测头
- 仅在使用ContrastDetect时生效

Maintains original interface (except for new gt_classes parameter), doesn't affect standard YOLO detection head, only active when using ContrastDetect.

### 与其他功能兼容 (Compatible With Other Features)
- ✅ 动态权重调整 (Dynamic weight adjustment)
- ✅ 多尺度特征 (Multi-scale features)
- ✅ 不同优化器 (Different optimizers)
- ✅ 数据增强 (Data augmentation)

## 总结 (Summary)

Per-class weighted contrastive loss通过以下方式改进训练：

1. **自动计算**每个类别的权重
2. **平衡**常见和稀有类别的贡献
3. **提升**稀有类别的学习效果
4. **保持**损失值的合理尺度
5. **无需**手动调参

Per-class weighted contrastive loss improves training by:
1. Automatically calculating weights for each class
2. Balancing contribution between common and rare classes
3. Improving learning effectiveness for rare classes
4. Maintaining reasonable loss value scale
5. Requiring no manual tuning

这使得模型在类别不平衡的数据集上能够更好地学习所有类别的特征表示。

This enables the model to better learn feature representations for all classes on imbalanced datasets.

---

**文件位置 (File Location)**: `ultralytics/utils/loss.py`
**类 (Class)**: `v8ContrastiveDetectionLoss`
**方法 (Method)**: `compute_contrastive_loss`
**行号 (Lines)**: 903-1086
