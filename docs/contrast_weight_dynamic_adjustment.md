# Contrast Weight 动态调整功能说明

## 概述

此功能将 `contrast_weight` 参数集成到 YAML 配置文件中，并实现了一种动态调整机制，可以在训练过程中逐步改变对比损失与其他损失的权重比例。

## ⚠️ 重要：向后兼容性

**当 `contrast_weight_init` 等于 `contrast_weight_final` 时，系统会自动使用静态权重（旧行为），保证向后兼容。**

- 如果 `init == final`：使用简单的 `contrast_loss * contrast_weight` 计算（与旧版本相同）
- 如果 `init != final`：使用动态调整公式，在训练过程中逐步改变权重比例

这意味着即使设置了新参数，只要 init 和 final 相同，训练损失会和旧版本完全一致。

## 配置参数

在 `ultralytics/cfg/default.yaml` 中添加了以下三个参数：

```yaml
contrast_weight: 0.1       # 对比损失的基础权重（用于对比学习任务）
contrast_weight_init: 0.1  # 初始对比损失比例（contrast / (contrast + other)）
contrast_weight_final: 0.1 # 最终对比损失比例（contrast / (contrast + other)）
```

### 参数说明

1. **contrast_weight** (float, 默认: 0.1)
   - 对比损失的基础权重
   - 在不使用动态调整时，直接用于加权对比损失

2. **contrast_weight_init** (float, 默认: 0.1)
   - 训练开始时的对比损失权重
   - 例如 0.1 表示：对比损失会乘以 0.1

3. **contrast_weight_final** (float, 默认: 0.1)
   - 训练结束时的对比损失权重
   - 例如 0.5 表示：对比损失会乘以 0.5

## 使用示例

### 示例 1：静态权重（不动态调整）

如果 `contrast_weight_init` 和 `contrast_weight_final` 相同，则权重保持不变：

```yaml
contrast_weight: 0.2
contrast_weight_init: 0.3
contrast_weight_final: 0.3  # 与 init 相同，保持恒定
```

### 示例 2：从低到高动态调整

从较小的对比损失权重开始，逐步增加到更高的权重：

```yaml
contrast_weight: 1.0
contrast_weight_init: 0.1   # 开始：对比损失 * 0.1
contrast_weight_final: 0.5  # 结束：对比损失 * 0.5
```

训练过程中的权重变化：
- Epoch 0: 权重 = 0.1
- Epoch 25: 权重 = 0.2
- Epoch 50: 权重 = 0.3
- Epoch 75: 权重 = 0.4
- Epoch 100: 权重 = 0.5

### 示例 3：从高到低动态调整

从较高的对比损失权重开始，逐步降低：

```yaml
contrast_weight: 1.0
contrast_weight_init: 0.5   # 开始：对比损失 * 0.5
contrast_weight_final: 0.1  # 结束：对比损失 * 0.1
```

## 训练命令示例

### 使用默认配置

```bash
yolo detect train data=coco.yaml model=yolov8-contrast.yaml epochs=100
```

### 通过命令行覆盖参数

```bash
yolo detect train data=coco.yaml model=yolov8-contrast.yaml \
    epochs=100 \
    contrast_weight=1.0 \
    contrast_weight_init=0.1 \
    contrast_weight_final=0.5
```

### 通过自定义 YAML 配置

创建 `my_config.yaml`:

```yaml
# 继承默认配置
# 覆盖对比损失参数
contrast_weight: 1.0
contrast_weight_init: 0.15
contrast_weight_final: 0.45
epochs: 150
```

训练命令：

```bash
yolo detect train data=coco.yaml model=yolov8-contrast.yaml cfg=my_config.yaml
```

## 实现细节

### 1. 参数读取

`v8ContrastiveDetectionLoss` 类在初始化时从模型的超参数中读取配置：

```python
self.contrast_weight_base = getattr(self.hyp, 'contrast_weight', contrast_weight)
self.contrast_weight_init = getattr(self.hyp, 'contrast_weight_init', 0.1)
self.contrast_weight_final = getattr(self.hyp, 'contrast_weight_final', 0.1)
```

### 2. 动态权重更新

每个 epoch 结束时，训练器自动调用 `criterion.update()` 方法：

```python
def update(self) -> None:
    """更新对比损失权重"""
    # 只在 init != final 时应用动态调整
    if self.contrast_weight_init == self.contrast_weight_final:
        return  # 不需要动态调整

    self.updates += 1
    total_epochs = max(self.hyp.epochs - 1, 1)
    progress = min(self.updates / total_epochs, 1.0)
    current_weight = self.contrast_weight_init + \
        (self.contrast_weight_final - self.contrast_weight_init) * progress
    self.contrast_weight_dynamic = current_weight
```

### 3. 损失计算

在 `loss()` 方法中应用动态权重：

```python
if hasattr(self, 'contrast_weight_dynamic'):
    # 使用动态调整的权重
    loss[3] = contrast_loss * self.contrast_weight_dynamic
else:
    # 没有动态调整，使用基础权重
    loss[3] = contrast_loss * self.contrast_weight
```

## 公式说明

### 简化的线性插值

动态权重使用简单的线性插值：

```
weight(epoch) = init + (final - init) * (epoch / total_epochs)
weighted_contrast_loss = contrast_loss * weight(epoch)
```

例如：
- `init = 0.1, final = 0.5, total_epochs = 100`
- Epoch 0: `weight = 0.1 + (0.5 - 0.1) * 0 = 0.1`
- Epoch 50: `weight = 0.1 + (0.5 - 0.1) * 0.5 = 0.3`
- Epoch 100: `weight = 0.1 + (0.5 - 0.1) * 1.0 = 0.5`

## 注意事项

1. **参数范围**：`contrast_weight_init` 和 `contrast_weight_final` 可以是任意正数
2. **线性调整**：权重变化遵循线性插值，可以修改 `update()` 方法实现其他调度策略
3. **自动集成**：无需修改训练代码，训练器会自动调用 `update()` 方法
4. **兼容性**：如果不设置这些参数，系统会使用默认值，保持向后兼容
5. **简化实现**：权重直接从 init 插值到 final，不涉及复杂的比例计算

## 监控和调试

在训练过程中，可以通过日志查看对比损失的权重变化：

```python
# 在训练循环中添加日志
if hasattr(model.criterion, 'contrast_weight_dynamic'):
    print(f"Epoch {epoch}: contrast_weight = {model.criterion.contrast_weight_dynamic:.4f}")
```

## 高级用法

### 自定义调度策略

如需非线性调度（如余弦退火），可以修改 `update()` 方法：

```python
import math

def update(self) -> None:
    if self.contrast_weight_init == self.contrast_weight_final:
        return

    self.updates += 1
    total_epochs = max(self.hyp.epochs - 1, 1)
    progress = min(self.updates / total_epochs, 1.0)

    # 余弦退火
    cos_progress = (1 - math.cos(progress * math.pi)) / 2
    current_weight = self.contrast_weight_init + \
        (self.contrast_weight_final - self.contrast_weight_init) * cos_progress

    self.contrast_weight_dynamic = current_weight
```

## 总结

此功能提供了灵活的对比损失权重管理：

1. ✅ 将配置参数移到 YAML 文件中，方便修改
2. ✅ 支持从 init 到 final 的简单线性插值
3. ✅ 自动集成到训练流程，无需修改训练代码
4. ✅ 保持向后兼容，不影响现有功能
5. ✅ 简化实现，去除复杂的比例计算

使用此功能可以更好地调整对比学习损失权重，从而获得更好的训练效果。
