# Contrast Weight 动态调整功能说明

## 概述

此功能将 `contrast_weight` 参数集成到 YAML 配置文件中，并实现了一种动态调整机制，可以在训练过程中逐步改变对比损失与其他损失的权重比例。

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
   - 训练开始时的对比损失占比
   - 表示 `contrast_loss / (contrast_loss + other_losses)` 的目标比例
   - 例如 0.1 表示：`0.1 * contrast_loss + 0.9 * (box + cls + dfl)`

3. **contrast_weight_final** (float, 默认: 0.1)
   - 训练结束时的对比损失占比
   - 表示最终期望的损失比例
   - 例如 0.5 表示：`0.5 * contrast_loss + 0.5 * (box + cls + dfl)`

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
contrast_weight_init: 0.1   # 开始：10% contrast, 90% other
contrast_weight_final: 0.5  # 结束：50% contrast, 50% other
```

训练过程中的权重变化：
- Epoch 0: 10% 对比损失
- Epoch 25: 20% 对比损失
- Epoch 50: 30% 对比损失
- Epoch 75: 40% 对比损失
- Epoch 100: 50% 对比损失

### 示例 3：从高到低动态调整

从较高的对比损失权重开始，逐步降低：

```yaml
contrast_weight: 1.0
contrast_weight_init: 0.5   # 开始：50% contrast, 50% other
contrast_weight_final: 0.1  # 结束：10% contrast, 90% other
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
    self.updates += 1
    total_epochs = max(self.hyp.epochs - 1, 1)
    progress = min(self.updates / total_epochs, 1.0)
    current_ratio = self.contrast_weight_init + \
        (self.contrast_weight_final - self.contrast_weight_init) * progress
    self.contrast_ratio = current_ratio
```

### 3. 损失计算

在 `loss()` 方法中应用动态权重：

```python
if hasattr(self, 'contrast_ratio'):
    other_loss_sum = det_loss.sum()  # box + cls + dfl

    # 计算动态权重以达到目标比例
    target_weight = self.contrast_ratio / (1 - self.contrast_ratio)
    dynamic_weight = target_weight * (other_loss_sum / contrast_loss)
    loss[3] = contrast_loss * dynamic_weight
else:
    # 没有动态调整，使用基础权重
    loss[3] = contrast_loss * self.contrast_weight
```

## 公式推导

目标：达到比例 `r = contrast / (contrast + other)`

设：
- `L_c` = 原始对比损失
- `L_o` = 其他损失总和（已加权的 box + cls + dfl）
- `r` = 目标比例（如 0.1 表示 10%）

我们希望：
```
r * weighted_contrast = (1 - r) * L_o
```

因此：
```
weighted_contrast = [(1 - r) / r] * L_o
dynamic_weight = weighted_contrast / L_c = [(1 - r) / r] * (L_o / L_c)
```

验证：
```
final_contrast = L_c * dynamic_weight = [(1 - r) / r] * L_o
total_loss = final_contrast + L_o = [(1 - r) / r] * L_o + L_o
ratio = final_contrast / total_loss = [(1 - r) / r] * L_o / {[(1 - r) / r] * L_o + L_o}
      = [(1 - r) / r] / {[(1 - r) / r] + 1}
      = (1 - r) / [(1 - r) + r]
      = (1 - r) / 1
      = 1 - r  ❌

正确推导：
设 w 为权重因子，我们希望：
```
r = w * L_c / (w * L_c + L_o)
```

解得：
```
r * (w * L_c + L_o) = w * L_c
r * w * L_c + r * L_o = w * L_c
r * L_o = w * L_c * (1 - r)
w = r * L_o / [L_c * (1 - r)]
w = [r / (1 - r)] * (L_o / L_c)
```

## 注意事项

1. **参数范围**：`contrast_weight_init` 和 `contrast_weight_final` 应在 0 到 1 之间
2. **线性调整**：权重变化遵循线性插值，可以修改 `update()` 方法实现其他调度策略
3. **自动集成**：无需修改训练代码，训练器会自动调用 `update()` 方法
4. **兼容性**：如果不设置这些参数，系统会使用默认值，保持向后兼容

## 监控和调试

在训练过程中，可以通过日志查看对比损失的权重变化：

```python
# 在训练循环中添加日志
if hasattr(model.criterion, 'contrast_ratio'):
    print(f"Epoch {epoch}: contrast_ratio = {model.criterion.contrast_ratio:.4f}")
```

## 高级用法

### 自定义调度策略

如需非线性调度（如余弦退火），可以修改 `update()` 方法：

```python
import math

def update(self) -> None:
    self.updates += 1
    total_epochs = max(self.hyp.epochs - 1, 1)
    progress = min(self.updates / total_epochs, 1.0)

    # 余弦退火
    cos_progress = (1 - math.cos(progress * math.pi)) / 2
    current_ratio = self.contrast_weight_init + \
        (self.contrast_weight_final - self.contrast_weight_init) * cos_progress

    self.contrast_ratio = current_ratio
```

## 总结

此功能提供了灵活的对比损失权重管理：

1. ✅ 将配置参数移到 YAML 文件中，方便修改
2. ✅ 支持动态调整对比损失与其他损失的比例
3. ✅ 自动集成到训练流程，无需修改训练代码
4. ✅ 保持向后兼容，不影响现有功能

使用此功能可以更好地平衡对比学习损失与检测损失，从而获得更好的训练效果。
