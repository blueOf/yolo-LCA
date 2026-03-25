# 自定义Loss文档 / Custom Loss Documentation

本目录包含完整的自定义Loss实现文档和示例代码。
This directory contains complete documentation and example code for custom loss implementation.

## 文档列表 / Document List

### 1. 中文文档 / Chinese Documentation
- **文件 / File**: `如何新增自定义Loss计算.md`
- **内容 / Content**: 完整的中文指南，包含概念、实现步骤和最佳实践
- **说明 / Description**: 详细介绍如何在YOLO框架下新增使用backbone指定层计算自定义Loss

### 2. 英文文档 / English Documentation
- **文件 / File**: `How-to-Add-Custom-Loss-with-Backbone-Layers.md`
- **内容 / Content**: Complete English guide with concepts, implementation steps, and best practices
- **说明 / Description**: Comprehensive guide on implementing custom loss using backbone layers in YOLO

### 3. 示例代码 / Example Code
- **文件 / File**: `../example_custom_loss.py`
- **内容 / Content**: 完整的可运行示例，实现特征一致性Loss
- **说明 / Description**: Complete runnable example implementing feature consistency loss

## 现有参考实现 / Existing Reference Implementation

代码库已包含一个完整的参考实现：
The codebase includes a complete reference implementation:

- **实现文件 / Implementation**: `../ultralytics/tools/neck_contrastive/trainer.py`
  - `NeckFeatureExtractor`: 特征提取器 / Feature extractor
  - `NeckContrastiveLoss`: 对比损失类 / Contrastive loss class
  - `NeckContrastiveTrainer`: 训练器 / Trainer

- **训练脚本 / Training Script**: `../train_neck_contrastive.py`
  - 展示完整的训练循环 / Shows complete training loop
  - 包含数据加载、优化器配置、模型保存 / Includes data loading, optimizer setup, model saving

- **测试脚本 / Test Script**: `../test_neck_contrastive.py`
  - 单元测试和集成测试 / Unit tests and integration tests

## 快速开始 / Quick Start

### 1. 运行示例代码 / Run Example Code

```bash
# 确保安装依赖 / Ensure dependencies installed
pip install torch torchvision ultralytics

# 运行示例 / Run example
PYTHONPATH=. python example_custom_loss.py
```

### 2. 使用现有实现 / Use Existing Implementation

```python
from ultralytics import YOLO
from ultralytics.tools.neck_contrastive import NeckContrastiveTrainer
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig

# 加载模型 / Load model
model = YOLO("yolo11n.pt")

# 获取特征层 / Get feature layers
feature_layers = list(model.model.model[-1].f)

# 创建训练器 / Create trainer
trainer = NeckContrastiveTrainer(
    model=model,
    config=NeckContrastiveConfig(
        feature_layers=feature_layers,
        contrastive_weight=0.3,
    ),
)

# 训练循环 / Training loop
# ... (see train_neck_contrastive.py for complete example)
```

## 核心概念 / Core Concepts

### 模型结构 / Model Structure

```
YOLO模型 / YOLO Model:
  model.model           # YOLO包装器 / YOLO wrapper
  model.model.model     # nn.Sequential (所有层 / all layers)
  model.model.model[i]  # 第i层 / Layer i
  model.model.model[-1] # 检测头 / Detect head
```

### 层索引示例 / Layer Index Example

```
Layer 0-10:   Backbone (特征提取 / Feature extraction)
  - Layer 3:  P3/8  (80x80)   ← 常用提取点 / Common extraction point
  - Layer 5:  P4/16 (40x40)   ← 常用提取点 / Common extraction point
  - Layer 8:  P5/32 (20x20)   ← 常用提取点 / Common extraction point

Layer 11-22: Neck (特征融合 / Feature fusion)
  - Layer 16: P3/8 output     ← Detect头输入 / Detect head input
  - Layer 19: P4/16 output    ← Detect头输入 / Detect head input
  - Layer 22: P5/32 output    ← Detect头输入 / Detect head input

Layer 23: Detect (检测头 / Detection head)
```

### 获取特征层索引 / Get Feature Layer Indices

```python
# 方法1：手动指定 / Method 1: Manual specification
feature_layers = [3, 5, 8]

# 方法2：从Detect头自动获取 / Method 2: Auto-get from Detect head
feature_layers = list(model.model.model[-1].f)

# 方法3：查看模型结构 / Method 3: View model structure
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {layer.type}, from={layer.f}")
```

## 实现检查清单 / Implementation Checklist

- [ ] 定义Loss类（继承nn.Module）/ Define Loss class (inherit nn.Module)
- [ ] 实现forward方法 / Implement forward method
- [ ] 创建FeatureExtractor类 / Create FeatureExtractor class
- [ ] 使用forward hooks注册到指定层 / Register forward hooks to specified layers
- [ ] 创建配置类（dataclass）/ Create config class (dataclass)
- [ ] 创建Trainer类 / Create Trainer class
- [ ] 实现training_step方法 / Implement training_step method
- [ ] 编写训练循环 / Write training loop
- [ ] 测试特征提取是否正常 / Test if feature extraction works
- [ ] 验证loss数值范围合理 / Validate loss value range is reasonable
- [ ] 调整loss权重 / Adjust loss weight
- [ ] 完整训练并评估性能 / Full training and evaluate performance

## 常见问题 / Common Issues

### 问题1 / Issue 1: "Layer X feature not captured"

**原因 / Cause**: 层索引超出范围或hook未正确注册
**解决 / Solution**: 检查模型层数，确认索引有效

```python
# 查看总层数 / Check total layer count
print(f"Total layers: {len(model.model.model)}")

# 查看每层类型 / View each layer type
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {layer.type}")
```

### 问题2 / Issue 2: 内存溢出 / Out of Memory

**原因 / Cause**: 特征图太大，样本数过多
**解决 / Solution**: 限制采样数量或使用梯度累积

```python
# 减少batch size / Reduce batch size
batch_size = 8

# 使用梯度累积 / Use gradient accumulation
accumulation_steps = 4
```

### 问题3 / Issue 3: Loss不收敛 / Loss Not Converging

**原因 / Cause**: Loss权重过大或学习率不合适
**解决 / Solution**: 调整权重和学习率

```python
# 降低自定义loss权重 / Lower custom loss weight
config.loss_weight = 0.1

# 降低学习率 / Lower learning rate
lr = 0.001
```

## 参考资料 / References

- **YOLO官方文档 / YOLO Official Docs**: https://docs.ultralytics.com/
- **PyTorch Hooks教程 / PyTorch Hooks Tutorial**: https://pytorch.org/tutorials/
- **本代码库实现 / This Codebase Implementation**:
  - `ultralytics/tools/neck_contrastive/trainer.py`
  - `train_neck_contrastive.py`
  - `cheak_output.py`

## 下一步 / Next Steps

1. 阅读完整文档 / Read complete documentation
2. 理解现有实现 / Understand existing implementation
3. 根据需求修改示例代码 / Modify example code based on requirements
4. 在小数据集上测试 / Test on small dataset
5. 完整训练并评估 / Full training and evaluation

祝训练顺利！/ Happy training!
