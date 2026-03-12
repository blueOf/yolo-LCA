# RT-DETR ResNet18 Implementation Changes

本文档记录了在 Ultralytics YOLO 框架中实现 RT-DETR ResNet18 所做的所有修改，方便移植到其他 YOLO 仓库。

## 模型参数验证

| 模型 | 参数量 | GFLOPs | 论文参考 |
|------|--------|--------|----------|
| RT-DETR-R18 | 20.16M | 54.4G | ~20M, ~60G |
| RT-DETR-R50 | 42.93M | 130.8G | ~42M, ~136G |
| RT-DETR-R101 | 61.92M | 191.8G | ~76M, ~259G |

> 注：GFLOPs 差异是因为 ultralytics 使用 thop 库计算，部分操作（如 deformable attention、layer norm）的 FLOPs 计算可能与论文方法略有不同。

## 修改文件清单

### 1. `ultralytics/nn/modules/block.py` — 新增 ResNetBasicBlock 类

**位置**：在 `ResNetBlock` 类之后、`ResNetLayer` 类之前

**说明**：ResNet18/34 使用 BasicBlock（两个 3×3 卷积），而 ResNet50/101/152 使用 Bottleneck（1×1 + 3×3 + 1×1 卷积）。

```python
class ResNetBasicBlock(nn.Module):
    """ResNet basic block with two 3x3 convolution layers (for ResNet18/34)."""

    def __init__(self, c1, c2, s=1):
        """Initialize basic block with two 3x3 convolutions."""
        super().__init__()
        self.cv1 = Conv(c1, c2, k=3, s=s, p=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=1, p=1, act=False)
        self.shortcut = (
            nn.Sequential(Conv(c1, c2, k=1, s=s, act=False))
            if s != 1 or c1 != c2
            else nn.Identity()
        )

    def forward(self, x):
        """Forward pass through the ResNet basic block."""
        return F.relu(self.cv2(self.cv1(x)) + self.shortcut(x))
```

### 2. `ultralytics/nn/modules/block.py` — 修改 ResNetLayer 类

**说明**：修改 `ResNetLayer.__init__()` 使其在 `e=1` 时使用 `ResNetBasicBlock`（ResNet18/34），`e=4` 时使用 `ResNetBlock`（ResNet50/101/152）。

**修改前**：
```python
else:
    blocks = [ResNetBlock(c1, c2, s, e=e)]
    blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
    self.layer = nn.Sequential(*blocks)
```

**修改后**：
```python
else:
    if e == 1:  # BasicBlock (ResNet18/34)
        blocks = [ResNetBasicBlock(c1, c2, s)]
        blocks.extend([ResNetBasicBlock(c2, c2, 1) for _ in range(n - 1)])
    else:  # Bottleneck (ResNet50/101/152)
        blocks = [ResNetBlock(c1, c2, s, e=e)]
        blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
    self.layer = nn.Sequential(*blocks)
```

### 3. `ultralytics/nn/modules/block.py` — 修复 RepC3 的 expansion 参数

**说明**：原始 RepC3 的 `cv1` 和 `cv2` 直接使用 `c2` 作为输出通道，导致 `e<1.0` 时与内部 RepConv 通道不匹配。修复为使用 `c_=int(c2*e)` 作为隐藏通道。此修改向后兼容（`e=1.0` 时行为完全不变）。

**修改前**：
```python
self.cv1 = Conv(c1, c2, 1, 1)
self.cv2 = Conv(c1, c2, 1, 1)
```

**修改后**：
```python
self.cv1 = Conv(c1, c_, 1, 1)
self.cv2 = Conv(c1, c_, 1, 1)
```

### 4. `ultralytics/nn/tasks.py` — 修改 parse_model 中 ResNetLayer 的通道计算

**说明**：原始代码硬编码 expansion=4，修改为从 args 中读取 expansion 参数。

**修改前**：
```python
elif m is ResNetLayer:
    c2 = args[1] if args[3] else args[1] * 4
```

**修改后**：
```python
elif m is ResNetLayer:
    c2 = args[1] if args[3] else args[1] * (args[5] if len(args) > 5 else 4)
```

### 5. `ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml` — 新建配置文件

**关键配置说明**：

- **Backbone**：ResNet18，使用 `[2, 2, 2, 2]` 个 BasicBlock，expansion=1
  - 第 6 个参数 `1` 即为 expansion factor，用于区分 BasicBlock(e=1) 和 Bottleneck(e=4)
  - 输出通道：64 → 64 → 128 → 256 → 512

- **Head (Encoder/Neck)**：
  - AIFI 注意力层处理 P5 特征
  - FPN 自顶向下融合 + PAN 自底向上融合
  - RepC3 使用 `e=0.5` 减小隐藏通道（论文中 expansion=0.5），YAML 中写为 `[256, 0.5]`

- **RTDETRDecoder**：
  - `ndl=3`（3 层 decoder，论文中 R18 使用 3 层，R50/R101 使用 6 层）
  - YAML 参数 `[nc, 256, 300, 4, 8, 3]` 对应 `[nc, hd, nq, ndp, nh, ndl]`

## 架构对比

| 组件 | ResNet18 | ResNet50 |
|------|----------|----------|
| Block 类型 | BasicBlock (2×3×3) | Bottleneck (1×1+3×3+1×1) |
| Block 数量 | [2, 2, 2, 2] | [3, 4, 6, 3] |
| Expansion | 1 | 4 |
| 通道 (P3/P4/P5) | 128/256/512 | 512/1024/2048 |
| RepC3 expansion | 0.5 | 1.0 (默认) |
| Decoder 层数 | 3 | 6 |

## 移植步骤

1. 将 `ResNetBasicBlock` 类添加到 `nn/modules/block.py`
2. 修改 `ResNetLayer` 以支持 `e=1` 时使用 BasicBlock
3. 修复 `RepC3` 的 `cv1`/`cv2` 使用 `c_` 而非 `c2`
4. 修改 `parse_model` 中 `ResNetLayer` 的通道计算逻辑
5. 将 `rtdetr-resnet18.yaml` 放入对应的配置目录
6. 如果目标仓库的 `RepC3` 实现不同，确认其是否支持 `e` 参数
