# COCO 数据集测试说明 / COCO Dataset Testing Guide

## 概述 (Overview)

本目录包含用于在COCO 80类数据集上测试YOLO模型的脚本，特别是用于验证对比学习(Contrastive Learning)修复的效果。

This directory contains scripts for testing YOLO models on the COCO 80-class dataset, specifically to validate the contrastive learning fix.

## 背景 (Background)

### 问题 (Problem)
之前发现对比损失(contrastive loss)在训练早期就停止下降，即使增加权重也不像box loss那样持续下降。

Previously, we found that the contrastive loss plateaus early during training and doesn't continue to decrease like box loss, even when increasing weights.

### 修复 (Fix)
修复了对比损失的归一化方式，从按`batch_size`归一化改为按`total_num_objects`归一化，使其与box loss保持一致。

Fixed the contrastive loss normalization from `batch_size` to `total_num_objects`, making it consistent with box loss.

详细分析请参考 / For detailed analysis, see:
- `LOSS_PLATEAU_ANALYSIS.md` - 技术分析 (Technical analysis)
- `FIX_SUMMARY.md` - 使用指南 (Usage guide)

## 测试脚本 (Test Scripts)

### 1. 快速验证 (Quick Validation)

**文件**: `quick_validation.py`

快速验证对比学习修复是否正常工作（使用8张图像的小数据集，5个epoch）

Quick validation that contrastive learning fix works correctly (using 8-image subset, 5 epochs)

```bash
python quick_validation.py
```

**预期结果**:
- ✅ 训练正常完成，无错误
- ✅ 对比损失持续下降（不再plateau）
- ✅ 总损失稳定下降

**Expected Results**:
- ✅ Training completes without errors
- ✅ Contrastive loss decreases (no longer plateaus)
- ✅ Total loss decreases steadily

### 2. 完整COCO测试 (Full COCO Test)

**文件**: `train_coco_test.py`

使用完整COCO 80类数据集训练300轮，验证修复效果

Train for 300 epochs on full COCO 80-class dataset to validate the fix

```bash
python train_coco_test.py
```

**配置**:
- 模型: YOLOv8n with ContrastDetect head
- 数据集: COCO 80 classes
- 训练轮数: 300 epochs
- 批次大小: 16
- 对比学习权重: 0.5 → 1.5 (动态调整)

**Configuration**:
- Model: YOLOv8n with ContrastDetect head
- Dataset: COCO 80 classes
- Epochs: 300
- Batch size: 16
- Contrastive weights: 0.5 → 1.5 (dynamic adjustment)

**预期结果**:
- 对比损失持续下降300轮
- mAP性能可能优于标准模型
- 训练稳定，不受对象数量影响

**Expected Results**:
- Contrastive loss decreases throughout 300 epochs
- mAP performance potentially better than standard model
- Stable training regardless of object count

### 3. 检测头对比测试 (Detection Head Comparison)

**文件**: `compare_detect_heads.py`

同时训练两个模型进行对比：
1. 标准YOLOv8 (Detect head)
2. YOLOv8对比学习版 (ContrastDetect head)

Train two models simultaneously for comparison:
1. Standard YOLOv8 (Detect head)
2. YOLOv8 with contrastive learning (ContrastDetect head)

```bash
python compare_detect_heads.py
```

**输出目录**:
```
runs/coco_comparison/
├── yolov8n-standard-300ep/     # 标准模型
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.csv
│   └── *.png
└── yolov8n-contrast-300ep/     # 对比学习模型
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    └── *.png
```

## 数据集准备 (Dataset Setup)

### COCO数据集 (COCO Dataset)

COCO数据集会自动下载（如果不存在）。如果您已有COCO数据集，请确保目录结构如下：

The COCO dataset will be downloaded automatically (if not present). If you already have COCO, ensure the directory structure:

```
datasets/coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/
```

### 配置数据集路径 (Configure Dataset Path)

编辑 `ultralytics/cfg/datasets/coco.yaml` 设置正确的路径：

Edit `ultralytics/cfg/datasets/coco.yaml` to set correct path:

```yaml
path: /path/to/your/coco  # 修改为您的COCO数据集路径
train: train2017.txt
val: val2017.txt
```

## 训练配置 (Training Configuration)

### 硬件要求 (Hardware Requirements)

**最低要求 (Minimum)**:
- GPU: 8GB VRAM (NVIDIA)
- RAM: 16GB
- 存储: 50GB (COCO数据集 + 模型权重)

**推荐配置 (Recommended)**:
- GPU: 16GB+ VRAM
- RAM: 32GB
- 存储: 100GB

### 超参数 (Hyperparameters)

修复后的推荐配置 / Recommended configuration after fix:

```python
# Loss weights
box=7.5                    # Box loss weight
cls=0.5                    # Classification loss weight
dfl=1.5                    # DFL loss weight
contrast_weight=0.5        # Base contrastive weight (increased from 0.2)
contrast_weight_init=0.5   # Initial weight
contrast_weight_final=1.5  # Final weight (increased from 0.5)
```

**为什么增加权重？/ Why increase weights?**

修复后，对比损失按对象数量归一化，loss值变小，因此需要增加权重以保持影响力。

After the fix, contrastive loss is normalized by object count, resulting in smaller values, so weights need to be increased to maintain impact.

## 监控训练 (Monitor Training)

### 1. 查看训练日志 (View Training Logs)

```bash
tail -f runs/coco_test/yolov8n-contrast-300ep/train.log
```

### 2. 使用TensorBoard (Use TensorBoard)

```bash
tensorboard --logdir runs/coco_test
```

在浏览器打开: http://localhost:6006

Open in browser: http://localhost:6006

### 3. 检查loss曲线 (Check Loss Curves)

训练完成后查看生成的图表 / After training, check generated plots:

```
runs/coco_test/yolov8n-contrast-300ep/
├── results.png           # 所有指标 (all metrics)
├── F1_curve.png          # F1曲线
├── P_curve.png           # Precision曲线
├── R_curve.png           # Recall曲线
└── confusion_matrix.png  # 混淆矩阵
```

### 4. 分析结果CSV (Analyze Results CSV)

```bash
# 查看训练指标
head -20 runs/coco_test/yolov8n-contrast-300ep/results.csv

# 使用Python分析
import pandas as pd
df = pd.read_csv('runs/coco_test/yolov8n-contrast-300ep/results.csv')
print(df.columns)
print(df[['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'train/contrast_loss']].tail(20))
```

## 验证修复效果 (Validate the Fix)

### 期望的训练行为 (Expected Training Behavior)

**修复前 (Before Fix)**:
```
Epoch 1:   Box=0.850, Contrast=2.150, Total=12.935
Epoch 50:  Box=0.420, Contrast=2.080, Total=9.485   ← Contrast几乎不变
Epoch 100: Box=0.280, Contrast=2.070, Total=8.595   ← 仍然plateau
```

**修复后 (After Fix)**:
```
Epoch 1:   Box=0.850, Contrast=0.850, Total=8.935
Epoch 50:  Box=0.420, Contrast=0.380, Total=5.485   ← Contrast持续下降
Epoch 100: Box=0.280, Contrast=0.220, Total=4.195   ← 稳定下降
```

### 检查要点 (Check Points)

✅ **对比损失持续下降** - Contrastive loss decreases throughout training
✅ **动态权重生效** - Dynamic weight adjustment works (0.5→1.5)
✅ **训练稳定** - Training is stable across different object counts
✅ **总损失平滑下降** - Total loss decreases smoothly
✅ **mAP提升** - mAP improves (or at least not worse)

## 诊断工具 (Diagnostic Tools)

### 使用损失诊断脚本 (Use Loss Diagnostic Script)

```bash
python diagnose_loss.py \
    --model yolo11n.pt \
    --data coco.yaml \
    --epochs 50 \
    --batch 16
```

这会生成详细的loss分析报告和可视化图表。

This generates detailed loss analysis reports and visualization plots.

## 故障排查 (Troubleshooting)

### 问题1: CUDA内存不足 (CUDA Out of Memory)

**解决方案**:
```bash
# 减小批次大小
python train_coco_test.py  # 编辑脚本，将batch_size改为8或4
```

### 问题2: 数据集下载失败 (Dataset Download Fails)

**解决方案**:
```bash
# 手动下载COCO数据集
# 1. 下载图像
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# 2. 下载标签
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip

# 3. 解压到正确位置
unzip train2017.zip -d datasets/coco/images/
unzip val2017.zip -d datasets/coco/images/
unzip coco2017labels.zip -d datasets/coco/
```

### 问题3: 对比损失仍然plateau (Contrastive Loss Still Plateaus)

**检查**:
1. 确认使用了修复后的代码
2. 检查 `ultralytics/utils/loss.py:1050` 是否使用 `total_num_objects` 归一化
3. 查看详细日志确认loss值范围

## 结果对比 (Results Comparison)

训练完成后，对比两个模型的性能：

After training, compare the performance of both models:

| 指标 (Metric) | 标准模型 (Standard) | 对比学习模型 (Contrast) | 改进 (Improvement) |
|---------------|---------------------|-------------------------|-------------------|
| mAP@0.5 | - | - | - |
| mAP@0.5:0.95 | - | - | - |
| Precision | - | - | - |
| Recall | - | - | - |
| Box Loss (final) | - | - | - |
| Contrast Loss (final) | N/A | - | - |

## 下一步 (Next Steps)

1. ✅ 运行快速验证确认修复有效
2. ✅ 运行完整300轮训练
3. ✅ 分析训练曲线和最终性能
4. ✅ 对比标准模型和对比学习模型
5. ✅ 根据结果进一步优化超参数

## 联系和支持 (Contact and Support)

如有问题，请参考：
- `LOSS_PLATEAU_ANALYSIS.md` - 详细技术分析
- `FIX_SUMMARY.md` - 使用指南和FAQ
- GitHub Issues - 提交问题和讨论

For questions, refer to:
- `LOSS_PLATEAU_ANALYSIS.md` - Detailed technical analysis
- `FIX_SUMMARY.md` - Usage guide and FAQ
- GitHub Issues - Submit questions and discussions

---

**注意**: 这些脚本使用修复后的对比损失归一化。确保您的代码包含最新的修复。

**Note**: These scripts use the fixed contrastive loss normalization. Ensure your code includes the latest fix.
