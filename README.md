# NeckContrastive 训练使用手册（YOLO 标准数据集）

本文档说明如何使用本仓库中的 `NeckContrastiveTrainer` + `NeckContrastiveYOLOData`，基于 **YOLO 标准格式数据集 YAML** 进行训练，并明确输入、输出与模型存储路径。

---

## 1. 你需要准备的输入

### 1.1 数据集 YAML（YOLO 标准格式）

示例：`./datasets/mydata.yaml`

```yaml
path: /data/my_dataset
train: images/train
val: images/val

names:
  0: person
  1: car
```

> 要求：
> - `train`、`val` 必须存在
> - 标注是 YOLO txt 格式（`class cx cy w h`，归一化）
> - `names` 与类别数量一致

### 1.2 模型权重

例如：
- `./yolo11n.pt`

---

## 2. 训练代码（完整示例）

将以下内容保存为 `train_neck_contrastive_manual.py`（路径可自定义）：

```python
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.tools.neck_contrastive import NeckContrastiveTrainer, NeckContrastiveYOLOData
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig


def main():
    # ===== 1) 基础配置 =====
    data_yaml = "./datasets/mydata.yaml"   # 你的 YOLO 数据集 YAML
    weights = "./yolo11n.pt"               # 初始权重
    save_dir = Path("./runs/neck_contrastive/exp1")
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = 10
    batch_size = 16
    imgsz = 640
    lr = 1e-3
    stride = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 2) 模型 =====
    model = YOLO(weights)
    model.model.to(device)

    # 检测头输入层索引，常用于抽取 P3/P4/P5 neck 特征
    feature_layers = list(model.model.model[-1].f)

    # ===== 3) 数据加载（读取 YOLO YAML）=====
    data_helper = NeckContrastiveYOLOData(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        workers=8,
        autodownload=False,
    )
    train_loader = data_helper.build_dataloader(mode="train", stride=stride)
    val_loader = data_helper.build_dataloader(mode="val", stride=stride, shuffle=False)

    # ===== 4) Trainer 与优化器 =====
    trainer = NeckContrastiveTrainer(
        model=model,
        config=NeckContrastiveConfig(
            feature_layers=feature_layers,
            contrastive_weight=0.3,
            use_detection_loss=True,
            freeze_head=True,
        ),
    )
    optimizer = torch.optim.SGD(model.model.parameters(), lr=lr, momentum=0.937, weight_decay=5e-4)

    # ===== 5) 训练循环 =====
    for epoch in range(epochs):
        model.model.train()
        running_total = 0.0
        running_det = 0.0
        running_ctr = 0.0
        n = 0

        for batch in train_loader:
            batch = data_helper.preprocess_batch(batch, device=device)
            metrics = trainer.training_step(batch, optimizer)
            # metrics: total_loss / det_loss / contrastive_loss / det_items
            running_total += float(metrics["total_loss"])
            running_det += float(metrics["det_loss"])
            running_ctr += float(metrics["contrastive_loss"])
            n += 1

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"total={running_total / max(n,1):.4f}, "
            f"det={running_det / max(n,1):.4f}, "
            f"contrastive={running_ctr / max(n,1):.4f}"
        )

        # ===== 6) 每轮保存模型 =====
        ckpt = {
            "epoch": epoch + 1,
            "model": model.model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch + 1:03d}.pt")

    # 可选：保存最终权重
    torch.save(model.model.state_dict(), save_dir / "last.pt")

    # 关闭 hook
    trainer.close()


if __name__ == "__main__":
    main()
```

运行：

```bash
cd /path/to/yolo-jf
PYTHONPATH=. python train_neck_contrastive_manual.py
```

---

## 3. 输入 / 输出说明

### 3.1 输入

1. **数据集配置**：YOLO YAML 文件路径（如 `datasets/mydata.yaml`）  
2. **模型权重**：如 `yolo11n.pt`  
3. **训练参数**：`epochs`、`batch`、`imgsz`、`lr` 等  
4. **Neck 对比学习参数**（`NeckContrastiveConfig`）：
   - `feature_layers`: 抽取特征的层索引（必填）
   - `contrastive_weight`: 对比损失权重
   - `use_detection_loss`: 是否叠加检测损失
   - `freeze_head`: 是否冻结检测头

### 3.2 输出

每个 step 的 `training_step` 返回：

- `total_loss`：总损失（`det_loss + contrastive_weight * contrastive_loss`）
- `det_loss`：检测损失（若关闭检测损失则为 0）
- `contrastive_loss`：Neck 对比损失
- `det_items`：检测损失细项（用于日志）

---

## 4. 模型存储路径

在上面示例中，模型默认保存到：

`./runs/neck_contrastive/exp1`

会生成：

- `epoch_001.pt`
- `epoch_002.pt`
- ...
- `last.pt`

你只需修改代码中的 `save_dir`，即可自定义模型存储位置。

---

## 5. 常见问题

1. **数据 YAML 报错**：请确认 `train`、`val`、`names` 字段正确。  
2. **显存不足**：减小 `batch_size` 或 `imgsz`。  
3. **loss 不下降**：尝试开启 `use_detection_loss=True`，并降低 `lr`。  
4. **只想做对比学习**：设置 `use_detection_loss=False`。  
