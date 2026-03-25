import os

# 设置可见的 CUDA 设备编号，例如使用 0 号甚至 0,1 号设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from ultralytics import YOLO
from ultralytics.tools.neck_contrastive import (
    NeckContrastiveTrainer,
    NeckContrastiveYOLOData,
)
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig
import torch

# 1) 模型
model = YOLO("runs/train/yolo11-sub_mean/weights/best.pt")
feature_layers = list(model.model.model[-1].f)

# 2) 数据（你的 YOLO 标准 yaml）
data = NeckContrastiveYOLOData(
    data="/home/nkq/yolo26/datasets/SVRDD/svrdd.yaml",
    imgsz=640,
    batch=16,
    workers=8,
)

train_loader = data.build_dataloader(mode="train", stride=32)

# 3) Trainer
trainer = NeckContrastiveTrainer(
    model=model,
    config=NeckContrastiveConfig(feature_layers=feature_layers, contrastive_weight=0.3),
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(
    model.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=300, eta_min=1e-5
)

epochs = 300
print(
    f"Starting neck contrastive training loop for {epochs} epochs on device {device}..."
)

# 4) 训练循环 (多 epoch)
for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    epoch_metrics = {"total_loss": 0.0, "det_loss": 0.0, "contrastive_loss": 0.0}
    num_batches = 0

    for i, batch in enumerate(train_loader):
        batch = data.preprocess_batch(batch, device=device)
        metrics = trainer.training_step(batch, optimizer)

        # Accumulate metrics
        epoch_metrics["total_loss"] += metrics["total_loss"].item()
        epoch_metrics["det_loss"] += metrics["det_loss"].item()
        epoch_metrics["contrastive_loss"] += metrics["contrastive_loss"].item()
        num_batches += 1

        if i % 10 == 0:
            print(
                f"  Batch {i}/{len(train_loader)} | Total Loss: {metrics['total_loss']:.4f} | Det Loss: {metrics['det_loss']:.4f} | Cont Loss: {metrics['contrastive_loss']:.4f}"
            )

    scheduler.step()

    # Calculate averages
    avg_total_loss = epoch_metrics["total_loss"] / max(num_batches, 1)
    avg_det_loss = epoch_metrics["det_loss"] / max(num_batches, 1)
    avg_cont_loss = epoch_metrics["contrastive_loss"] / max(num_batches, 1)

    print(
        f"Epoch {epoch+1} Completed | Avg Total: {avg_total_loss:.4f} | Avg Det: {avg_det_loss:.4f} | Avg Cont: {avg_cont_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    # 保存模型 / Save model
    save_path = f"runs/train/neck_contrastive/neck_contrastive_epoch_{epoch+1}.pt"
    # Ensure directory exists:
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model in a format compatible with Ultralytics YOLO loading
    import copy

    ckpt = {"model": copy.deepcopy(model.model).half(), "epoch": epoch}
    torch.save(ckpt, save_path)
    print(f"Model saved to {save_path}")

trainer.close()
