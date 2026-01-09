from ultralytics import YOLO

# ---------- 加载模型 ----------
# model = YOLO("runs/train/4class-jpeg/cnn_dataset5/weights/best.pt")
model = YOLO("runs/train/4class/cnn_dataset52/weights/best.pt")

# ---------- 模型预测 ----------
model.predict(
    source="datasets/dataset_1/val/images/",
    save=True,
    save_conf=True,
)
