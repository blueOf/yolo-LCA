import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/mobilent/yolov11-DWJF.yaml")

# 训练参数配置
results = model.train(
    data=f"datasets/SVRDD/svrdd.yaml",
    imgsz=640,
    epochs=100,
    batch=32,
    project="runs/train_quick",
    name="quick_test",
    cache="disk",
    device=3,
    amp=True,
    workers=16,
    optimizer="SGD",
    save_period=30,
    seed=0,
    cos_lr=True,
    mosaic=1,
)
