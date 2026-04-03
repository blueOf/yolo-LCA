from ultralytics.nn.tasks import temporary_modules
from ultralytics import YOLO

# 1. 定义映射关系：{"旧路径": "新路径"}
# 假设你将模块名从 MyOldModule 改为了 MyNewModule
mapping = {
    "ultralytics.nn.modules.block.FastLocalDETRBlock": "ultralytics.nn.modules.block.FastLocalDETRBlock"
}

with temporary_modules(modules=mapping):
    # 2. 在上下文管理器中加载模型
    model = YOLO("runs/train/yolo11-fast-dert-142-clossv3-2/weights/best.pt")

    # 3. 重新保存，保存后的文件将使用新的模块名
    model.save("runs/train/yolo11-fast-dert-142-clossv3-2/weights/best-2.pt")
