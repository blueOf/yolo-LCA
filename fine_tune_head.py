from ultralytics import YOLO


def main():
    # 1. 加载转换好的权重文件
    model_path = "runs/detect/runs/train/neck_contrastive_finetune_all/weights/last.pt"
    model = YOLO(model_path)

    # 2. 冻结除 head 之外的层并进行训练
    # YOLOv8/YOLO11 的架构通常前22层是 Backbone(0~9) 和 Neck(10~21)。
    # 最后一层(索引起自22)是 Detect(检测头)。
    # freeze=22 表示冻结前 22 层(即 0 到 21 层)。
    results = model.train(
        data="/home/nkq/yolo26/datasets/SVRDD/svrdd.yaml",  # 您的数据集yaml配置文件
        epochs=300,  # 训练论数，您可以根据需要调整
        imgsz=640,  # 图像大小
        batch=32,  # 批次大小
        resume=True,  # 从上次中断的地方继续训练
        # freeze=35,  # 仅解冻 Head (冻结 0~21 层)
        device="3",  # 指定使用第 3 号 GPU (依据您之前的可见设备设置)
        project="runs/train",
        name="neck_contrastive_finetune_all",
    )


if __name__ == "__main__":
    main()
