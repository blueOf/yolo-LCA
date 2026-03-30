import os
import sys
import argparse
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="开始训练 (YOLO Ultraytics wrapper)")

    p.add_argument(
        "model_path",
        type=str,
        help="模型配置文件路径（.yaml）或权重文件路径（.pt）",
    )
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU 设备编号（整数），不指定则自动选择 CPU/GPU（None）",
    )
    p.add_argument("--epochs", type=int, default=800, help="训练总轮数（默认：800）")
    p.add_argument("--batch", type=int, default=32, help="批次大小（默认：64）")
    p.add_argument(
        "--workers", type=int, default=8, help="DataLoader 进程数（默认：8）"
    )
    p.add_argument(
        "--yaml_file",
        type=str,
        default="/home/nkq/yolo26/datasets/SVRDD/svrdd.yaml",
        help="数据集yaml文件路径（COCO-style/Ultralytics dataset yaml）",
    )

    p.add_argument(
        "--cache",
        type=str,
        default="disk",
        help="dataset cache 模式（'disk'/'ram'/'none'）",
    )
    p.add_argument("--optimizer", type=str, default="SGD", help="优化器 (SGD/AdamW...)")
    p.add_argument(
        "--save_period", type=int, default=50, help="保存权重的周期（epoch）"
    )
    p.add_argument("--patience", type=int, default=100, help="早停 patience")
    return p.parse_args()


from pathlib import Path


def get_model_name(model_path):
    """从模型路径提取模型名称"""
    if model_path.endswith(".yaml"):
        # 提取yaml文件名（不含扩展名）
        return Path(model_path).stem
    elif model_path.endswith(".pt"):
        return Path(model_path).stem
    else:
        return "model"


def main():
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件或路径 {args.model_path}")
        sys.exit(1)

    # Validate data yaml
    if not os.path.exists(args.yaml_file):
        print(f"错误: 找不到数据集 yaml 文件 {args.yaml_file}")
        sys.exit(1)

    # Load model (YOLO accepts .yaml or .pt path)
    project = "/home/nkq/yolo11/runs/train"
    model_name = get_model_name(args.model_path)
    # model_name = "yolo11-v2"
    model = YOLO(args.model_path, task="detect")
    # if os.path.exists(f"{project}/{model_name}/weights/last.pt"):
    #    model = YOLO(f"{project}/{model_name}/weights/last.pt")
    # else:
    #    model = YOLO(args.model_path)

    model.train(
        data=args.yaml_file,
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch,
        optimizer=args.optimizer,
        save_period=args.save_period,
        seed=0,
        cos_lr=True,
        mosaic=1,
        project=project,
        name=model_name,
        amp=True,
        workers=args.workers,
        resume=True,
        patience=args.patience,
        device=args.device,
        cache=args.cache,
        # box=10,
        # cls=1,
        # dfl=2,
        # mixup=0.1,
        contrast_weight=2,
        contrast_weight_init=2,
        contrast_weight_final=2,
    )


if __name__ == "__main__":
    main()
