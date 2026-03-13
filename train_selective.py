import os
import sys
import argparse
import torch
import torch.nn as nn
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

    # 1. 加载新模型结构
    model = YOLO(args.model_path)
    new_csd = model.model.state_dict()
    updated_csd = {}

    # 2. 加载 YOLO11 官方/基础权重 (yolo11.yaml 对应的权重)
    # 映射规则：0-3 不变, 4 变为新模型的 5, 5 变为 6, 依此类推
    base_ckpt_path = "runs/train/yolo11/weights/best.pt"
    if os.path.exists(base_ckpt_path):
        print(f"Loading base weights from {base_ckpt_path}...")
        ckpt_base = torch.load(base_ckpt_path, map_location="cpu")
        old_csd_base = (
            ckpt_base["model"].float().state_dict()
            if "model" in ckpt_base
            else ckpt_base.state_dict()
        )

        for k, v in old_csd_base.items():
            if not k.startswith("model."):
                continue
            parts = k.split(".")
            try:
                idx = int(parts[1])
                if idx <= 3:
                    new_k = k  # 0, 1, 2, 3 层保持一致
                else:
                    new_k = f"model.{idx + 1}.{'.'.join(parts[2:])}"  # 4层及以后索引+1

                if new_k in new_csd and v.shape == new_csd[new_k].shape:
                    updated_csd[new_k] = v
            except (ValueError, IndexError):
                continue

    # 3. 初始化新模块为恒等映射 (Identity Initialization)
    # 我们不再加载 old-convWithA2 的权重，而是让第 4 层在初始状态下输出尽量接近输入
    print("Initializing layer 4 as identity-like mapping...")
    layer_4 = model.model.model[4]
    if hasattr(layer_4, "norm_o"):
        # 如果使用了 InstanceNorm，将其 weight 初始化为 0，让初始输出受限
        # 或者将最后投影/归一化的权重设为极小值，bias 设为 0
        nn.init.constant_(layer_4.norm_o.weight, 0.0)
        nn.init.constant_(layer_4.norm_o.bias, 0.0)

    # 如果 ConvWithAttention 内部有 w3 或类似的输出层，也可以将其 weight 设为 0
    # 这样在残差结构中（如果有），初始输出就完全等于输入 x

    # 4. 加载到模型
    load_info = model.model.load_state_dict(updated_csd, strict=False)
    print(f"Total mapped keys: {len(updated_csd)}")

    # 获取缺失权重的层索引，方便调试
    missing_layers = sorted(
        list(
            set(
                [
                    k.split(".")[1]
                    for k in load_info.missing_keys
                    if k.startswith("model.")
                ]
            )
        )
    )
    print(f"Layers with missing keys: {missing_layers}")

    # 验证 Detect 层（用户反馈 Detect 是第 24 层，索引 24）
    detect_keys = [k for k in updated_csd.keys() if "model.24" in k]
    print(f"Detect layer (Layer 24) mapped keys: {len(detect_keys)}")
    if len(detect_keys) == 0:
        print("WARNING: Detect layer weights NOT mapped! Metrics will be zero.")

    num_layers = len(model.model.model)
    # 5. 冻结策略：解冻第 4 层（新模块）和第 24 层（Detect 头）
    trainable_indices = [4, 24]
    freeze_list = [i for i in range(num_layers) if i not in trainable_indices]
    print(
        f"Freezing {len(freeze_list)} layers, training layers {trainable_indices} only."
    )

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
        # resume=True,
        patience=args.patience,
        device=args.device,
        cache=args.cache,
        # box=10,
        # cls=1,
        # dfl=1.5,
        mixup=0.1,
        # freeze=freeze_list,
    )


if __name__ == "__main__":
    main()
