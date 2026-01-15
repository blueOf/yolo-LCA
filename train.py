"""
YOLOv8 顺序训练脚本
依次使用 dataset_1 到 dataset_5 训练模型
每个数据集训练完成后，使用最佳模型继续下一个数据集的训练

使用方法:
    python train.py <model_path>
    例如: python train.py ultralytics/cfg/models/v8/yolov8.yaml
    或: python train.py ultralytics/cfg/models/11/mobilev4.yaml
"""

from ultralytics import YOLO
import os
import sys
import argparse
from pathlib import Path
import subprocess


def get_available_gpu(available_gpus=[0, 1, 3, 4]):
    """
    自动查找可用GPU

    Args:
        available_gpus: 可用的GPU列表

    Returns:
        最空闲的GPU编号
    """
    try:
        # 使用nvidia-smi查询GPU使用情况
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_info = []
        print("\n" + "=" * 70)
        print("GPU 状态信息:")
        print("=" * 70)

        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 3:
                gpu_id = int(parts[0].strip())
                mem_used = float(parts[1].strip())
                mem_total = float(parts[2].strip())
                mem_free = mem_total - mem_used
                usage_percent = (mem_used / mem_total) * 100

                # 显示所有GPU状态（不仅仅是可用列表中的）
                status = "✓ 可选" if gpu_id in available_gpus else "✗ 不可用"
                print(
                    f"GPU {gpu_id}: {status:8} | 已用: {mem_used:>7.0f}MB ({usage_percent:>5.1f}%) | 空闲: {mem_free:>7.0f}MB | 总计: {mem_total:>7.0f}MB"
                )

                if gpu_id in available_gpus:
                    gpu_info.append((gpu_id, mem_free, mem_used, mem_total))

        print("=" * 70)

        # 按可用内存排序，选择最空闲的GPU
        if gpu_info:
            gpu_info.sort(key=lambda x: x[1], reverse=True)
            best_gpu = gpu_info[0]

            # 显示所有可选GPU的排名
            print("\n可选GPU排名 (按空闲内存):")
            for rank, (gpu_id, mem_free, mem_used, mem_total) in enumerate(gpu_info, 1):
                marker = "→" if rank == 1 else " "
                print(f"  {marker} #{rank} GPU {gpu_id}: 空闲 {mem_free:>7.0f}MB")

            print(
                f"\n✓ 自动选择 GPU {best_gpu[0]} (空闲内存最多: {best_gpu[1]:.0f}MB / {best_gpu[3]:.0f}MB)\n"
            )
            return best_gpu[0] if best_gpu[0] <= 2 else best_gpu[0] - 1
        else:
            print(f"\n警告: 未找到可用GPU，使用默认GPU 0\n")
            return 0

    except Exception as e:
        print(f"\n警告: 无法查询GPU信息 ({e})，使用默认GPU 0\n")
        return 0


def get_model_name(model_path):
    """从模型路径提取模型名称"""
    if model_path.endswith(".yaml"):
        # 提取yaml文件名（不含扩展名）
        return Path(model_path).stem
    elif model_path.endswith(".pt"):
        return Path(model_path).stem
    else:
        return "model"


def train_on_dataset(
    model_path, dataset_num, model_name, device, epochs, batch, workers, is_first=False
):
    """
    在指定数据集上训练模型

    Args:
        model_path: 模型路径（yaml配置文件或权重文件）
        dataset_num: 数据集编号 (1-5)
        model_name: 模型名称（用于保存路径）
        device: GPU设备编号
        epochs: 训练轮数
        batch: 批次大小
        is_first: 是否为第一个数据集（从头训练）

    Returns:
        best_model_path: 训练后的最佳模型路径
    """
    print("\n" + "=" * 70)
    print(f"开始训练 Dataset {dataset_num} - 模型: {model_name} - GPU: {device}")
    print("=" * 70)

    # 加载模型
    model = YOLO("runs/train/yolov11-DWJF_datasets2/weights/best.pt")

    # 训练参数配置
    results = model.train(
        # data=yaml_file,
        data=f"datasets/SVRDD/svrdd.yaml",
        # data=f"datasets/yaml/RDD2022_United_States_{dataset_num}.yaml",
        imgsz=640,
        epochs=epochs,
        batch=batch,
        optimizer="SGD",
        save_period=30,
        seed=0,
        cos_lr=True,
        mosaic=1,
        project="runs/train",
        name=f"{model_name}_datasets",
        amp=True,
        # 为避免多进程 DataLoader 在某些环境下卡死，这里显式使用单进程
        workers=workers,
        resume=True,  # 确保不同数据集的训练结果不会覆盖
        patience=900,
        device=device,
        cache="disk",
    )
    # 获取最佳模型路径
    best_model_path = f"{results.save_dir}/weights/best.pt"

    # 打印训练结果摘要
    print("\n" + "=" * 70)
    print(f"Dataset {dataset_num} 训练完成!")
    print("=" * 70)
    print(f"最佳模型保存在: {best_model_path}")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")
    print("=" * 70 + "\n")

    # 在验证集上评估最佳模型
    print(f"正在评估 Dataset {dataset_num} 的最佳模型...")
    best_model = YOLO(best_model_path)
    metrics = best_model.val()

    print(f"\nDataset {dataset_num} 验证集性能指标:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    return best_model_path


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="开始训练")
    parser.add_argument(
        "model_path", type=str, help="模型配置文件路径（.yaml）或权重文件路径（.pt）"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="GPU设备编号（不指定则自动选择）"
    )
    parser.add_argument(
        "--epochs", type=int, default=900, help="每个数据集的训练轮数（默认：900）"
    )
    parser.add_argument("--batch", type=int, default=16, help="批次大小（默认：16）")
    parser.add_argument(
        "--workers", type=int, default=16, help="数据加载进程数（默认：8）"
    )

    args = parser.parse_args()

    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)

    # 自动选择GPU或使用指定GPU
    if args.device is None:
        device = get_available_gpu([0, 1, 2, 3, 4, 5, 6, 7])
    else:
        device = args.device
        print(f"\n使用指定GPU: {device}")

    # 获取模型名称
    model_name = get_model_name(args.model_path)

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + f"  开始训练 {model_name} 模型".center(68) + "#")
    print("#" + f"  模型路径: {args.model_path}".center(68) + "#")
    print(
        "#"
        + f"  GPU设备: {device}  Epochs: {args.epochs}  Batch: {args.batch}".center(68)
        + "#"
    )
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    # 第一个数据集：从头训练
    current_model = args.model_path
    best_model_path = train_on_dataset(
        current_model,
        1,
        model_name,
        device,
        args.epochs,
        args.batch,
        args.workers,
        is_first=True,
    )

    # for dataset_num in range(2, 4):
    #    # 后续数据集：使用上一个数据集的最佳模型继续训练
    #    current_model = best_model_path
    #    best_model_path = train_on_dataset(
    #        current_model,
    #        dataset_num,
    #        model_name,
    #        device,
    #        args.epochs,
    #        args.batch,
    #        args.workers,
    #    )

    # 最终评估
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  所有数据集训练完成！".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\n最终模型保存在: {best_model_path}")


if __name__ == "__main__":
    main()
