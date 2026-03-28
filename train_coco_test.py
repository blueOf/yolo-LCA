#!/usr/bin/env python3
"""
训练脚本：使用COCO 80类数据集测试300轮
Train script: Test with COCO 80-class dataset for 300 epochs
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    """
    使用COCO数据集训练模型，测试对比学习的修复效果
    Train model with COCO dataset to test contrastive learning fix
    """

    print("="*80)
    print("YOLO COCO 80-Class Training Test - 300 Epochs")
    print("="*80)

    # Configuration
    model_config = "ultralytics/cfg/models/v8/yolov8-contrast.yaml"  # 使用对比学习版本
    data_config = "coco.yaml"  # COCO 80类数据集
    epochs = 300
    batch_size = 16  # 适中的批次大小，适合云端测试
    imgsz = 640

    # Training settings
    project = "runs/coco_test"
    name = "yolov8n-contrast-300ep"

    print(f"\nConfiguration:")
    print(f"  Model: {model_config}")
    print(f"  Dataset: {data_config}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Project: {project}")
    print(f"  Name: {name}")
    print("="*80)

    # Check if model config exists
    if not os.path.exists(model_config):
        print(f"错误: 找不到模型配置文件 {model_config}")
        print(f"Error: Model config file not found {model_config}")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    model = YOLO(model_config, task="detect")

    # Training hyperparameters with updated contrastive loss weights
    # 使用修复后的对比学习权重配置
    print("\nStarting training with updated contrastive loss configuration...")
    print("使用修复后的对比损失归一化...")

    results = model.train(
        data=data_config,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch_size,

        # Optimizer settings
        optimizer="SGD",
        lr0=0.01,  # initial learning rate
        lrf=0.01,  # final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,

        # Learning rate scheduler
        cos_lr=True,  # Use cosine learning rate scheduler

        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # Training settings
        seed=0,
        device=0,  # GPU 0
        workers=8,
        amp=True,  # Automatic Mixed Precision

        # Save settings
        project=project,
        name=name,
        exist_ok=False,
        save=True,
        save_period=50,  # Save checkpoint every 50 epochs

        # Validation settings
        val=True,

        # Early stopping
        patience=100,  # Stop if no improvement for 100 epochs

        # Loss weights - 使用修复后的推荐配置
        # Updated weights after normalization fix
        box=7.5,
        cls=0.5,
        dfl=1.5,
        contrast_weight=0.5,  # 增加基础权重（修复后loss值更小）
        contrast_weight_init=0.5,  # 起始权重
        contrast_weight_final=1.5,  # 终止权重（动态增加）

        # Cache for faster training
        cache=False,  # Don't cache (COCO is large)

        # Resume settings
        resume=False,  # Don't resume from checkpoint

        # Verbose
        verbose=True,
    )

    print("\n" + "="*80)
    print("Training completed!")
    print("训练完成！")
    print("="*80)

    # Print results location
    save_dir = Path(project) / name
    print(f"\nResults saved to: {save_dir}")
    print(f"结果保存在: {save_dir}")

    print("\nKey files:")
    print(f"  - Best weights: {save_dir}/weights/best.pt")
    print(f"  - Last weights: {save_dir}/weights/last.pt")
    print(f"  - Results CSV: {save_dir}/results.csv")
    print(f"  - Training plots: {save_dir}/*.png")

    # Print final metrics if available
    if hasattr(results, 'results_dict'):
        print("\nFinal metrics:")
        print(results.results_dict)

    print("\n" + "="*80)
    print("Test complete! 测试完成！")
    print("="*80)

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ Training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
