#!/usr/bin/env python3
"""
对比测试脚本：对比标准检测头和对比学习检测头
Comparison test: Standard Detect head vs ContrastDetect head
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO


def train_model(model_config, data_config, name, epochs=300, batch_size=16):
    """
    训练单个模型
    Train a single model

    Args:
        model_config: Model configuration file path
        data_config: Dataset configuration file path
        name: Experiment name
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print("\n" + "="*80)
    print(f"Training: {name}")
    print("="*80)
    print(f"Model: {model_config}")
    print(f"Dataset: {data_config}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {batch_size}")
    print("="*80)

    # Load model
    model = YOLO(model_config, task="detect")

    # Training settings
    results = model.train(
        data=data_config,
        imgsz=640,
        epochs=epochs,
        batch=batch_size,

        # Optimizer
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        # Training
        seed=0,
        device=0,
        workers=8,
        amp=True,

        # Save
        project="runs/coco_comparison",
        name=name,
        exist_ok=False,
        save=True,
        save_period=50,

        # Validation
        val=True,
        patience=100,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Contrastive weights (只对ContrastDetect有效)
        contrast_weight=0.5,
        contrast_weight_init=0.5,
        contrast_weight_final=1.5,

        # Cache
        cache=False,
        resume=False,
        verbose=True,
    )

    print(f"\n✓ {name} training completed!")
    return results


def main():
    """
    主函数：运行对比实验
    Main function: Run comparison experiments
    """
    print("="*80)
    print("YOLO Detection Head Comparison Test")
    print("YOLO检测头对比测试")
    print("="*80)
    print("\nThis script will train two models:")
    print("本脚本将训练两个模型：")
    print("1. Standard YOLOv8 with Detect head (标准检测头)")
    print("2. YOLOv8 with ContrastDetect head (对比学习检测头)")
    print("\nBoth models will be trained for 300 epochs on COCO dataset.")
    print("两个模型都将在COCO数据集上训练300轮。")
    print("="*80)

    # Configuration
    data_config = "coco.yaml"
    epochs = 300
    batch_size = 16

    # Test 1: Standard Detect head
    print("\n\n" + "#"*80)
    print("# Test 1: Standard YOLOv8 with Detect head")
    print("# 测试1：标准YOLOv8检测头")
    print("#"*80)

    try:
        results_standard = train_model(
            model_config="ultralytics/cfg/models/v8/yolov8.yaml",
            data_config=data_config,
            name="yolov8n-standard-300ep",
            epochs=epochs,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"\n✗ Standard model training failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: ContrastDetect head with fixed normalization
    print("\n\n" + "#"*80)
    print("# Test 2: YOLOv8 with ContrastDetect head (Fixed Normalization)")
    print("# 测试2：YOLOv8对比学习检测头（修复归一化后）")
    print("#"*80)

    try:
        results_contrast = train_model(
            model_config="ultralytics/cfg/models/v8/yolov8-contrast.yaml",
            data_config=data_config,
            name="yolov8n-contrast-300ep",
            epochs=epochs,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"\n✗ Contrast model training failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n\n" + "="*80)
    print("COMPARISON TEST COMPLETED")
    print("对比测试完成")
    print("="*80)

    print("\nResults are saved in:")
    print("结果保存在：")
    print("  - runs/coco_comparison/yolov8n-standard-300ep/")
    print("  - runs/coco_comparison/yolov8n-contrast-300ep/")

    print("\nTo compare results, check:")
    print("要对比结果，请查看：")
    print("  - results.csv files for training metrics")
    print("  - weights/best.pt for best model weights")
    print("  - *.png files for training plots")

    print("\nYou can also use TensorBoard:")
    print("您也可以使用TensorBoard：")
    print("  tensorboard --logdir runs/coco_comparison")

    print("\n" + "="*80)
    print("All tests completed! 所有测试完成！")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n✗ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
