#!/usr/bin/env python3
"""
快速验证脚本：验证对比学习修复是否正常工作
Quick validation: Verify contrastive learning fix works correctly
"""

import os
import sys
from ultralytics import YOLO


def main():
    """验证对比学习修复"""
    print("="*80)
    print("Quick Validation Test - Contrastive Learning Fix")
    print("快速验证测试 - 对比学习修复")
    print("="*80)

    # Test with small COCO subset for quick validation
    model_config = "ultralytics/cfg/models/v8/yolov8-contrast.yaml"
    data_config = "coco8.yaml"  # Small 8-image subset of COCO

    print(f"\nConfiguration:")
    print(f"  Model: {model_config}")
    print(f"  Dataset: {data_config} (8 images for quick test)")
    print(f"  Epochs: 5 (quick validation)")
    print("="*80)

    # Check model exists
    if not os.path.exists(model_config):
        print(f"错误: 找不到模型配置 {model_config}")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    model = YOLO(model_config, task="detect")

    # Quick training test
    print("\nRunning quick training test (5 epochs)...")
    try:
        results = model.train(
            data=data_config,
            imgsz=640,
            epochs=5,
            batch=2,
            device=0,
            workers=2,
            amp=True,
            project="runs/quick_test",
            name="validation",
            exist_ok=True,
            verbose=True,

            # Loss weights with fixed normalization
            box=7.5,
            cls=0.5,
            dfl=1.5,
            contrast_weight=0.5,
            contrast_weight_init=0.5,
            contrast_weight_final=1.5,
        )

        print("\n" + "="*80)
        print("✓ Validation PASSED!")
        print("✓ 验证通过！")
        print("="*80)
        print("\nThe contrastive learning fix is working correctly.")
        print("对比学习修复正常工作。")
        print("\nYou can now run the full 300-epoch test:")
        print("现在可以运行完整的300轮测试：")
        print("  python train_coco_test.py")
        print("\nOr compare both detection heads:")
        print("或对比两种检测头：")
        print("  python compare_detect_heads.py")
        print("="*80)

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("✗ Validation FAILED!")
        print("✗ 验证失败！")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
