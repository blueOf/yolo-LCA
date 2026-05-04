"""
YOLOv11 验证脚本
在 SVRDD/test 数据集上进行验证
生成性能报告和可视化结果

使用方法:
    python valid_svrdd.py <best_model_path>
    例如:  python valid_svrdd.py runs/train/4class-jpeg/cnn_dataset5/weights/best.pt
"""

from ultralytics import YOLO
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import io
from contextlib import redirect_stdout


def validate_on_svrdd(model, save_dir, device="0"):
    """
    在 SVRDD/test 数据集上验证模型

    Args:
        model: YOLO模型实例
        save_dir: 结果保存目录
        device: 运行设备 (默认："0")

    Returns:
        metrics_dict: 包含验证指标的字典
    """
    print("\n" + "=" * 70)
    print("正在验证 SVRDD/test 数据集...")
    print("=" * 70)

    # 获取模型摘要信息
    print("\n获取模型信息...")
    model_summary = model.info()
    print(f"模型信息: {model_summary}\n")

    # SVRDD test 数据集配置文件路径
    yaml_file = "datasets/RDD2022_YOLO_United_States/dataset_test.yaml"
    # yaml_file = "/home/nkq/yolo26/datasets/SVRDD/svrdd_test.yaml"
    # 验证
    print("正在开始验证...")
    f = io.StringIO()
    with redirect_stdout(f):
        results = model.val(
            data=yaml_file,
            imgsz=640,
            batch=8,
            device=device,
            project=save_dir,
            name="SVRDD_test",
            save_json=True,
            plots=True,
            verbose=True,
            # batch=1,
            # visualize=True,
            # conf=0.25,
            # iou=0.6,
        )
    val_output = f.getvalue()

    # 在控制台打印捕获的输出
    print(val_output)

    # 提取关键指标
    stage_times_ms = results.speed if isinstance(results.speed, dict) else {}
    total_ms = sum(stage_times_ms.values()) if stage_times_ms else 0.0
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    metrics_dict = {
        "dataset": "SVRDD/test",
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1": float(
            2
            * results.box.mp
            * results.box.mr
            / (results.box.mp + results.box.mr + 1e-8)
        ),
        "fps": fps,
        "model_summary": model_summary,
        "val_output": val_output,  # 保存原始输出
        # 新增曲线数据 (全类别平均)
        "px": (
            np.array(results.box.px).flatten() if hasattr(results.box, "px") else None
        ),
        "p_curve": (
            results.box.p_curve.mean(0)
            if hasattr(results.box, "p_curve") and len(results.box.p_curve) > 0
            else None
        ),
        "r_curve": (
            results.box.r_curve.mean(0)
            if hasattr(results.box, "r_curve") and len(results.box.r_curve) > 0
            else None
        ),
        "f1_curve": (
            results.box.f1_curve.mean(0)
            if hasattr(results.box, "f1_curve") and len(results.box.f1_curve) > 0
            else None
        ),
    }

    # 如果有类别指标，也保存
    if hasattr(results.box, "maps"):
        for i, map_val in enumerate(results.box.maps):
            metrics_dict[f"class_{i}_mAP50-95"] = float(map_val)

    print(f"\nSVRDD/test 验证结果:")
    print(f"  mAP50:     {metrics_dict['mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics_dict['mAP50-95']:.4f}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall:    {metrics_dict['recall']:.4f}")
    print(f"  F1-Score:  {metrics_dict['f1']:.4f}")
    print(f"  FPS:       {metrics_dict['fps']:.2f}")

    return metrics_dict


def plot_validation_results(metrics, save_dir):
    """
    绘制验证结果，包含 P-R-F1 曲线和指标摘要表
    """
    # 创建图表 (改为 1x2 布局: 左侧曲线图，右侧表格)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "SVRDD/test Validation Results",
        fontsize=16,
        fontweight="bold",
    )

    # 1. P, R, F1 曲线图
    ax = axes[0]
    if metrics.get("px") is not None and metrics.get("p_curve") is not None:
        px = metrics["px"]
        ax.plot(px, metrics["p_curve"], color="#F18F01", label="Precision", linewidth=2)
        ax.plot(px, metrics["r_curve"], color="#2E86AB", label="Recall", linewidth=2)
        ax.plot(px, metrics["f1_curve"], color="#6A994E", label="F1-Score", linewidth=2)

        # 寻找并标注 F1 最大值点
        f1_mean = metrics["f1_curve"]
        best_idx = np.argmax(f1_mean)
        best_conf = px[best_idx]
        best_f1 = f1_mean[best_idx]

        ax.plot(best_conf, best_f1, "ro", markersize=6)
        ax.annotate(
            f"Best F1: {best_f1:.4f} @ {best_conf:.3f}",
            xy=(best_conf, best_f1),
            xytext=(best_conf, best_f1 + 0.05),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
            horizontalalignment="center",
            fontweight="bold",
        )

        ax.set_xlabel("Confidence", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title("Performance Curves vs Confidence", fontsize=12, fontweight="bold")
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1])
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
    else:
        ax.text(0.5, 0.5, "Curve data not available", ha="center", va="center")

    # 2. 统计摘要表格
    ax = axes[1]
    ax.axis("off")

    summary_data = [
        ["Metric", "Value"],
        ["mAP@0.5", f"{metrics['mAP50']:.6f}"],
        ["mAP@0.5:0.95", f"{metrics['mAP50-95']:.6f}"],
        ["Precision", f"{metrics['precision']:.6f}"],
        ["Recall", f"{metrics['recall']:.6f}"],
        ["F1-Score", f"{metrics['f1']:.6f}"],
        ["FPS", f"{metrics.get('fps', 0):.2f}"],
    ]

    table = ax.table(cellText=summary_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(0.8, 2.5)

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor("#4A90E2")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # 设置行颜色
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F0F0F0")

    ax.set_title("Metrics Summary", fontsize=12, fontweight="bold", pad=20)

    # Add model summary as footer text on the figure
    fig.text(
        0.5,
        0.02,
        metrics.get("model_summary", "Model information not available"),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        wrap=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 保存图表
    plot_path = os.path.join(save_dir, "validation_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n验证结果图表已保存至: {plot_path}")
    plt.close()


def save_results_to_csv(metrics, save_dir):
    """
    将结果保存为CSV文件，仅保存标量指标
    """
    # 过滤掉数组型数据
    scalar_metrics = {
        k: v
        for k, v in metrics.items()
        if not isinstance(v, (np.ndarray, list)) or k == "dataset"
    }

    # 创建DataFrame
    df = pd.DataFrame([scalar_metrics])

    csv_path = os.path.join(save_dir, "validation_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"详细结果已保存至: {csv_path}")


def main():
    """验证主函数"""
    parser = argparse.ArgumentParser(description="在 SVRDD/test 数据集上进行验证")
    parser.add_argument("model_path", type=str, help="最佳模型权重路径 (. pt)")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="运行设备 (默认：0，即GPU 0；可选：cpu, 0, 1, 2等)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/nkq/yolo11/valid",
        help="验证结果保存目录（默认：valid）",
    )

    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)

    # 创建保存目录根与时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).parent.parent.name
    save_dir = os.path.join(args.save_dir, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  YOLOv11 SVRDD/test 验证".center(68) + "#")
    print("#" + f"  模型: {args.model_path}".center(68) + "#")
    print("#" + f"  保存目录: {save_dir}".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    # 加载模型
    print("正在加载模型...")
    model = YOLO(args.model_path, task="detect")
    print(f"✓ 模型加载成功:  {args.model_path}")
    print(f"✓ 运行设备: {args.device}\n")

    # 在 SVRDD/test 上验证
    metrics = validate_on_svrdd(model, save_dir, args.device)

    # 生成报告
    print("\n" + "=" * 70)
    print("生成验证报告...")
    print("=" * 70)

    # 保存CSV
    save_results_to_csv(metrics, save_dir)

    # 绘制图表
    plot_validation_results(metrics, save_dir)

    # 打印最终摘要
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  验证完成！".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print("\n性能摘要:")
    print("=" * 70)
    print(f"mAP@0.5:       {metrics['mAP50']:.6f}")
    print(f"mAP@0.5:0.95:  {metrics['mAP50-95']:.6f}")
    print(f"Precision:     {metrics['precision']:.6f}")
    print(f"Recall:        {metrics['recall']:.6f}")
    print(f"F1-Score:      {metrics['f1']:.6f}")
    print(f"FPS:            {metrics.get('fps', 0):.2f}")
    print(f"Model:         {metrics.get('model_summary')}")
    print("=" * 70)

    print(f"\n所有结果已保存至: {save_dir}/")
    print("  - validation_results.csv  (详细数据)")
    print("  - validation_results.png  (可视化图表)")
    print("  - SVRDD_test/            (验证详细结果)\n")


if __name__ == "__main__":
    main()
