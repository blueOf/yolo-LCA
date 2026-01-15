"""
plot_yolo_metrics.py

函数：plot_csv_metrics(csv_path, out_file=None, smooth_window=0, figsize=(14,10))
- 读取 CSV（示例为 YOLO 训练日志 CSV）
- 去掉以 lr/ 开头的列（或通过 exclude_prefixes 自定义）
- 将 loss 类列（train/xx_loss, val/xx_loss）和 metric 类列分组
- 在一张大图中绘制：
    - 上半部分：所有 loss 曲线（train 与 val）
    - 下半部分：所有 metrics（precision/recall/mAP 等）
- 支持简单滑动平均平滑（smooth_window）

示例用法（脚本方式）：
    python plot_yolo_metrics.py --csv train_log.csv --out train_metrics.png --smooth 3

依赖：
    pandas, matplotlib, seaborn, numpy
"""

from typing import List, Optional
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def _clean_label(col: str) -> str:
    # 更可读的标签：将 "/" -> " "，去掉括号及多余空格
    label = col.replace("/", " ")
    label = re.sub(r"\(.*?\)", "", label)  # remove parentheses content like (B)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def plot_csv_metrics(
    csv_path: str,
    out_file: Optional[str] = None,
    exclude_prefixes: Optional[List[str]] = None,
    smooth_window: int = 0,
    figsize: tuple = (14, 10),
    dpi: int = 150,
    title: Optional[str] = None,
):
    """
    读取 CSV 并绘制训练曲线（合并为一张大图，分两行：loss 与 metrics）。

    参数:
        csv_path: CSV 文件路径
        out_file: 若提供则保存图片到该路径；否则显示图像
        exclude_prefixes: 排除列名前缀列表，默认 ['lr/']
        smooth_window: 若 >0，使用 rolling mean 平滑（窗口大小）
        figsize: 输出图像大小
        dpi: 输出分辨率
        title: 图像标题（可选）
    """
    if exclude_prefixes is None:
        exclude_prefixes = ["lr/"]

    # 读取 CSV（自动去掉列名两端空格）
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # drop excluded columns
    drop_cols = [
        c for c in df.columns if any(c.startswith(p) for p in exclude_prefixes)
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 尝试定位 epoch 列
    epoch_cols = [
        c for c in df.columns if c.lower() == "epoch" or c.lower().startswith("epoch")
    ]
    if epoch_cols:
        x_col = epoch_cols[0]
    else:
        # 如果没有 epoch 列，默认使用第一列作为 x
        x_col = df.columns[0]

    x = df[x_col].astype(float)

    # 分类列：loss 列（名称包含 'loss'），其余为 metrics（排除 epoch）
    other_cols = [c for c in df.columns if c != x_col]
    loss_cols = [c for c in other_cols if "loss" in c.lower()]
    metric_cols = [c for c in other_cols if c not in loss_cols]

    # 如果没有 losses 或 metrics，保证不会报错
    sns.set(style="darkgrid")
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1, 1.0]}
    )
    ax_loss, ax_metric = axes

    # 绘制 loss 曲线（train 与 val 在同一子图）
    palette = sns.color_palette("tab10")
    for i, col in enumerate(sorted(loss_cols)):
        y = df[col].astype(float)
        if smooth_window and smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1, center=False).mean()
        label = _clean_label(col)
        # 用虚线区分 val 系列（列名中包含 'val'）
        linestyle = "--" if "val" in col.lower() else "-"
        ax_loss.plot(
            x,
            y,
            label=label,
            color=palette[i % len(palette)],
            linestyle=linestyle,
            linewidth=1.6,
        )

    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss curves" if title is None else f"{title} — Loss")
    ax_loss.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize="small"
    )

    # 绘制 metrics（precision/recall/mAP 等）
    for i, col in enumerate(sorted(metric_cols)):
        y = pd.to_numeric(df[col], errors="coerce")
        if smooth_window and smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1, center=False).mean()
        label = _clean_label(col)
        ax_metric.plot(
            x, y, label=label, color=palette[(i + 3) % len(palette)], linewidth=1.6
        )

    ax_metric.set_ylabel("Metric (precision / recall / mAP / ...)")
    ax_metric.set_xlabel(x_col)
    ax_metric.set_title("Metrics" if title is None else f"{title} — Metrics")
    ax_metric.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize="small"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title:
        fig.suptitle(title, fontsize=14)

    if out_file:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot YOLO training CSV: losses + metrics in one large figure"
    )
    parser.add_argument("--csv", type=str, required=True, help="path to CSV file")
    parser.add_argument(
        "--out",
        type=str,
        default="all.png",
        help="output image (png). If omitted, show interactively",
    )
    parser.add_argument(
        "--smooth", type=int, default=0, help="rolling smooth window (int > 0)"
    )
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(14.0, 10.0), help="figure size (w h)"
    )
    parser.add_argument("--dpi", type=int, default=150, help="output DPI")
    parser.add_argument("--title", type=str, default=None, help="figure title")
    args = parser.parse_args()

    plot_csv_metrics(
        csv_path=args.csv,
        out_file=args.out,
        smooth_window=args.smooth,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        title=args.title,
    )
