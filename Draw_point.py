import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ============= 数据提取 =============
models = [
    "RT-DETR",
    "FPDDN-DA",
    "Faster-RCNN",
    "RetinaNet",
    "YOLOv8n",
    "YOLOv8s",
    "YOLOv11n",
    "YOLOv11s",
    "RPDNet",
    "LRDD-YOLOv8",
    "EdgeAware-YOLO",
    "Ours",
]

map50 = [
    54.42,
    67.34,
    64.9,
    60.4,
    63.5,
    68.76,
    65.2,
    68.84,
    70.88,
    64.4,
    65.1,
    69.64,
]
flops = [101.0, 37.26, 63.3, 39.7, 8.7, 28.8, 6.5, 21.5, 35.6, 5.8, 5.8, 7.7]
parameters = [44.0, 5.43, 41.14, 36.17, 3.2, 11.2, 2.6, 9.44, 15.39, 2.02, 2.4, 2.69]

# 手动设置高区分度颜色
colors = [
    "#E6194B",  # RT-DETR - 红色
    "#BCFEC1",  # FPDDN-DA - 绿色
    "#FFE119",  # Faster-RCNN - 黄色
    "#4363D8",  # RetinaNet - 蓝色
    "#F58231",  # YOLOv8n - 橙色
    "#911EB4",  # YOLOv8s - 紫色
    "#46F0F0",  # YOLOv11n - 青色
    "#F032E6",  # YOLOv11s - 品红
    "#3CB44B",  # RPDNet - 浅绿
    "#FABEBE",  # LRDD-YOLOv8 - 粉色
    "#008080",  # EdgeAware-YOLO - 鸭羽色
    "#000000",  # Ours - 黑色 (为了突出)
]


def plot_artistic_scatter(
    data_x, data_y, x_label, y_label, title, x_min=0, x_max=None, y_min=50, y_max=90
):

    if x_max is None:
        x_max = max(data_x) * 1.1

    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置背景
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # 绘制网格（更精细）
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.8, color="gray")
    ax.set_axisbelow(True)

    # 绘制每个点和连线
    for i, (x, y) in enumerate(zip(data_x, data_y)):
        # 计算显示的点坐标
        if x > x_max:
            # 线性收缩：(display_x - x_min) / (x - x_min) = (x_max - x_min) / (actual_x - x_min)
            # 这里简化为将超出部分等比压缩到边界，点落在 x_max 上
            display_x = x_max
            # y 也需要按照比例收缩以保持连线斜率一致
            ratio = (x_max - x_min) / (x - x_min)
            display_y = y_min + (y - y_min) * ratio
        else:
            display_x = x
            display_y = y

        # 绘制从原点 (x_min, y_min) 到显示点的直线
        ax.plot(
            [x_min, display_x],
            [y_min, display_y],
            color=colors[i],
            alpha=0.7,
            linewidth=2.2,
            zorder=2,
            linestyle="-",
        )

        # 绘制散点
        ax.scatter(
            display_x,
            display_y,
            s=150,
            color=colors[i],
            edgecolors="white",
            linewidth=2,
            alpha=1,
            zorder=3,
        )

        # 如果超出范围，添加真实数值标注
        if x > x_max:
            ax.annotate(
                f"FLOPs={x:.1f}",
                xy=(display_x, display_y),
                xytext=(-10, 10),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                ha="right",
                color=colors[i],
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor=colors[i],
                    linewidth=1,
                ),
                zorder=5,
            )

    # 标记原点
    ax.plot(
        x_min,
        y_min,
        "o",
        color="black",
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=4,
    )

    # 设置坐标轴
    ax.set_xlabel(x_label, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 美化坐标轴
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(labelsize=11)

    return fig, ax


# ============= 创建两个独立图表（仿照美术风格）=============

# 图1: mAP50 vs FLOPs
print("生成图1: mAP50 vs FLOPs (美术风格)")
fig1, ax1 = plot_artistic_scatter(
    flops,
    map50,
    x_label="FLOPs (G)",
    y_label="mAP@0.5 (%)",
    title="Detection Models: mAP@0.5 vs FLOPs",
    x_min=0,
    x_max=45,
    y_min=50,
    y_max=90,
)

# 添加图例（在右上角）
legend_elements_1 = [
    Line2D([0], [0], color=colors[i], linewidth=10, label=models[i])
    for i in range(len(models))
]
fig1.legend(
    handles=legend_elements_1,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.93),
    fontsize=9.5,
    frameon=False,
    fancybox=False,
    shadow=False,
    edgecolor="black",
    labelspacing=1,
)

plt.tight_layout()
plt.savefig(
    "mAP50_vs_FLOPs_artistic.png", dpi=300, bbox_inches="tight", facecolor="white"
)
plt.show()

# 图2: mAP50 vs Parameters
print("生成图2: mAP50 vs Parameters (美术风格)")
fig2, ax2 = plot_artistic_scatter(
    parameters,
    map50,
    x_label="Parameters (M)",
    y_label="mAP@0.5 (%)",
    title="Detection Models: mAP@0.5 vs Parameters",
    x_min=0,
    x_max=20,
    y_min=50,
    y_max=90,
)

# 添加图例（在右上角）
legend_elements_2 = [
    Line2D([0], [0], color=colors[i], linewidth=10, label=models[i])
    for i in range(len(models))
]
fig2.legend(
    handles=legend_elements_1,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.93),
    fontsize=9.5,
    frameon=False,
    fancybox=False,
    shadow=False,
    edgecolor="black",
    labelspacing=1,
)

plt.tight_layout()
plt.savefig(
    "mAP50_vs_Parameters_artistic.png", dpi=300, bbox_inches="tight", facecolor="white"
)
plt.show()

print("\n✓ 两个美术风格的图表已生成")
print("  - mAP50_vs_FLOPs_artistic.png")
print("  - mAP50_vs_Parameters_artistic.png")
print("\n特点:")
print("  ✓ 仿照参考图的美术风格")
print("  ✓ 直线连接到原点 (0, 50)")
print("  ✓ 彩色点和线条设计")
print("  ✓ 白色边框的散点")
print("  ✓ 精细网格背景")
print("  ✓ 右上角图例")
