import torch
import cv2
import numpy as np
from ultralytics import YOLO

# model = YOLO("runs/train/yolo11-fast-dert-142/weights/best.pt")
model = YOLO("/home/nkq/yolo11/runs/train/yolo11/weights/best.pt")
# model = YOLO("runs/train/yolo11-sub_mean/weights/best.pt")
# model = YOLO("runs/train/yolo11-fast-dert-142-clossv3-2/weights/best.pt")
features = []


def hook_fn(module, input, output):
    # output 是原始特征图，不会被自动池化
    features.append(output)


# 注册到你想要的层（例如第 4 层）
layer = 19
handle = model.model.model[layer].register_forward_hook(hook_fn)

# 执行常规推理
img_path = "test1.jpg"
model(img_path)

# 1. 提取特征图
feat = features[1]  # Shape: [1, 64, 80, 80]


# --- 可选预处理步骤 ---
preprocess_mode = "none"  # 选项: "none", "norm", "sub_mean", "sub_median"

if preprocess_mode == "norm":
    # L2 归一化 (在通道维度)
    feat = torch.nn.functional.normalize(feat, p=2, dim=1)
elif preprocess_mode == "sub_mean":
    # 减去均值 (全图均值或通道均值，这里采用通道级均值)
    mean = feat.mean(dim=(2, 3), keepdim=True)
    feat = feat - mean
elif preprocess_mode == "sub_median":
    # 减去中位数 (通道级中位数)
    # 转换为 numpy 处理更方便，或者在 torch 中使用 median
    median = (
        feat.view(1, feat.size(1), -1).median(dim=-1)[0].view(1, feat.size(1), 1, 1)
    )
    feat = feat - median
# ---------------------

_, C, H, W = feat.shape

# 2. 指定坐标 (x, y)，提取对应的 64 维通道向量
# 注意：在图像坐标系中，x 是列，y 是行
target_x, target_y = 14, 28  # 示例坐标 28, 57 竖直裂缝 42, 59 patch  20, 20树叶
target_vec = feat[0, :, target_y, target_x].view(1, C, 1, 1)  # Shape: [1, 64, 1, 1]

similarity = torch.nn.functional.cosine_similarity(
    feat, target_vec, dim=1
)  # Shape: [1, 80, 80]
similarity = similarity.squeeze(0).cpu().numpy()

flattened_sim = similarity.flatten()
threshold = np.percentile(flattened_sim, 100)  # 前 10% 即 90 百分位数
max_val = flattened_sim.max()
mask = (similarity >= threshold).astype(np.uint8) * 255

# 保存相似度矩阵为 CSV (2D: H x W)
np.savetxt("similarity_matrix.csv", similarity, delimiter=",")
feat_2d = feat.squeeze(0).cpu().numpy().reshape(C, -1).T
np.savetxt("feat_matrix.csv", feat_2d, delimiter=",")

cv2.imwrite("similarity_mask.png", mask)

# 6. 优化热力图：将 90% 到 100% 分位数拉伸到 0-255 进行显示
# 这样可以让原本在 0.9x 附近的差异更明显
stretched_sim = np.clip((similarity - threshold) / (max_val - threshold + 1e-8), 0, 1)
sim_norm = (stretched_sim * 255).astype(np.uint8)
sim_norm = cv2.normalize(similarity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

heatmap = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
cv2.imwrite("similarity_heatmap.png", heatmap)

print(f"Target vector at ({target_x}, {target_y}) extracted.")
print(f"Similarity mask saved to similarity_mask.png (threshold: {threshold:.4f})")

handle.remove()
