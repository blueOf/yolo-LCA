#!/usr/bin/env python3
"""
基于你提供的脚本，增加把推理可视化与结果保存到 ./debug_images 的功能。

每张输入图片会在 ./debug_images 下生成：
 - <stem>_debug.jpg        : 用原图（经 LetterBox 缩放到 imgsz）并在其上绘制检测框
 - <stem>_preds.txt        : 检测结果文本，每行: <cls> <conf> <x1> <y1> <x2> <y2>
 - <stem>_12ch.npy         : （可选）保存拼接后的 12 通道原始数组，便于调试

用法:
  python test_yolojfdataset_debug.py

注意:
 - 该脚本沿用你原有的多通道加载和预处理逻辑，仅在推理后把可视化与文本结果写出到 ./debug_images
 - 可视化使用原图的前三通道来画框（cv2 的 BGR 顺序），因为 cv2.imwrite 默认 BGR
"""
from __future__ import annotations
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils import LOGGER
import os

# 确保能导入本地 ultralytics（你之前加的路径）
sys.path.append("/home/nkq/yolov11-main")

DEBUG_DIR = Path("./debug_images")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def run_predictions():
    # 1. 模型路径
    model_path = "runs/train/yolov11-DWJF_datasets2/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    device = model.device

    # 2. 预处理
    imgsz = 640
    pre_transform = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)

    img_path = Path("datasets/SVRDD/train/images")
    img_files = sorted(list(img_path.glob("*.jpg")))[:100]

    if not img_files:
        print("No images found in", img_path)
        return

    print(f"Found {len(img_files)} images. Starting prediction...")

    q_list = [10, 25, 50]

    def load_multichannel_image(original_path, q_list=[10, 25, 50]):
        original_path = Path(original_path)
        im = cv2.imread(str(original_path))
        if im is None:
            return None

        h, w = im.shape[:2]
        images = [im]

        parent_dir = original_path.parent
        q_parent_dir = parent_dir.parent / "q-images"

        for q_value in q_list:
            stem = original_path.stem
            ext = original_path.suffix
            q_filename = f"{stem}_q{q_value}{ext}"
            q_path = q_parent_dir / q_filename
            q_npy_path = q_path.with_suffix(".npy")

            q_im = None
            if q_npy_path.exists():
                try:
                    q_im = np.load(str(q_npy_path))
                except Exception:
                    q_im = None

            if q_im is None:
                if q_path.exists():
                    q_im = cv2.imread(str(q_path))

            if q_im is None:
                # 缺失则复制原图
                q_im = im.copy()
            else:
                if q_im.shape[:2] != (h, w):
                    q_im = cv2.resize(q_im, (w, h), interpolation=cv2.INTER_LINEAR)

            images.append(q_im)

        # 最终 channels = 3 * (1 + len(q_list))，例如 12
        return np.concatenate(images, axis=2)

    # 切换模型到评估模式（你之前已有）
    model.model.eval()

    for i, img_file in enumerate(img_files):
        im0 = load_multichannel_image(img_file, q_list)  # HWC [H, W, C]
        if im0 is None:
            print(f"Failed to load {img_file}")
            continue

        # Save raw 12ch for debug (可选)
        stem = img_file.stem
        raw_npy_path = DEBUG_DIR / f"{stem}_12ch.npy"
        try:
            np.save(str(raw_npy_path), im0)
        except Exception as e:
            print("Warning: failed to save raw npy:", e)

        # Manual Pre-processing
        # LetterBox: 返回 resized image (H, W, C)
        im_resized = pre_transform(image=im0)  # -> [imgsz, imgsz, C]
        # For visualization we'll use the first 3 channels (original RGB/BGR)
        vis_img = im_resized[..., :3].copy()  # keep as BGR for cv2

        # HWC to CHW
        im_chw = im_resized.transpose(2, 0, 1)  # -> [C, H, W]
        im_chw = np.ascontiguousarray(im_chw)

        # To Tensor
        im_tensor = torch.from_numpy(im_chw).to(device)
        im_tensor = im_tensor.float() / 255.0

        # Add batch
        if im_tensor.ndim == 3:
            im_tensor = im_tensor.unsqueeze(0)  # [1, C, H, W]

        # Inference
        with torch.no_grad():
            preds = model.model(im_tensor)

        # NMS / postprocess
        det = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)[0]

        # Prepare output files
        debug_img_path = DEBUG_DIR / f"{stem}_debug.jpg"
        preds_txt_path = DEBUG_DIR / f"{stem}_preds.txt"

        # If no detections, still save the resized visualization
        if det is None or len(det) == 0:
            # save vis image
            try:
                cv2.imwrite(str(debug_img_path), vis_img)
            except Exception as e:
                print("Warning: failed to save debug image:", e)
            # write empty preds file
            with open(preds_txt_path, "w") as f:
                f.write("")
            print(
                f"[{i+1}/{len(img_files)}] {img_file.name}: 0 detections (saved debug image)"
            )
        else:
            # det is tensor (N,6) or (N, 6+) ; move to cpu numpy
            if isinstance(det, torch.Tensor):
                det_np = det.cpu().numpy()
            else:
                det_np = np.asarray(det)

            # draw boxes and save preds
            h_vis, w_vis = vis_img.shape[:2]
            lines = []
            for d in det_np:
                # Typical format: [x1, y1, x2, y2, conf, cls]
                if d.size < 6:
                    # unexpected format, skip
                    continue
                x1, y1, x2, y2 = map(int, map(round, d[0:4]))
                conf = float(d[4])
                cls_id = int(d[5])
                # clip coords
                x1 = max(0, min(x1, w_vis - 1))
                x2 = max(0, min(x2, w_vis - 1))
                y1 = max(0, min(y1, h_vis - 1))
                y2 = max(0, min(y2, h_vis - 1))

                # draw rectangle and label on vis_img
                color = (0, 255, 0)
                cv2.rectangle(
                    vis_img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA
                )
                label = f"{cls_id} {conf:.2f}"
                # text background
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    vis_img,
                    (x1, y1 - t_size[1] - 6),
                    (x1 + t_size[0] + 6, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    vis_img,
                    label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )

                lines.append(f"{cls_id} {conf:.6f} {x1} {y1} {x2} {y2}")

            # save debug image and preds txt
            try:
                cv2.imwrite(str(debug_img_path), vis_img)
            except Exception as e:
                print("Warning: failed to save debug image:", e)

            try:
                with open(preds_txt_path, "w") as f:
                    f.write("\n".join(lines))
            except Exception as e:
                print("Warning: failed to save preds txt:", e)

            print(
                f"[{i+1}/{len(img_files)}] {img_file.name}: {len(lines)} detections (saved debug image & preds)"
            )

    print("Done.")


if __name__ == "__main__":
    run_predictions()
