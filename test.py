"""
提取并保存批次中第一张图片的所有通道组
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics.data.dataset import YOLOJFDataset, YOLODataset
from ultralytics.utils import LOGGER, IterableSimpleNamespace
import cv2
from torch.utils.data import DataLoader


def create_default_hyp():
    """创建默认的超参数配置"""
    hyp_dict = {
        "mask_ratio": 4,
        "overlap_mask": True,
        "bgr": 0.0,
        "mosaic": 1,
        "mixup": 1,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.4,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "crop_fraction": 1.0,
    }

    return IterableSimpleNamespace(**hyp_dict)


def extract_and_save_channels(
    yaml_path, q_list=None, save_dir="channel_debug", num_samples=10
):
    """
    提取随机批次的随机图片的所有通道组并保存

    Args:
        yaml_path: YAML配置文件路径
        q_list: 质量因子列表
        save_dir: 保存目录
        num_samples: 随机采样次数
    """
    print("=" * 80)
    print("通道提取诊断 - 随机采样")
    print("=" * 80)

    # 加载配置
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        data_dict = yaml.safe_load(f)

    if q_list is None:
        q_list = [10, 25, 50]

    print(f"\n质量列表: {q_list}")
    print(f"随机采样次数: {num_samples}")

    # 创建数据集
    train_path = data_dict.get("train")
    if not Path(train_path).is_absolute():
        if "path" in data_dict:
            train_path = str(Path(data_dict["path"]) / train_path)

    hyp = create_default_hyp()

    dataset = YOLOJFDataset(
        img_path=train_path,
        imgsz=640,
        batch_size=16,
        augment=False,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="test:  ",
        data=data_dict,
        task="detect",
        q_list=q_list,
    )

    # 创建 DataLoader
    from torch.utils.data import DataLoader
    import random

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,  # 开启shuffle
        num_workers=0,
        collate_fn=YOLOJFDataset.collate_fn,
        pin_memory=False,
    )

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始随机采样并保存.. .\n")

    num_versions = len(q_list) + 1
    version_names = ["Original"] + [f"Q{q}" for q in q_list]

    # 随机采样10次
    for sample_idx in range(num_samples):
        # 随机选择一个批次
        batch = next(iter(dataloader))

        # 从批次中随机选择一张图片
        batch_size = batch["img"].shape[0]
        random_img_idx = random.randint(0, batch_size - 1)

        selected_img = batch["img"][random_img_idx]  # [C, H, W]

        print(f"样本 {sample_idx + 1}/{num_samples}:")
        print(f"  随机批次中的第 {random_img_idx + 1} 张图片")
        print(f"  Shape: {selected_img.shape}, dtype: {selected_img.dtype}")

        # 创建子目录
        sample_dir = save_dir / f"sample_{sample_idx: 02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 提取并保存每3个通道
        for v in range(num_versions):
            start_ch = v * 3
            end_ch = (v + 1) * 3

            # 提取3个通道 [3, H, W]
            img_3ch = selected_img[start_ch:end_ch, :, :]

            # 转换为 [H, W, 3]
            img_np = img_3ch.permute(1, 2, 0).cpu().numpy()

            # 转换为uint8
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # RGB -> BGR (cv2.imwrite需要)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 保存
            save_path = (
                sample_dir / f"channel_{start_ch}-{end_ch-1}_{version_names[v]}.png"
            )
            cv2.imwrite(str(save_path), img_bgr)

        print(f"  保存到: {sample_dir. name}/")
        print()

    print(f"\n所有图片已保存到:  {save_dir.absolute()}")
    print(f"共 {num_samples} 个样本，每个样本 {num_versions} 张图片")
    print("=" * 80)


if __name__ == "__main__":
    yaml_path = "datasets/SVRDD/svrdd_1.yaml"
    q_list = [10, 25, 50]

    extract_and_save_channels(
        yaml_path=yaml_path,
        q_list=q_list,
        save_dir="channel_debug_random",
        num_samples=10,  # 随机采样10次
    )
