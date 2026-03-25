import torch
import copy
from ultralytics import YOLO


def convert_weights():
    # 1. 加载基础模型，获取对应的网络结构
    base_model_path = "/home/nkq/yolo11/runs/train/yolo11-sub_mean/weights/best.pt"
    print(f"Loading base model structure from: {base_model_path}")
    model = YOLO(base_model_path)

    # 2. 读取仅包含 state_dict 的错误权重文件
    state_dict_path = (
        "/home/nkq/yolo11/runs/train/neck_contrastive/neck_contrastive_epoch_300.pt"
    )
    print(f"Loading state_dict from: {state_dict_path}")
    state_dict = torch.load(state_dict_path)

    # 3. 将参数应用到这个结构中
    print("Applying weights to model...")
    model.model.load_state_dict(state_dict)

    # 4. 构建兼容 Ultralytics 的 checkpoint 字典
    print("Creating YOLO compatible checkpoint...")
    ckpt = {
        "epoch": 300,
        "model": copy.deepcopy(
            model.model
        ).half(),  # YOLO 默认保存为 FP16 格式以节省空间
    }

    # 5. 保存新的权重文件
    output_path = "/home/nkq/yolo11/runs/train/neck_contrastive/neck_contrastive_epoch_300_yolo.pt"
    torch.save(ckpt, output_path)
    print(f"Done! Compatible YOLO weights saved to: {output_path}")


if __name__ == "__main__":
    convert_weights()
