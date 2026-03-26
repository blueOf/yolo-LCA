from ultralytics import YOLO

model = YOLO("runs/train/yolo11-fast-dert-142-submean/weights/best.pt")

features = []


def hook_fn(module, input, output):
    # output 是原始特征图，不会被自动池化
    features.append(output)


# 注册到你想要的层（例如第 4 层）
layer = 7
handle = model.model.model[layer].register_forward_hook(hook_fn)

model("test1.jpg")

print(features[1])
