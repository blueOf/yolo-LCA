import torch
import argparse
from ultralytics import YOLO
from ultralytics.utils.torch_utils import time_sync


def benchmark(
    model_path, imgsz=640, batch=1, half=True, num_samples=200, warmup=20, device=""
):
    """
    测试 YOLO 模型的 FPS。
    """
    # 选择设备
    from ultralytics.utils.torch_utils import select_device

    device = select_device(device)
    print(f"Loading model: {model_path} on {device}...")

    # 加载模型
    model = YOLO(model_path)
    model.to(device)
    model.eval()  # 确保在 eval 模式

    # 核心优化 1: 融合 Conv2d + BatchNorm2d (YOLO 官方推理必做)
    print("Fusing layers...")
    model.fuse()

    # 模拟输入数据 (B, C, H, W)
    try:
        ch = next(model.model.parameters()).shape[1]
        if ch not in [1, 3, 4, 6, 9]:
            ch = 3
    except:
        ch = 3

    im = torch.zeros((batch, ch, imgsz, imgsz), device=device)

    if half and device.type != "cpu":
        model.model.half()
        im = im.half()

    # 核心优化 2: 禁用梯度计算
    with torch.no_grad():
        # 预热 (Warmup)
        print(f"Warming up {warmup} iterations...")
        for _ in range(warmup):
            model.model(im)

        # 同步并开始计时
        if device.type == "cuda":
            torch.cuda.synchronize()

        print(
            f"Testing FPS with {num_samples} iterations (imgsz={imgsz}, channels={ch})..."
        )
        t_start = time_sync()

        for _ in range(num_samples):
            model.model(im)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time_sync()

    # 计算统计数据
    total_time = t_end - t_start
    avg_latency = (total_time / num_samples) * 1000  # ms (per batch)
    fps = (batch * num_samples) / total_time

    print("\n" + "=" * 40)
    print(f"Benchmark Results for: {model_path}")
    print(f"Device: {device}")
    print(f"Image Size: {imgsz}x{imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Channels: {ch}")
    print(f"Precision: {'FP16' if half and device.type != 'cpu' else 'FP32'}")
    print(f"Average Batch Latency: {avg_latency:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO FPS Benchmark Tool")
    parser.add_argument("weights", type=str, help="Model weights path (.pt)")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference size (pixels)"
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num", type=int, default=300, help="Number of iterations for test"
    )
    parser.add_argument(
        "--warmup", type=int, default=30, help="Number of warmup iterations"
    )
    parser.add_argument("--full", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument(
        "--device", type=str, default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )

    args = parser.parse_args()
    benchmark(
        args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        half=not args.full,
        num_samples=args.num,
        warmup=args.warmup,
        device=args.device,
    )
