import torch
import torch.nn as nn
import time
import numpy as np
from ultralytics.nn.modules.block import JpegFusionModule


def benchmark_jpeg_fusion_module(
    batch_size=8,
    input_channels=3,
    output_channels=64,
    input_size=(640, 640),
    conv_channels=64,
    conv_stride=2,
    q1=10,
    q2=20,
    q3=50,
    warmup=10,
    iterations=100,
    device="cuda",
):
    """
    测试 JpegFusionModule 的前向传播速度

    Args:
        batch_size: 批次大小
        input_channels:  输入通道数（RGB=3）
        output_channels:  输出通道数
        input_size: 输入图像尺寸 (H, W)
        conv_channels: 中间卷积通道数
        conv_stride: 卷积步长
        q1, q2, q3: JPEG 质量参数
        warmup: 预热迭代次数
        iterations: 测试迭代次数
        device: 设备 ('cuda' 或 'cpu')
    """
    print("=" * 80)
    print(f"JpegFusionModule 前向传播速度测试")
    print("=" * 80)
    print(f"配置:")
    print(f"  Batch Size: {batch_size}")
    print(
        f"  Input:  ({batch_size}, {input_channels}, {input_size[0]}, {input_size[1]})"
    )
    print(f"  Output Channels: {output_channels}")
    print(f"  Conv Channels: {conv_channels}, Stride: {conv_stride}")
    print(f"  JPEG Qualities: q1={q1}, q2={q2}, q3={q3}")
    print(f"  Device: {device}")
    print(f"  Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 80)

    # 创建模块
    module = JpegFusionModule(
        input_channels=input_channels,
        output_channels=output_channels,
        conv_channels=conv_channels,
        conv_stride=conv_stride,
        q1=q1,
        q2=q2,
        q3=q3,
        subsampling=2,
    ).to(device)

    module.eval()  # 评估模式

    # 创建随机输入（模拟归一化后的图像 0-1）
    input_tensor = torch.rand(
        batch_size,
        input_channels,
        input_size[0],
        input_size[1],
        device=device,
        dtype=torch.float32,
    )

    print(f"Input shape: {input_tensor.shape}")
    print(f"Input dtype: {input_tensor.dtype}")
    print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    print(f"Input device: {input_tensor.device}")
    print("-" * 80)

    # 计算模块参数量
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"模块参数:")
    print(f"  Total params: {total_params: ,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Params size: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    print("-" * 80)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for i in range(warmup):
            if device == "cuda":
                torch.cuda.synchronize()
            _ = module(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            if (i + 1) % 5 == 0:
                print(f"  Warmup {i+1}/{warmup}")

    print("-" * 80)

    # 测试前向传播
    print(f"Testing forward pass ({iterations} iterations)...")
    times = []

    with torch.no_grad():
        for i in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output = module(input_tensor)

            if device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{iterations}")

    # 统计结果
    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    min_time = times.min()
    max_time = times.max()
    median_time = np.median(times)

    fps = 1.0 / mean_time if mean_time > 0 else float("inf")
    throughput = batch_size * fps  # images per second

    print("=" * 80)
    print("测试结果:")
    print("=" * 80)
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output. min():.3f}, {output.max():.3f}]")
    print("-" * 80)
    print(f"时间统计 (per forward pass):")
    print(f"  Mean:    {mean_time*1000:>8.3f} ms  (± {std_time*1000:.3f} ms)")
    print(f"  Median: {median_time*1000:>8.3f} ms")
    print(f"  Min:     {min_time*1000:>8.3f} ms")
    print(f"  Max:    {max_time*1000:>8.3f} ms")
    print("-" * 80)
    print(f"吞吐量:")
    print(f"  FPS (batch):        {fps: >8.2f} batches/sec")
    print(f"  Throughput (imgs):  {throughput:>8.2f} images/sec")
    print(f"  Latency per image:  {mean_time*1000/batch_size:>8.3f} ms/image")
    print("=" * 80)

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "fps": fps,
        "throughput": throughput,
        "output_shape": output.shape,
        "params": total_params,
    }


def compare_different_configs():
    """对比不同配置的性能"""
    print("\n\n")
    print("=" * 80)
    print("多配置对比测试")
    print("=" * 80)

    configs = [
        # (batch_size, input_size, conv_channels, name)
        (1, (640, 640), 64, "Single Image (640x640)"),
        (4, (640, 640), 64, "Small Batch (BS=4, 640x640)"),
        (8, (640, 640), 64, "Medium Batch (BS=8, 640x640)"),
        (8, (640, 640), 32, "Fewer Channels (BS=8, ch=32)"),
        (8, (640, 640), 128, "More Channels (BS=8, ch=128)"),
    ]

    results = []
    for batch_size, input_size, conv_channels, name in configs:
        print(f"\n{'='*80}")
        print(f"配置: {name}")
        print(f"{'='*80}")

        result = benchmark_jpeg_fusion_module(
            batch_size=batch_size,
            input_channels=3,
            output_channels=conv_channels,
            input_size=input_size,
            conv_channels=conv_channels,
            conv_stride=2,
            q1=10,
            q2=20,
            q3=50,
            warmup=5,
            iterations=50,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        results.append(
            {
                "name": name,
                "batch_size": batch_size,
                "input_size": input_size,
                "conv_channels": conv_channels,
                **result,
            }
        )

    # 打印对比表格
    print("\n\n")
    print("=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print(f"{'配置':<35} {'延迟(ms)':<12} {'FPS':<10} {'吞吐量(img/s)':<15}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<35} {r['mean_time']*1000:>10.2f}  {r['fps']:>8.2f}  {r['throughput']:>13.2f}"
        )
    print("=" * 80)


def test_with_real_image():
    """使用真实图像测试"""
    import cv2

    print("\n\n")
    print("=" * 80)
    print("真实图像测试")
    print("=" * 80)

    # 读取图像
    img_path = "datasets/SVRDD/datasets_1/train/images/0100220000130722112635631J2_1297660900_483426400.jpg"
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            print("使用随机生成的图像代替")
            img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        else:
            print(f"成功读取图像: {img_path}")
            print(f"原始尺寸: {img.shape}")
            # Resize to 640x640
            img = cv2.resize(img, (640, 640))
    except:
        print("使用随机生成的图像")
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HWC -> CHW, normalize to 0-1
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = img_tensor.to(device)

    print(f"输入张量:  {img_tensor.shape}, device: {img_tensor.device}")

    # 测试
    result = benchmark_jpeg_fusion_module(
        batch_size=1,
        input_channels=3,
        output_channels=64,
        input_size=(640, 640),
        conv_channels=64,
        conv_stride=2,
        q1=10,
        q2=20,
        q3=50,
        warmup=5,
        iterations=100,
        device=device,
    )

    # 保存输出可视化
    module = (
        JpegFusionModule(
            input_channels=3,
            output_channels=64,
            conv_channels=64,
            conv_stride=2,
            q1=10,
            q2=20,
            q3=50,
        )
        .to(device)
        .eval()
    )

    with torch.no_grad():
        output = module(img_tensor)

    print(f"\n输出特征图: {output.shape}")
    print(
        f"输出统计: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}"
    )


if __name__ == "__main__":
    # 基础测试
    benchmark_jpeg_fusion_module(
        batch_size=8,
        input_channels=3,
        output_channels=64,
        input_size=(640, 640),
        conv_channels=64,
        conv_stride=2,
        q1=10,
        q2=20,
        q3=50,
        warmup=10,
        iterations=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 多配置对比
    compare_different_configs()

    # 真实图像测试
    test_with_real_image()
