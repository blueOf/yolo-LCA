import torch
import time
import argparse
from ultralytics import YOLO

def test_fps(weights, device_id, imgsz=640, num_rounds=300, warmup=30):
    # 1. 指定设备并加载模型
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLO(weights)
    model.to(device)
    model.model.eval()
    
    # 2. 准备数据
    # YOLOv8+ 默认输入是 B, C, H, W (3通道)
    im = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    # 3. 预热 (Warmup)
    print(f"Warming up for {warmup} rounds...")
    with torch.no_grad():
        for _ in range(warmup):
            model.model(im)
    
    # 4. 正式测试
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"Starting benchmark for {num_rounds} rounds...")
    t_start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_rounds):
            model.model(im)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    # 5. 计算指标
    total_time = t_end - t_start
    latency = (total_time / num_rounds) * 1000 # ms
    fps = num_rounds / total_time
    
    print("\n" + "="*30)
    print(f"Model: {weights}")
    print(f"Device: {device}")
    print(f"Latency: {latency:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=str, help="path to weights file")
    parser.add_argument("--device", type=str, default="0", help="cuda device id")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    
    test_fps(args.weights, args.device, args.imgsz)
