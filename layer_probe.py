"""
层探测工具：帮助确定哪些层适合提取特征
Layer Probe Tool: Help determine which layers are suitable for feature extraction

使用方法 / Usage:
    PYTHONPATH=. python layer_probe.py --model yolo11n.pt --image test1.jpg

功能 / Features:
    1. 显示所有层的结构和索引 / Show all layer structures and indices
    2. 打印每层的输出形状 / Print output shape of each layer
    3. 标记backbone/neck/head边界 / Mark backbone/neck/head boundaries
    4. 推荐常用的特征提取点 / Recommend common feature extraction points
"""

import argparse
import torch
from pathlib import Path


def analyze_model_structure(model_path: str, verbose: bool = True):
    """
    分析YOLO模型结构。
    Analyze YOLO model structure.

    Args:
        model_path: 模型路径 / Model path
        verbose: 是否打印详细信息 / Whether to print detailed info
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误：需要安装ultralytics / Error: ultralytics required")
        print("pip install ultralytics")
        return None

    print("=" * 80)
    print(f"分析模型 / Analyzing Model: {model_path}")
    print("=" * 80)

    # 加载模型 / Load model
    model = YOLO(model_path)
    train_model = model.model
    if hasattr(train_model, "model"):
        train_model = train_model.model

    total_layers = len(train_model)
    print(f"\n总层数 / Total Layers: {total_layers}\n")

    # 分析每一层 / Analyze each layer
    backbone_end = None
    neck_end = None

    print(f"{'Layer':<6} {'Type':<20} {'From':<10} {'Description':<40}")
    print("-" * 80)

    for i, layer in enumerate(train_model):
        layer_type = getattr(layer, "type", type(layer).__name__)
        layer_from = getattr(layer, "f", "?")

        # 判断阶段 / Determine stage
        stage = ""
        if "SPPF" in layer_type or "C2PSA" in layer_type:
            backbone_end = i
            stage = "← Backbone结束 / Backbone End"
        elif layer_type == "Detect":
            neck_end = i - 1
            stage = "← Head开始 / Head Start"

        # 判断是否为推荐的特征提取点
        # Determine if recommended extraction point
        is_recommended = False
        rec_reason = ""

        if layer_type in ["Conv", "C3k2", "C3"] and isinstance(layer_from, int) and layer_from == -1:
            # 可能是下采样层 / Possible downsampling layer
            if "Conv" in layer_type:
                is_recommended = True
                rec_reason = "可能的下采样点 / Possible downsample point"

        if "SPPF" in layer_type or "C2PSA" in layer_type:
            is_recommended = True
            rec_reason = "Backbone最终输出 / Backbone final output"

        # 检查是否为Detect头输入 / Check if Detect head input
        if i == total_layers - 1 and layer_type == "Detect":
            if isinstance(layer_from, list):
                rec_reason = f"Detect头输入来自层 / Detect inputs from: {layer_from}"
                is_recommended = True

        # 打印信息 / Print info
        marker = "★" if is_recommended else " "
        print(f"{marker} {i:<4} {layer_type:<20} {str(layer_from):<10} {rec_reason:<40} {stage}")

    print("-" * 80)

    # 总结推荐 / Summary recommendations
    print("\n" + "=" * 80)
    print("推荐的特征提取层 / Recommended Feature Extraction Layers")
    print("=" * 80)

    # 获取Detect头输入 / Get Detect head inputs
    detect_head = train_model[-1]
    if hasattr(detect_head, "f"):
        neck_outputs = detect_head.f
        print(f"\n1. Neck输出层（推荐用于对比学习）/ Neck outputs (recommended for contrastive learning):")
        print(f"   Layers: {list(neck_outputs)}")
        print(f"   用途 / Use: 多尺度融合特征，语义丰富 / Multi-scale fused features, semantically rich")

    # Backbone关键层 / Backbone key layers
    if backbone_end:
        potential_backbone = []
        for i in range(backbone_end + 1):
            layer_type = getattr(train_model[i], "type", "")
            if "Conv" in layer_type and i > 0:
                potential_backbone.append(i)

        # 选择3个代表性层 / Select 3 representative layers
        if len(potential_backbone) >= 3:
            step = len(potential_backbone) // 3
            selected = [potential_backbone[step], potential_backbone[2*step], backbone_end]
            print(f"\n2. Backbone输出层（推荐用于特征蒸馏）/ Backbone outputs (recommended for distillation):")
            print(f"   Layers: {selected}")
            print(f"   用途 / Use: 早期特征，细粒度信息 / Early features, fine-grained info")

    print("\n" + "=" * 80 + "\n")

    return model


def probe_layer_outputs(model, layer_indices: list[int], image_path: str = None):
    """
    探测指定层的实际输出形状。
    Probe actual output shapes of specified layers.

    Args:
        model: YOLO模型 / YOLO model
        layer_indices: 要探测的层 / Layers to probe
        image_path: 测试图像路径（可选）/ Test image path (optional)
    """
    print("=" * 80)
    print("探测层输出形状 / Probing Layer Output Shapes")
    print("=" * 80)

    train_model = model.model
    if hasattr(train_model, "model"):
        train_model = train_model.model

    features = {}
    handles = []

    def make_hook(layer_id):
        def hook_fn(module, input, output):
            features[layer_id] = output
        return hook_fn

    # 注册hooks / Register hooks
    print(f"\n注册hooks到层 / Registering hooks to layers: {layer_indices}")
    for layer_id in layer_indices:
        if layer_id < len(train_model):
            handle = train_model[layer_id].register_forward_hook(make_hook(layer_id))
            handles.append(handle)
        else:
            print(f"警告：层{layer_id}超出范围 / Warning: Layer {layer_id} out of range")

    # 准备输入 / Prepare input
    if image_path and Path(image_path).exists():
        print(f"\n使用真实图像 / Using real image: {image_path}")
        input_tensor = model(image_path, verbose=False)  # This triggers hooks
    else:
        print(f"\n使用随机输入 / Using random input: (1, 3, 640, 640)")
        dummy_input = torch.randn(1, 3, 640, 640)
        model.model.eval()
        with torch.no_grad():
            _ = train_model(dummy_input)

    # 打印结果 / Print results
    print("\n" + "-" * 80)
    print(f"{'Layer':<8} {'Shape':<30} {'Memory (MB)':<15} {'Dtype':<10}")
    print("-" * 80)

    total_memory = 0
    for layer_id in sorted(features.keys()):
        feat = features[layer_id]
        memory_mb = feat.numel() * feat.element_size() / (1024 ** 2)
        total_memory += memory_mb

        shape_str = "×".join(str(s) for s in feat.shape)
        print(f"{layer_id:<8} {shape_str:<30} {memory_mb:<15.2f} {str(feat.dtype):<10}")

    print("-" * 80)
    print(f"总内存占用 / Total Memory: {total_memory:.2f} MB")
    print("=" * 80 + "\n")

    # 清理 / Cleanup
    for handle in handles:
        handle.remove()

    return features


def recommend_layers(model):
    """
    根据模型结构推荐特征层。
    Recommend feature layers based on model structure.

    Args:
        model: YOLO模型 / YOLO model

    Returns:
        Dict with recommendations
    """
    train_model = model.model
    if hasattr(train_model, "model"):
        train_model = train_model.model

    recommendations = {
        "backbone_early": [],    # 早期backbone层 / Early backbone layers
        "backbone_late": [],     # 后期backbone层 / Late backbone layers
        "neck_outputs": [],      # Neck输出层 / Neck output layers
    }

    # 获取Detect头输入 / Get Detect head inputs
    detect_head = train_model[-1]
    if hasattr(detect_head, "f") and isinstance(detect_head.f, list):
        recommendations["neck_outputs"] = list(detect_head.f)

    # 分析backbone层 / Analyze backbone layers
    for i, layer in enumerate(train_model):
        layer_type = getattr(layer, "type", "")

        # 寻找下采样层 / Find downsampling layers
        if "Conv" in layer_type:
            if i < len(train_model) // 3:
                recommendations["backbone_early"].append(i)
            elif i < len(train_model) * 2 // 3:
                recommendations["backbone_late"].append(i)

        # Backbone结束标志 / Backbone end marker
        if "SPPF" in layer_type or "C2PSA" in layer_type:
            recommendations["backbone_late"].append(i)
            break

    return recommendations


def interactive_mode(model_path: str, image_path: str = None):
    """
    交互式探测模式。
    Interactive probing mode.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误：需要安装ultralytics / Error: ultralytics required")
        return

    # 加载模型 / Load model
    model = YOLO(model_path)

    while True:
        print("\n" + "=" * 80)
        print("交互式层探测工具 / Interactive Layer Probe Tool")
        print("=" * 80)
        print("\n选项 / Options:")
        print("  1 - 显示所有层结构 / Show all layer structure")
        print("  2 - 探测特定层输出 / Probe specific layer outputs")
        print("  3 - 获取推荐层 / Get recommended layers")
        print("  4 - 退出 / Exit")

        choice = input("\n选择 / Choose (1-4): ").strip()

        if choice == "1":
            analyze_model_structure(model_path)

        elif choice == "2":
            layers_input = input("输入层索引（逗号分隔）/ Enter layer indices (comma-separated): ").strip()
            try:
                layer_indices = [int(x.strip()) for x in layers_input.split(",")]
                probe_layer_outputs(model, layer_indices, image_path)
            except ValueError:
                print("错误：无效的层索引 / Error: Invalid layer indices")

        elif choice == "3":
            recs = recommend_layers(model)
            print("\n推荐配置 / Recommended Configurations:")
            print(f"\n1. 对比学习（Neck特征）/ Contrastive Learning (Neck features):")
            print(f"   feature_layers = {recs['neck_outputs']}")
            print(f"\n2. 特征蒸馏（Backbone特征）/ Feature Distillation (Backbone features):")
            backbone_all = recs['backbone_early'] + recs['backbone_late']
            if len(backbone_all) >= 3:
                selected = [backbone_all[len(backbone_all)//4], backbone_all[len(backbone_all)//2], backbone_all[-1]]
                print(f"   feature_layers = {selected}")
            print(f"\n3. 混合策略 / Mixed Strategy:")
            if recs['neck_outputs']:
                mixed = recs['backbone_late'][-2:] + recs['neck_outputs'][:2] if recs['backbone_late'] else recs['neck_outputs']
                print(f"   feature_layers = {mixed}")

        elif choice == "4":
            print("\n再见！/ Goodbye!")
            break

        else:
            print("无效选择 / Invalid choice")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description="YOLO模型层探测工具 / YOLO Model Layer Probe Tool"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="模型路径 / Model path (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="测试图像路径 / Test image path (optional)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="要探测的层，逗号分隔 / Layers to probe, comma-separated (e.g., '3,5,8')"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式 / Interactive mode"
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="只显示推荐层 / Only show recommended layers"
    )

    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ 错误：需要安装ultralytics / Error: ultralytics required")
        print("   安装方法 / Install: pip install ultralytics")
        return

    # 检查模型文件 / Check model file
    if not Path(args.model).exists():
        print(f"\n❌ 错误：模型文件不存在 / Error: Model file not found: {args.model}")
        return

    # 交互式模式 / Interactive mode
    if args.interactive:
        interactive_mode(args.model, args.image)
        return

    # 加载模型 / Load model
    model = YOLO(args.model)

    # 只显示推荐 / Only show recommendations
    if args.recommend:
        recs = recommend_layers(model)
        print("\n" + "=" * 80)
        print("推荐的特征提取层配置 / Recommended Feature Extraction Layer Configs")
        print("=" * 80)

        print("\n📊 配置1：Neck特征（适合对比学习）")
        print("    Config 1: Neck features (for contrastive learning)")
        print(f"    feature_layers = {recs['neck_outputs']}")

        print("\n📊 配置2：Backbone特征（适合特征蒸馏）")
        print("    Config 2: Backbone features (for feature distillation)")
        backbone_all = recs['backbone_early'] + recs['backbone_late']
        if len(backbone_all) >= 3:
            step = len(backbone_all) // 3
            selected = [backbone_all[step], backbone_all[2*step], backbone_all[-1]]
            print(f"    feature_layers = {selected}")

        print("\n📊 配置3：混合特征（适合多任务学习）")
        print("    Config 3: Mixed features (for multi-task learning)")
        if recs['neck_outputs']:
            mixed = (recs['backbone_late'][-2:] if recs['backbone_late'] else []) + recs['neck_outputs'][:2]
            print(f"    feature_layers = {mixed}")

        print("\n" + "=" * 80 + "\n")
        return

    # 分析结构 / Analyze structure
    analyze_model_structure(args.model)

    # 探测特定层 / Probe specific layers
    if args.layers:
        try:
            layer_indices = [int(x.strip()) for x in args.layers.split(",")]
            print(f"\n探测指定层 / Probing specified layers: {layer_indices}")
            probe_layer_outputs(model, layer_indices, args.image)
        except ValueError:
            print("❌ 错误：无效的层索引格式 / Error: Invalid layer index format")
    else:
        # 探测推荐层 / Probe recommended layers
        recs = recommend_layers(model)
        if recs['neck_outputs']:
            print(f"\n探测推荐的Neck输出层 / Probing recommended Neck output layers...")
            probe_layer_outputs(model, recs['neck_outputs'], args.image)


if __name__ == "__main__":
    # 如果没有参数，显示帮助
    # If no arguments, show help
    import sys
    if len(sys.argv) == 1:
        print("\n" + "=" * 80)
        print("YOLO层探测工具 / YOLO Layer Probe Tool")
        print("=" * 80)
        print("\n快速使用 / Quick Usage:")
        print("  python layer_probe.py --model yolo11n.pt --recommend")
        print("  python layer_probe.py --model yolo11n.pt --layers 3,5,8")
        print("  python layer_probe.py --model yolo11n.pt --interactive")
        print("\n详细帮助 / Detailed Help:")
        print("  python layer_probe.py --help")
        print("\n" + "=" * 80 + "\n")
    else:
        main()
