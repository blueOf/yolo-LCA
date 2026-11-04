"""
完整示例：使用Backbone层实现特征一致性Loss
Complete Example: Feature Consistency Loss Using Backbone Layers

此脚本展示如何创建一个自定义Loss，用于约束不同尺度特征之间的一致性。
This script demonstrates how to create a custom loss that enforces
consistency between features at different scales.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict


# ===========================
# 步骤1: 定义自定义Loss类
# Step 1: Define Custom Loss Class
# ===========================

class FeatureConsistencyLoss(nn.Module):
    """
    特征一致性损失：确保不同尺度的特征在相同空间位置具有一致的表达。
    Feature Consistency Loss: Ensures features at different scales have
    consistent representations at corresponding spatial locations.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-8):
        """
        初始化特征一致性Loss。
        Initialize feature consistency loss.

        Args:
            weight: Loss权重 / Loss weight
            eps: 数值稳定性的小常数 / Small constant for numerical stability
        """
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(
        self,
        features: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算特征一致性loss。
        Compute feature consistency loss.

        Args:
            features: 来自不同层的特征列表 [B, C, H, W]
                     List of features from different layers [B, C, H, W]
            batch: 训练batch（本例中未使用，但保持接口一致）
                  Training batch (not used in this example, but keeps interface consistent)

        Returns:
            Scalar loss tensor
        """
        if len(features) < 2:
            # 至少需要两个特征才能计算一致性
            # Need at least two features to compute consistency
            return torch.tensor(0.0, device=features[0].device)

        total_loss = 0.0
        num_pairs = 0

        # 对所有特征对计算一致性
        # Compute consistency for all feature pairs
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat1 = features[i]
                feat2 = features[j]

                # 将feat2上采样到feat1的尺寸
                # Upsample feat2 to feat1's size
                if feat1.shape[-2:] != feat2.shape[-2:]:
                    feat2_upsampled = F.interpolate(
                        feat2,
                        size=feat1.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    feat2_upsampled = feat2

                # 如果通道数不同，使用1x1卷积对齐（简化版本：使用自适应池化）
                # If different channel counts, align with 1x1 conv (simplified: use adaptive pooling)
                if feat1.shape[1] != feat2_upsampled.shape[1]:
                    # 简化处理：全局平均池化后比较
                    # Simplified: compare after global average pooling
                    feat1_pooled = F.adaptive_avg_pool2d(feat1, 1).flatten(1)
                    feat2_pooled = F.adaptive_avg_pool2d(feat2_upsampled, 1).flatten(1)

                    # L2归一化
                    # L2 normalization
                    feat1_norm = F.normalize(feat1_pooled, dim=1, eps=self.eps)
                    feat2_norm = F.normalize(feat2_pooled, dim=1, eps=self.eps)

                    # 余弦距离loss: 1 - cosine_similarity
                    # Cosine distance loss: 1 - cosine_similarity
                    cosine_sim = (feat1_norm * feat2_norm).sum(dim=1).mean()
                    loss_ij = 1.0 - cosine_sim
                else:
                    # 通道数相同，直接计算MSE
                    # Same channel count, directly compute MSE
                    loss_ij = F.mse_loss(feat1, feat2_upsampled)

                total_loss += loss_ij
                num_pairs += 1

        # 平均所有特征对的loss
        # Average loss over all feature pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs

        return total_loss * self.weight


# ===========================
# 步骤2: 创建特征提取器
# Step 2: Create Feature Extractor
# ===========================

class FeatureExtractor:
    """
    使用forward hooks从指定层提取特征。
    Extract features from specified layers using forward hooks.
    """

    def __init__(self, model: nn.Module, feature_layers: List[int]):
        """
        初始化特征提取器。
        Initialize feature extractor.

        Args:
            model: YOLO模型 / YOLO model
            feature_layers: 要提取特征的层索引列表 / List of layer indices to extract
        """
        self.model = model
        self.feature_layers = list(feature_layers)
        self._feature_dict: Dict[int, torch.Tensor] = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        """注册forward hooks / Register forward hooks"""
        # 获取底层模型 / Get underlying model
        modules = getattr(self.model, "model", None)
        if modules is None:
            raise AttributeError("Model must have .model attribute")

        # 处理嵌套结构 / Handle nested structure
        if hasattr(modules, "model"):
            modules = modules.model

        # 为每个指定层注册hook / Register hook for each specified layer
        for layer_idx in self.feature_layers:
            if layer_idx >= len(modules):
                raise IndexError(
                    f"Layer index {layer_idx} out of range. "
                    f"Model has {len(modules)} layers."
                )
            handle = modules[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        """创建hook函数 / Create hook function"""
        def hook_fn(module, input, output):
            self._feature_dict[layer_idx] = output
        return hook_fn

    def clear(self):
        """清空已捕获的特征 / Clear captured features"""
        self._feature_dict.clear()

    def get_features(self) -> List[torch.Tensor]:
        """获取特征列表 / Get feature list"""
        features = []
        for layer_idx in self.feature_layers:
            if layer_idx not in self._feature_dict:
                raise RuntimeError(
                    f"Layer {layer_idx} feature not captured. "
                    f"Available layers: {list(self._feature_dict.keys())}"
                )
            features.append(self._feature_dict[layer_idx])
        return features

    def close(self):
        """移除所有hooks / Remove all hooks"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ===========================
# 步骤3: 创建配置类
# Step 3: Create Configuration Class
# ===========================

@dataclass
class CustomLossConfig:
    """
    自定义Loss的配置参数。
    Configuration parameters for custom loss.
    """
    feature_layers: List[int]       # 要提取的层索引 / Layer indices to extract
    loss_weight: float = 0.1        # Loss权重 / Loss weight
    use_detection_loss: bool = True # 是否使用检测Loss / Whether to use detection loss
    freeze_head: bool = False       # 是否冻结检测头 / Whether to freeze detection head


# ===========================
# 步骤4: 创建训练器
# Step 4: Create Trainer
# ===========================

class CustomLossTrainer:
    """
    组合检测Loss和自定义Loss的训练器。
    Trainer that combines detection loss with custom loss.
    """

    def __init__(self, model: nn.Module, config: CustomLossConfig):
        self.model = model
        self.config = config

        # 获取可训练的底层模型 / Get trainable underlying model
        self.train_model = self._resolve_train_model(model)

        # 创建特征提取器 / Create feature extractor
        self.extractor = FeatureExtractor(
            self.train_model,
            config.feature_layers
        )

        # 创建自定义Loss / Create custom loss
        self.criterion = FeatureConsistencyLoss(weight=config.loss_weight)

        # 可选：冻结检测头 / Optional: freeze detection head
        if config.freeze_head:
            self._freeze_head()

    @staticmethod
    def _resolve_train_model(model):
        """解析YOLO包装器 / Resolve YOLO wrapper"""
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, nn.Module) and hasattr(inner_model, "model"):
            return inner_model
        return model

    def _freeze_head(self):
        """冻结检测头 / Freeze detection head"""
        modules = self.train_model.model
        if hasattr(modules, "__len__") and len(modules) > 0:
            head = modules[-1]
            for p in head.parameters():
                p.requires_grad_(False)
            print(f"检测头已冻结 / Detection head frozen")

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, torch.Tensor]:
        """
        执行一次训练步骤。
        Execute one training step.

        Args:
            batch: 包含'img', 'cls', 'bboxes', 'batch_idx'的字典
                  Dict containing 'img', 'cls', 'bboxes', 'batch_idx'
            optimizer: 优化器 / Optimizer

        Returns:
            包含各项loss的字典 / Dict containing various losses
        """
        self.train_model.train()
        optimizer.zero_grad()
        self.extractor.clear()

        # 计算检测Loss（可选） / Compute detection loss (optional)
        det_loss = torch.tensor(0.0, device=batch["img"].device)
        det_items = None

        if self.config.use_detection_loss:
            try:
                result = self.train_model(batch)
                if isinstance(result, tuple) and len(result) == 2:
                    det_loss, det_items = result
                else:
                    det_loss = result
                if not isinstance(det_loss, torch.Tensor):
                    raise TypeError("Detection loss must be a tensor")
                det_loss = det_loss.sum() if det_loss.ndim != 0 else det_loss
            except Exception as e:
                print(f"警告：检测Loss计算失败，仅使用自定义Loss / "
                      f"Warning: Detection loss computation failed, using custom loss only: {e}")
                # 仍然需要前向传播以触发hooks / Still need forward pass to trigger hooks
                _ = self.train_model(batch["img"])
        else:
            # 仅前向传播以触发hooks / Only forward pass to trigger hooks
            _ = self.train_model(batch["img"])

        # 获取提取的特征 / Get extracted features
        features = self.extractor.get_features()

        # 计算自定义Loss / Compute custom loss
        custom_loss = self.criterion(features, batch)

        # 组合总Loss / Combine total loss
        total_loss = det_loss + custom_loss

        # 反向传播和优化 / Backpropagation and optimization
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.detach(),
            "det_loss": det_loss.detach(),
            "custom_loss": custom_loss.detach(),
            "det_items": det_items,
        }

    def close(self):
        """清理资源 / Clean up resources"""
        self.extractor.close()


# ===========================
# 步骤5: 主训练函数（示例）
# Step 5: Main Training Function (Example)
# ===========================

def main():
    """主训练函数示例 / Main training function example"""
    print("=" * 60)
    print("自定义Loss训练示例 / Custom Loss Training Example")
    print("=" * 60)

    # 导入YOLO（需要安装ultralytics）
    # Import YOLO (requires ultralytics installation)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误：需要安装ultralytics库 / Error: ultralytics library required")
        print("安装方法 / Install via: pip install ultralytics")
        return

    # 设置GPU / Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备 / Using device: {device}")

    # 1. 加载预训练模型 / Load pretrained model
    print("\n1. 加载模型 / Loading model...")
    model = YOLO("yolo11n.pt")
    model.to(device)
    print(f"   模型已加载 / Model loaded: yolo11n.pt")

    # 2. 确定要提取特征的层 / Determine feature layers to extract
    print("\n2. 配置特征层 / Configuring feature layers...")

    # 方法A：手动指定backbone层
    # Method A: Manually specify backbone layers
    feature_layers = [3, 5, 8]  # P3/8, P4/16, P5/32
    print(f"   使用backbone层 / Using backbone layers: {feature_layers}")

    # 方法B：自动获取neck输出层（取消注释以使用）
    # Method B: Auto-get neck output layers (uncomment to use)
    # feature_layers = list(model.model.model[-1].f)
    # print(f"   使用neck层 / Using neck layers: {feature_layers}")

    # 3. 创建配置 / Create configuration
    config = CustomLossConfig(
        feature_layers=feature_layers,
        loss_weight=0.5,           # 自定义Loss权重 / Custom loss weight
        use_detection_loss=False,  # 本示例不使用检测Loss / Not using detection loss in this example
        freeze_head=False          # 不冻结检测头 / Don't freeze detection head
    )
    print(f"   配置 / Config:")
    print(f"     - 特征层 / Feature layers: {config.feature_layers}")
    print(f"     - Loss权重 / Loss weight: {config.loss_weight}")
    print(f"     - 使用检测Loss / Use detection loss: {config.use_detection_loss}")
    print(f"     - 冻结检测头 / Freeze head: {config.freeze_head}")

    # 4. 创建训练器 / Create trainer
    print("\n3. 创建训练器 / Creating trainer...")
    trainer = CustomLossTrainer(model=model, config=config)
    print("   训练器已创建 / Trainer created")

    # 5. 创建模拟数据进行测试 / Create dummy data for testing
    print("\n4. 创建测试数据 / Creating test data...")
    batch = {
        "img": torch.randn(2, 3, 640, 640, device=device),
        "cls": torch.randint(0, 80, (10, 1), device=device).float(),
        "bboxes": torch.rand(10, 4, device=device),  # normalized xywh
        "batch_idx": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], device=device).float(),
    }
    print(f"   Batch大小 / Batch size: {batch['img'].shape}")

    # 6. 设置优化器 / Setup optimizer
    print("\n5. 设置优化器 / Setting up optimizer...")
    optimizer = torch.optim.SGD(
        model.model.parameters(),
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005
    )
    print("   优化器已配置 / Optimizer configured")

    # 7. 执行几个训练步骤进行测试 / Execute a few training steps for testing
    print("\n6. 执行测试训练步骤 / Executing test training steps...")
    print("-" * 60)

    num_test_steps = 5
    for step in range(num_test_steps):
        metrics = trainer.training_step(batch, optimizer)

        print(f"   步骤 {step+1}/{num_test_steps} / Step {step+1}/{num_test_steps}:")
        print(f"     - 总Loss / Total Loss: {metrics['total_loss']:.6f}")
        print(f"     - 检测Loss / Det Loss: {metrics['det_loss']:.6f}")
        print(f"     - 自定义Loss / Custom Loss: {metrics['custom_loss']:.6f}")

    print("-" * 60)
    print("\n测试完成！/ Test completed!")

    # 8. 清理 / Cleanup
    print("\n7. 清理资源 / Cleaning up...")
    trainer.close()
    print("   完成！/ Done!")

    print("\n" + "=" * 60)
    print("要使用真实数据训练，请参考文档中的完整训练循环示例。")
    print("To train with real data, refer to the complete training loop in docs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
