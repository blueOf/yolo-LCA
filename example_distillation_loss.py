"""
实战示例：多尺度特征蒸馏Loss
Practical Example: Multi-Scale Feature Distillation Loss

场景：使用教师模型的backbone特征指导学生模型学习更好的特征表达
Scenario: Use teacher model's backbone features to guide student model learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class DistillationLossConfig:
    """蒸馏Loss配置 / Distillation Loss Configuration"""
    student_layers: List[int]       # 学生模型提取层 / Student model layers
    teacher_layers: List[int]       # 教师模型提取层 / Teacher model layers
    distill_weight: float = 1.0     # 蒸馏loss权重 / Distillation loss weight
    temperature: float = 4.0        # 温度参数 / Temperature parameter
    use_detection_loss: bool = True # 是否使用检测loss / Use detection loss


class MultiScaleFeatureExtractor:
    """
    从教师和学生模型同时提取特征。
    Extract features from both teacher and student models simultaneously.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        student_layers: List[int],
        teacher_layers: List[int]
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_layers = list(student_layers)
        self.teacher_layers = list(teacher_layers)

        self._student_features: Dict[int, torch.Tensor] = {}
        self._teacher_features: Dict[int, torch.Tensor] = {}
        self._student_handles = []
        self._teacher_handles = []

        self._register_hooks()

    def _register_hooks(self):
        """注册hooks到学生和教师模型 / Register hooks to student and teacher models"""
        # 学生模型hooks / Student model hooks
        s_modules = self._get_modules(self.student_model)
        for layer_idx in self.student_layers:
            handle = s_modules[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, self._student_features)
            )
            self._student_handles.append(handle)

        # 教师模型hooks / Teacher model hooks
        t_modules = self._get_modules(self.teacher_model)
        for layer_idx in self.teacher_layers:
            handle = t_modules[layer_idx].register_forward_hook(
                self._make_hook(layer_idx, self._teacher_features)
            )
            self._teacher_handles.append(handle)

    @staticmethod
    def _get_modules(model):
        """获取模型的层序列 / Get model's layer sequence"""
        modules = getattr(model, "model", None)
        if modules is None:
            raise AttributeError("Model must have .model attribute")
        if hasattr(modules, "model"):
            modules = modules.model
        return modules

    @staticmethod
    def _make_hook(layer_idx: int, feature_dict: Dict[int, torch.Tensor]):
        """创建hook函数 / Create hook function"""
        def hook_fn(module, input, output):
            feature_dict[layer_idx] = output
        return hook_fn

    def clear(self):
        """清空特征字典 / Clear feature dictionaries"""
        self._student_features.clear()
        self._teacher_features.clear()

    def get_features(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        获取学生和教师的特征。
        Get student and teacher features.

        Returns:
            (student_features, teacher_features) 两个特征列表
        """
        student_feats = [
            self._student_features[idx] for idx in self.student_layers
        ]
        teacher_feats = [
            self._teacher_features[idx] for idx in self.teacher_layers
        ]
        return student_feats, teacher_feats

    def close(self):
        """移除所有hooks / Remove all hooks"""
        for handle in self._student_handles + self._teacher_handles:
            handle.remove()
        self._student_handles.clear()
        self._teacher_handles.clear()


class FeatureDistillationLoss(nn.Module):
    """
    特征蒸馏Loss：使学生模型的特征接近教师模型。
    Feature Distillation Loss: Make student features similar to teacher features.
    """

    def __init__(self, temperature: float = 4.0, weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def _align_spatial_size(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐学生和教师特征的空间尺寸。
        Align spatial sizes of student and teacher features.
        """
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            # 将教师特征插值到学生特征的尺寸
            # Interpolate teacher feature to student feature's size
            teacher_feat = F.interpolate(
                teacher_feat,
                size=student_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        return student_feat, teacher_feat

    def _align_channels(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐通道数（简化版本：使用全局平均池化）。
        Align channel counts (simplified: use global average pooling).

        注意：在实际应用中，可以使用1x1卷积进行通道对齐。
        Note: In practice, 1x1 conv can be used for channel alignment.
        """
        B, C_s, H, W = student_feat.shape
        C_t = teacher_feat.shape[1]

        if C_s != C_t:
            # 全局平均池化后比较 / Compare after global average pooling
            student_feat = F.adaptive_avg_pool2d(student_feat, 1).flatten(1)
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, 1).flatten(1)

        return student_feat, teacher_feat

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算蒸馏loss。
        Compute distillation loss.

        Args:
            student_features: 学生模型特征 / Student model features
            teacher_features: 教师模型特征 / Teacher model features
            batch: 训练batch / Training batch

        Returns:
            Scalar loss tensor
        """
        if len(student_features) != len(teacher_features):
            raise ValueError(
                f"Mismatch: {len(student_features)} student features vs "
                f"{len(teacher_features)} teacher features"
            )

        total_loss = 0.0

        for s_feat, t_feat in zip(student_features, teacher_features):
            # 对齐空间尺寸 / Align spatial sizes
            s_feat, t_feat = self._align_spatial_size(s_feat, t_feat)

            # 对齐通道数 / Align channel counts
            s_feat, t_feat = self._align_channels(s_feat, t_feat)

            # 应用温度缩放（对于概率分布式的特征）
            # Apply temperature scaling (for probability-like features)
            s_feat_scaled = s_feat / self.temperature
            t_feat_scaled = t_feat / self.temperature

            # 计算MSE loss
            # Compute MSE loss
            loss_i = F.mse_loss(s_feat_scaled, t_feat_scaled)
            total_loss += loss_i

        # 平均多个尺度的loss / Average loss over multiple scales
        avg_loss = total_loss / len(student_features)

        return avg_loss * self.weight


class DistillationTrainer:
    """
    知识蒸馏训练器。
    Knowledge Distillation Trainer.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        config: DistillationLossConfig
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.config = config

        # 解析底层模型 / Resolve underlying models
        self.train_student = self._resolve_model(student_model)
        self.train_teacher = self._resolve_model(teacher_model)

        # 冻结教师模型 / Freeze teacher model
        self.train_teacher.eval()
        for p in self.train_teacher.parameters():
            p.requires_grad_(False)

        # 创建特征提取器 / Create feature extractor
        self.extractor = MultiScaleFeatureExtractor(
            student_model=self.train_student,
            teacher_model=self.train_teacher,
            student_layers=config.student_layers,
            teacher_layers=config.teacher_layers,
        )

        # 创建蒸馏Loss / Create distillation loss
        self.criterion = FeatureDistillationLoss(
            temperature=config.temperature,
            weight=config.distill_weight,
        )

    @staticmethod
    def _resolve_model(model):
        """解析YOLO包装器 / Resolve YOLO wrapper"""
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, nn.Module) and hasattr(inner_model, "model"):
            return inner_model
        return model

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, torch.Tensor]:
        """
        执行一次训练步骤。
        Execute one training step.
        """
        # 学生模型训练模式，教师模型评估模式
        # Student in train mode, teacher in eval mode
        self.train_student.train()
        self.train_teacher.eval()

        optimizer.zero_grad()
        self.extractor.clear()

        # 计算检测Loss（可选）/ Compute detection loss (optional)
        det_loss = torch.tensor(0.0, device=batch["img"].device)
        if self.config.use_detection_loss:
            try:
                result = self.train_student(batch)
                if isinstance(result, tuple):
                    det_loss = result[0]
                else:
                    det_loss = result
                det_loss = det_loss.sum() if det_loss.ndim != 0 else det_loss
            except:
                # 如果检测loss失败，仍然做前向传播以捕获特征
                # If detection loss fails, still do forward to capture features
                _ = self.train_student(batch["img"])
        else:
            # 学生模型前向传播 / Student forward pass
            _ = self.train_student(batch["img"])

        # 教师模型前向传播（不计算梯度）
        # Teacher forward pass (no gradient)
        with torch.no_grad():
            _ = self.train_teacher(batch["img"])

        # 获取学生和教师特征 / Get student and teacher features
        student_feats, teacher_feats = self.extractor.get_features()

        # 计算蒸馏Loss / Compute distillation loss
        distill_loss = self.criterion(student_feats, teacher_feats, batch)

        # 组合总Loss / Combine total loss
        total_loss = det_loss + distill_loss

        # 反向传播（只更新学生模型）
        # Backpropagation (only update student model)
        total_loss.backward()
        optimizer.step()

        return {
            "total_loss": total_loss.detach(),
            "det_loss": det_loss.detach(),
            "distill_loss": distill_loss.detach(),
        }

    def close(self):
        """清理资源 / Clean up resources"""
        self.extractor.close()


# ===========================
# 使用示例 / Usage Example
# ===========================

def example_distillation_training():
    """
    特征蒸馏训练示例。
    Feature distillation training example.
    """
    print("=" * 70)
    print("多尺度特征蒸馏训练示例 / Multi-Scale Feature Distillation Training Example")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n错误：需要安装ultralytics / Error: ultralytics required")
        print("安装 / Install: pip install ultralytics")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备 / Device: {device}")

    # 1. 加载教师模型（大模型）/ Load teacher model (large model)
    print("\n1. 加载教师模型 / Loading teacher model...")
    teacher = YOLO("yolo11s.pt")  # 更大的模型 / Larger model
    teacher.to(device)
    print("   教师模型 / Teacher: yolo11s.pt")

    # 2. 加载学生模型（小模型）/ Load student model (small model)
    print("\n2. 加载学生模型 / Loading student model...")
    student = YOLO("yolo11n.pt")  # 更小的模型 / Smaller model
    student.to(device)
    print("   学生模型 / Student: yolo11n.pt")

    # 3. 配置要提取的层 / Configure layers to extract
    print("\n3. 配置提取层 / Configuring extraction layers...")

    # 对于相同架构的模型，可以使用相同的层索引
    # For models with same architecture, use same layer indices
    student_layers = [3, 5, 8]  # Backbone P3, P4, P5
    teacher_layers = [3, 5, 8]  # 对应的教师模型层 / Corresponding teacher layers

    print(f"   学生层 / Student layers: {student_layers}")
    print(f"   教师层 / Teacher layers: {teacher_layers}")

    # 4. 创建配置 / Create configuration
    config = DistillationLossConfig(
        student_layers=student_layers,
        teacher_layers=teacher_layers,
        distill_weight=1.0,
        temperature=4.0,
        use_detection_loss=False,  # 仅使用蒸馏loss / Only use distillation loss
    )

    # 5. 创建训练器 / Create trainer
    print("\n4. 创建训练器 / Creating trainer...")
    trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        config=config,
    )
    print("   训练器已创建 / Trainer created")

    # 6. 创建测试数据 / Create test data
    print("\n5. 创建测试数据 / Creating test data...")
    batch = {
        "img": torch.randn(2, 3, 640, 640, device=device),
        "cls": torch.randint(0, 80, (8, 1), device=device).float(),
        "bboxes": torch.rand(8, 4, device=device),
        "batch_idx": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device).float(),
    }

    # 7. 设置优化器（只优化学生模型）
    # Setup optimizer (only optimize student model)
    print("\n6. 设置优化器 / Setting up optimizer...")
    optimizer = torch.optim.Adam(
        student.model.parameters(),
        lr=0.001,
    )

    # 8. 执行训练步骤 / Execute training steps
    print("\n7. 执行训练步骤 / Executing training steps...")
    print("-" * 70)

    for step in range(5):
        metrics = trainer.training_step(batch, optimizer)

        print(f"\n   步骤 {step+1}/5 / Step {step+1}/5:")
        print(f"     总Loss / Total Loss:     {metrics['total_loss']:.6f}")
        print(f"     检测Loss / Det Loss:     {metrics['det_loss']:.6f}")
        print(f"     蒸馏Loss / Distill Loss: {metrics['distill_loss']:.6f}")

    print("\n" + "-" * 70)

    # 9. 清理 / Cleanup
    print("\n8. 清理资源 / Cleaning up...")
    trainer.close()
    print("   完成！/ Done!")

    print("\n" + "=" * 70)
    print("蒸馏训练测试成功！/ Distillation training test successful!")
    print("=" * 70)


# ===========================
# 另一个示例：区域感知Loss
# Another Example: Region-Aware Loss
# ===========================

class RegionAwareLoss(nn.Module):
    """
    区域感知Loss：根据bbox位置提取特征并计算损失。
    Region-Aware Loss: Extract features at bbox locations and compute loss.

    用途：强制模型在物体区域生成更有判别力的特征。
    Purpose: Force model to generate more discriminative features in object regions.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """将归一化的xywh转换为xyxy / Convert normalized xywh to xyxy"""
        xy = boxes[..., :2]
        wh = boxes[..., 2:]
        return torch.cat([xy - wh/2, xy + wh/2], dim=-1)

    def _extract_box_features(
        self,
        feat: torch.Tensor,
        boxes: torch.Tensor,
        batch_idx: torch.Tensor,
        image_idx: int
    ) -> torch.Tensor:
        """
        从特征图提取bbox区域的特征。
        Extract features from bbox regions in feature map.

        Args:
            feat: 特征图 [C, H, W] / Feature map
            boxes: 所有boxes [N, 4] (normalized xyxy) / All boxes
            batch_idx: 每个box的batch索引 [N] / Batch index for each box
            image_idx: 当前图像索引 / Current image index

        Returns:
            [M, C] 区域特征向量 / Region feature vectors
        """
        C, H, W = feat.shape
        device = feat.device

        # 找到属于当前图像的boxes / Find boxes belonging to current image
        mask = batch_idx == image_idx
        if not mask.any():
            return torch.empty((0, C), device=device)

        image_boxes = boxes[mask]  # [M, 4]
        box_features = []

        for box in image_boxes:
            x1, y1, x2, y2 = box
            # 转换到特征图坐标 / Convert to feature map coordinates
            x1_idx = int(torch.floor(x1 * W).item())
            y1_idx = int(torch.floor(y1 * H).item())
            x2_idx = int(torch.ceil(x2 * W).item())
            y2_idx = int(torch.ceil(y2 * H).item())

            # 裁剪到有效范围 / Clip to valid range
            x1_idx = max(0, min(x1_idx, W - 1))
            y1_idx = max(0, min(y1_idx, H - 1))
            x2_idx = max(x1_idx, min(x2_idx, W - 1))
            y2_idx = max(y1_idx, min(y2_idx, H - 1))

            # 提取区域特征并平均池化 / Extract region and average pool
            if x2_idx > x1_idx and y2_idx > y1_idx:
                region = feat[:, y1_idx:y2_idx+1, x1_idx:x2_idx+1]
                # 全局平均池化 / Global average pooling
                region_feat = region.mean(dim=(1, 2))  # [C]
                box_features.append(region_feat)

        if not box_features:
            return torch.empty((0, C), device=device)

        return torch.stack(box_features, dim=0)  # [M, C]

    def forward(
        self,
        features: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算区域感知loss。
        Compute region-aware loss.

        策略：对于同一个bbox，在不同尺度的特征应该保持一致。
        Strategy: For the same bbox, features at different scales should be consistent.
        """
        boxes = batch["bboxes"]  # [N, 4] normalized xywh
        batch_idx = batch["batch_idx"].view(-1).long()  # [N]

        # 转换为xyxy格式 / Convert to xyxy format
        boxes_xyxy = self._xywh_to_xyxy(boxes)

        bs = batch["img"].shape[0]
        total_loss = 0.0
        num_comparisons = 0

        # 对每张图像的每个尺度特征提取box区域
        # Extract box regions for each image at each scale
        for img_idx in range(bs):
            # 从每个尺度提取该图像的box特征
            # Extract box features from each scale for this image
            scale_box_features = []

            for feat in features:
                feat_i = feat[img_idx]  # [C, H, W]
                box_feats = self._extract_box_features(
                    feat_i, boxes_xyxy, batch_idx, img_idx
                )
                if box_feats.shape[0] > 0:
                    # L2归一化 / L2 normalize
                    box_feats = F.normalize(box_feats, dim=1)
                    scale_box_features.append(box_feats)

            # 计算不同尺度间的一致性 / Compute consistency across scales
            if len(scale_box_features) >= 2:
                for i in range(len(scale_box_features)):
                    for j in range(i + 1, len(scale_box_features)):
                        feats1 = scale_box_features[i]
                        feats2 = scale_box_features[j]

                        # 确保box数量一致 / Ensure same number of boxes
                        min_boxes = min(feats1.shape[0], feats2.shape[0])
                        if min_boxes > 0:
                            # 计算余弦距离 / Compute cosine distance
                            cosine_sim = (feats1[:min_boxes] * feats2[:min_boxes]).sum(dim=1).mean()
                            loss_ij = 1.0 - cosine_sim
                            total_loss += loss_ij
                            num_comparisons += 1

        if num_comparisons == 0:
            return feat.new_tensor(0.0) if features else torch.tensor(0.0)

        avg_loss = total_loss / num_comparisons
        return avg_loss * self.weight


# ===========================
# 调试工具函数
# Debug Utility Functions
# ===========================

def debug_layer_outputs(model, layer_indices: List[int]):
    """
    调试工具：查看指定层的输出形状。
    Debug utility: View output shapes of specified layers.

    Args:
        model: YOLO模型 / YOLO model
        layer_indices: 要检查的层索引 / Layer indices to check
    """
    print("\n" + "=" * 70)
    print("调试：查看层输出形状 / Debug: View Layer Output Shapes")
    print("=" * 70)

    features = {}

    def hook_fn(layer_id):
        def fn(module, input, output):
            features[layer_id] = output
            print(f"✓ Layer {layer_id:2d} captured: {output.shape}")
        return fn

    # 注册hooks / Register hooks
    train_model = getattr(model, "model", model)
    if hasattr(train_model, "model"):
        train_model = train_model.model

    handles = []
    for layer_id in layer_indices:
        handle = train_model[layer_id].register_forward_hook(hook_fn(layer_id))
        handles.append(handle)

    # 测试前向传播 / Test forward pass
    print(f"\n运行前向传播... / Running forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)

    model.eval()
    with torch.no_grad():
        try:
            _ = train_model(dummy_input)
        except Exception as e:
            print(f"前向传播错误 / Forward pass error: {e}")

    # 清理 / Cleanup
    for handle in handles:
        handle.remove()

    print("\n特征提取完成！/ Feature extraction completed!")
    print("=" * 70 + "\n")

    return features


def find_model_layers(model):
    """
    查看模型的所有层。
    View all layers in the model.

    Args:
        model: YOLO模型 / YOLO model
    """
    print("\n" + "=" * 70)
    print("模型结构 / Model Structure")
    print("=" * 70)

    train_model = getattr(model, "model", model)
    if hasattr(train_model, "model"):
        train_model = train_model.model

    print(f"\n总层数 / Total layers: {len(train_model)}\n")

    for i, layer in enumerate(train_model):
        layer_type = getattr(layer, "type", type(layer).__name__)
        layer_from = getattr(layer, "f", "?")
        print(f"Layer {i:2d}: {layer_type:15s} from={layer_from}")

    print("\n" + "=" * 70 + "\n")


# ===========================
# 主函数 / Main Function
# ===========================

def main():
    """主函数 / Main function"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        # 调试模式：查看模型结构和层输出
        # Debug mode: View model structure and layer outputs
        print("\n调试模式 / Debug Mode\n")

        try:
            from ultralytics import YOLO
        except ImportError:
            print("错误：需要安装ultralytics / Error: ultralytics required")
            return

        model = YOLO("yolo11n.pt")

        # 查看模型结构 / View model structure
        find_model_layers(model)

        # 调试特定层的输出 / Debug specific layer outputs
        debug_layer_outputs(model, [3, 5, 8, 16, 19, 22])

    else:
        # 正常模式：运行蒸馏训练示例
        # Normal mode: Run distillation training example
        example_distillation_training()


if __name__ == "__main__":
    main()
