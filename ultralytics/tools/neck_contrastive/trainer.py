# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils import DEFAULT_CFG_DICT


@dataclass
class NeckContrastiveConfig:
    """Config for neck feature contrastive training."""

    feature_layers: list[int]
    contrastive_weight: float = 0.1
    sample_per_box: int = 16
    max_bg_samples_per_image: int = 256
    max_pos_samples_per_class: int = 256
    margin: float = 0.2
    use_detection_loss: bool = False
    freeze_head: bool = True


class NeckFeatureExtractor:
    """Extract feature maps from selected model layers via forward hooks."""

    def __init__(self, model: torch.nn.Module, feature_layers: list[int]):
        if not feature_layers:
            raise ValueError("feature_layers must not be empty.")
        self.model = model
        self.feature_layers = list(feature_layers)
        self._feature_dict: dict[int, torch.Tensor] = {}
        self._handles = []
        self._register()

    def _register(self) -> None:
        modules = getattr(self.model, "model", None)
        if modules is None:
            raise AttributeError("model must provide a .model module list for layer hooks.")
        if not hasattr(modules, "__len__") and hasattr(modules, "model"):
            modules = modules.model
        if not hasattr(modules, "__len__"):
            raise TypeError("model.model must be an indexable module list (or expose nested .model).")

        max_idx = len(modules) - 1
        for layer in self.feature_layers:
            if layer < 0 or layer > max_idx:
                raise IndexError(f"Layer index {layer} out of range [0, {max_idx}].")
            handle = modules[layer].register_forward_hook(self._hook(layer))
            self._handles.append(handle)

    def _hook(self, layer: int):
        def fn(_, __, output):
            self._feature_dict[layer] = output

        return fn

    def clear(self) -> None:
        self._feature_dict.clear()

    def get_features(self) -> list[torch.Tensor]:
        features = []
        for layer in self.feature_layers:
            if layer not in self._feature_dict:
                raise RuntimeError(f"Feature for layer {layer} was not captured in current forward.")
            features.append(self._feature_dict[layer])
        return features

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class NeckContrastiveLoss(torch.nn.Module):
    """Pull same-class neck features together and push background features apart."""

    def __init__(
        self,
        sample_per_box: int = 16,
        max_bg_samples_per_image: int = 256,
        max_pos_samples_per_class: int = 256,
        margin: float = 0.2,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.sample_per_box = sample_per_box
        self.max_bg_samples_per_image = max_bg_samples_per_image
        self.max_pos_samples_per_class = max_pos_samples_per_class
        self.margin = margin
        self.eps = eps

    @staticmethod
    def _xywh_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
        """Convert normalized boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        xy = bboxes[..., :2]
        wh = bboxes[..., 2:]
        half = wh * 0.5
        return torch.cat((xy - half, xy + half), dim=-1)

    @staticmethod
    def _sample_rows(vectors: torch.Tensor, max_samples: int) -> torch.Tensor:
        """Randomly subsample rows to at most max_samples for memory/compute control."""
        if vectors.shape[0] <= max_samples:
            return vectors
        idx = torch.randperm(vectors.shape[0], device=vectors.device)[:max_samples]
        return vectors[idx]

    def _sample_box_vectors(self, feat_i: torch.Tensor, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
        """Extract (N,C) vectors from a box region on one feature map and sample them."""
        if x2 < x1 or y2 < y1:
            return feat_i.new_zeros((0, feat_i.shape[0]))
        region = feat_i[:, y1 : y2 + 1, x1 : x2 + 1].reshape(feat_i.shape[0], -1).transpose(0, 1)
        if region.shape[0] == 0:
            return region
        return self._sample_rows(region, self.sample_per_box)

    def _loss_single_level(self, feat: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute one-level contrastive loss with class pull-together and background push-away terms."""
        bs, _, h, w = feat.shape
        device = feat.device
        cls = batch["cls"].view(-1).long().to(device)
        batch_idx = batch["batch_idx"].view(-1).long().to(device)
        boxes = batch["bboxes"].to(device)

        class_vectors: dict[int, list[torch.Tensor]] = defaultdict(list)
        bg_vectors = []

        for i in range(bs):
            feat_i = feat[i]
            mask_fg = torch.zeros((h, w), device=device, dtype=torch.bool)
            select = batch_idx == i
            if select.any():
                cls_i = cls[select]
                xyxy = self._xywh_to_xyxy(boxes[select]).clamp_(0, 1)
                for box, c in zip(xyxy, cls_i):
                    x1 = int(torch.floor(box[0] * w).item())
                    y1 = int(torch.floor(box[1] * h).item())
                    x2 = int(torch.ceil(box[2] * w).item()) - 1
                    y2 = int(torch.ceil(box[3] * h).item()) - 1
                    x1 = min(max(x1, 0), w - 1)
                    y1 = min(max(y1, 0), h - 1)
                    x2 = min(max(x2, x1), w - 1)
                    y2 = min(max(y2, y1), h - 1)
                    mask_fg[y1 : y2 + 1, x1 : x2 + 1] = True
                    vectors = self._sample_box_vectors(feat_i, x1, y1, x2, y2)
                    if vectors.numel():
                        class_vectors[int(c.item())].append(vectors)

            bg = feat_i.permute(1, 2, 0)[~mask_fg]
            if bg.numel():
                bg_vectors.append(self._sample_rows(bg, self.max_bg_samples_per_image))

        pull_terms, push_terms = [], []
        bg_pool = torch.cat(bg_vectors, dim=0) if bg_vectors else None
        if bg_pool is not None:
            bg_pool = F.normalize(bg_pool, dim=1, eps=self.eps)

        for vectors_list in class_vectors.values():
            vectors = torch.cat(vectors_list, dim=0)
            vectors = self._sample_rows(vectors, self.max_pos_samples_per_class)
            vectors = F.normalize(vectors, dim=1, eps=self.eps)
            if vectors.shape[0] >= 2:
                sim = vectors @ vectors.transpose(0, 1)
                mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
                pull_terms.append(1.0 - sim[mask].mean())
            if bg_pool is not None and vectors.shape[0] > 0:
                neg_sim = vectors @ bg_pool.transpose(0, 1)
                push_terms.append(F.relu(neg_sim - self.margin).mean())

        if not pull_terms and not push_terms:
            return feat.sum() * 0.0

        loss = feat.new_tensor(0.0)
        if pull_terms:
            loss = loss + torch.stack(pull_terms).mean()
        if push_terms:
            loss = loss + torch.stack(push_terms).mean()
        return loss

    def forward(self, features: list[torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = [self._loss_single_level(feat, batch) for feat in features]
        if not losses:
            raise ValueError("No features were provided for contrastive loss.")
        return torch.stack(losses).mean()


class NeckContrastiveTrainer:
    """Training helper for combining detection loss with neck contrastive loss."""

    def __init__(self, model: torch.nn.Module, config: NeckContrastiveConfig):
        self.model = model
        self.config = config
        self.train_model = self._resolve_train_model(model)
        self.extractor = NeckFeatureExtractor(self.train_model, config.feature_layers)
        self.criterion = NeckContrastiveLoss(
            sample_per_box=config.sample_per_box,
            max_bg_samples_per_image=config.max_bg_samples_per_image,
            max_pos_samples_per_class=config.max_pos_samples_per_class,
            margin=config.margin,
        )
        self._apply_head_freeze()

    @staticmethod
    def _resolve_train_model(model: torch.nn.Module) -> torch.nn.Module:
        """Resolve trainable nn.Module for both raw models and Ultralytics YOLO wrappers."""
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, torch.nn.Module) and hasattr(inner_model, "model"):
            return inner_model
        return model

    @staticmethod
    def _normalize_det_items(det_items: Any) -> Any:
        """Detach tensor-like loss items for logging while preserving non-tensor structures."""
        if isinstance(det_items, torch.Tensor):
            return det_items.detach()
        if isinstance(det_items, (tuple, list)):
            return [x.detach() if isinstance(x, torch.Tensor) else x for x in det_items]
        return det_items

    def _ensure_loss_hypers(self) -> None:
        """Ensure detection-loss hyperparameters exist when using checkpoint-loaded wrapper models."""
        args = getattr(self.train_model, "args", None)
        if args is None:
            return
        defaults = {k: DEFAULT_CFG_DICT[k] for k in ("box", "cls", "dfl")}
        if isinstance(args, dict):
            self.train_model.args = SimpleNamespace(**{**defaults, **args})
            return
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)

    def _ensure_trainable_params(self) -> None:
        """Unfreeze parameters when loaded checkpoints have all grads disabled."""
        params = list(self.train_model.parameters())
        if params and not any(p.requires_grad for p in params):
            for p in params:
                p.requires_grad_(True)
            self._apply_head_freeze()

    def _apply_head_freeze(self) -> None:
        """Freeze detection head parameters when requested."""
        if not self.config.freeze_head:
            return
        modules = getattr(self.train_model, "model", None)
        if modules is None or not hasattr(modules, "__len__") or len(modules) == 0:
            return
        head = modules[-1]
        for p in head.parameters():
            p.requires_grad_(False)

    def training_step(
        self, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> dict[str, torch.Tensor]:
        self.train_model.train()
        self._ensure_trainable_params()
        optimizer.zero_grad(set_to_none=True)
        self.extractor.clear()
        det_loss = None
        det_items = None
        if self.config.use_detection_loss:
            self._ensure_loss_hypers()
            det_loss, det_items = self.train_model(batch)
            if not isinstance(det_loss, torch.Tensor):
                raise TypeError("Model training forward must return det_loss as a torch.Tensor.")
            if det_loss.ndim != 0:
                warnings.warn("Model returned non-scalar det_loss; reducing with sum() to obtain a scalar.", stacklevel=2)
                det_loss = det_loss.sum()
            if not det_loss.requires_grad:
                raise RuntimeError(
                    "Model forward must return a differentiable training loss tensor (requires_grad=True). "
                    "Check that model parameters are trainable and training mode is enabled."
                )
        else:
            _ = self.train_model(batch["img"])
        features = self.extractor.get_features()
        contrastive_loss = self.criterion(features, batch)
        if det_loss is None:
            det_loss = contrastive_loss.new_zeros(())
        total_loss = det_loss + self.config.contrastive_weight * contrastive_loss
        total_loss.backward()
        optimizer.step()
        return {
            "total_loss": total_loss.detach(),
            "det_loss": det_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "det_items": self._normalize_det_items(det_items),
        }

    def close(self) -> None:
        self.extractor.close()
