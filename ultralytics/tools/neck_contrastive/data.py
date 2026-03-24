# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG_DICT


class NeckContrastiveYOLOData:
    """Build YOLO-format datasets/dataloaders for NeckContrastiveTrainer from a standard dataset YAML."""

    def __init__(
        self,
        data: str,
        imgsz: int = 640,
        batch: int = 16,
        workers: int = 8,
        cache: bool | str = False,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        autodownload: bool = True,
    ):
        self.batch = batch
        self.workers = workers
        self.data = check_det_dataset(data, autodownload=autodownload)
        self.args = get_cfg(
            DEFAULT_CFG_DICT,
            overrides={
                "task": "detect",
                "imgsz": imgsz,
                "rect": False,
                "cache": cache,
                "single_cls": single_cls,
                "classes": classes,
                "fraction": fraction,
                "workers": workers,
            },
        )

    def build_dataset(self, mode: str = "train", batch: int | None = None, stride: int = 32):
        """Create a YOLO dataset for train/val mode from the dataset YAML."""
        assert mode in {"train", "val"}, f"mode must be 'train' or 'val', not {mode!r}."
        img_path = self.data[mode]
        return build_yolo_dataset(
            cfg=self.args,
            img_path=img_path,
            batch=batch or self.batch,
            data=self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
        )

    def build_dataloader(
        self,
        mode: str = "train",
        batch: int | None = None,
        stride: int = 32,
        rank: int = -1,
        shuffle: bool | None = None,
        drop_last: bool = False,
        pin_memory: bool = True,
    ):
        """Create a dataloader yielding batches with keys expected by NeckContrastiveTrainer."""
        dataset = self.build_dataset(mode=mode, batch=batch, stride=stride)
        if shuffle is None:
            shuffle = mode == "train"
        return build_dataloader(
            dataset=dataset,
            batch=batch or self.batch,
            workers=self.workers if mode == "train" else self.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    @staticmethod
    def preprocess_batch(batch: dict[str, Any], device: torch.device | str | None = None) -> dict[str, Any]:
        """Move tensor values to device and normalize images to float [0,1]."""
        use_cuda_non_blocking = isinstance(device, torch.device) and device.type == "cuda"
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=use_cuda_non_blocking)
        batch["img"] = batch["img"].float() / 255
        return batch
