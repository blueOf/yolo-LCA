from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from ultralytics.engine.predictor import BasePredictor


class _DummyDataset:
    def __init__(self):
        self.bs = 1
        self.mode = "image"
        self.count = 0
        self.source_type = SimpleNamespace(stream=False, screenshot=False, tensor=False, from_img=True)

    def __len__(self):
        return 1

    def __iter__(self):
        yield ["dummy.jpg"], [np.zeros((32, 32, 3), dtype=np.uint8)], [""]


class _DummyBackend:
    def __init__(self):
        self.pt = True
        self.triton = False
        self.ch = 3
        self.fp16 = False
        self.stride = 32
        self.imx = False
        self.dynamic = False
        self.model = nn.Sequential(nn.Conv2d(3, 3, 1))
        self.warmup_calls = 0

    def warmup(self, imgsz):
        self.warmup_calls += 1


class _DummyPredictor(BasePredictor):
    def setup_source(self, source, stride=None):
        self.imgsz = (32, 32)
        self.dataset = _DummyDataset()
        self.source_type = self.dataset.source_type
        self.vid_writer = {}

    def preprocess(self, im):
        return torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def inference(self, im, *args, **kwargs):
        return torch.zeros((1, 84, 1))

    def postprocess(self, preds, img, orig_imgs):
        return [preds]


def _new_predictor():
    p = _DummyPredictor(overrides={"save": False, "save_txt": False, "save_crop": False, "show": False, "verbose": False})
    p.model = _DummyBackend()
    p.device = torch.device("cpu")
    return p


def test_warmup_runs_without_hooks():
    p = _new_predictor()
    list(p.stream_inference(source="dummy"))
    assert p.model.warmup_calls == 1


def test_warmup_skipped_when_forward_hook_exists():
    p = _new_predictor()
    handle = p.model.model[0].register_forward_hook(lambda m, i, o: None)
    try:
        list(p.stream_inference(source="dummy"))
    finally:
        handle.remove()
    assert p.model.warmup_calls == 0


if __name__ == "__main__":
    test_warmup_runs_without_hooks()
    test_warmup_skipped_when_forward_hook_exists()
    print("predictor hook warmup tests passed")
