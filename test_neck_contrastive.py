import torch
import torch.nn as nn

from ultralytics.tools.neck_contrastive import NeckContrastiveTrainer
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig


class DummyYOLOLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Conv2d(3, 8, 3, padding=1, stride=2),
                nn.Conv2d(8, 16, 3, padding=1, stride=2),
                nn.Conv2d(16, 32, 3, padding=1, stride=2),
            ]
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, batch):
        x = batch["img"]
        for m in self.model:
            x = m(x)
        out = self.head(x)
        det_loss = out.pow(2).mean()
        det_items = torch.tensor([det_loss.detach(), 0.0, 0.0], device=det_loss.device)
        return det_loss, det_items


class DummyYOLOWrapper:
    def __init__(self):
        self.model = DummyYOLOLikeModel()

    def parameters(self):
        return self.model.parameters()


def main():
    torch.manual_seed(0)
    model = DummyYOLOWrapper()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = NeckContrastiveTrainer(
        model=model,
        config=NeckContrastiveConfig(feature_layers=[0, 1, 2], contrastive_weight=0.3),
    )

    batch = {
        "img": torch.randn(2, 3, 64, 64),
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([0, 1], dtype=torch.long),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
    }
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    metrics = trainer.training_step(batch, optimizer)
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    trainer.close()

    assert metrics["total_loss"].ndim == 0
    assert metrics["det_loss"].ndim == 0
    assert metrics["contrastive_loss"].ndim == 0
    assert torch.isfinite(metrics["total_loss"])
    assert torch.isfinite(metrics["contrastive_loss"])
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    print("neck_contrastive test passed")


if __name__ == "__main__":
    main()
