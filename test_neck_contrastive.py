import torch
from ultralytics import YOLO

from ultralytics.tools.neck_contrastive import NeckContrastiveTrainer
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig


def main():
    torch.manual_seed(0)
    model = YOLO("yolo11n.pt")
    feature_layers = list(model.model.model[-1].f)
    optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-3)
    trainer = NeckContrastiveTrainer(
        model=model,
        config=NeckContrastiveConfig(feature_layers=feature_layers, contrastive_weight=0.3),
    )

    batch = {
        "img": torch.randn(2, 3, 64, 64),
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([[0], [1]], dtype=torch.long),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
    }
    before = [p.detach().clone() for p in model.model.parameters()]
    metrics = trainer.training_step(batch, optimizer)
    after = [p.detach() for p in model.model.parameters()]
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
