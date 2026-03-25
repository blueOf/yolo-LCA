import torch
from ultralytics import YOLO

from ultralytics.tools.neck_contrastive import NeckContrastiveTrainer
from ultralytics.tools.neck_contrastive.trainer import NeckContrastiveConfig, NeckContrastiveLoss


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

    # verify label-grouped vectors are contrasted against other labels and background with configurable weights
    base_kwargs = dict(
        sample_per_box=64,
        max_bg_samples_per_image=16,
        max_pos_samples_per_class=64,
        # negative margins force active push terms so weighted decomposition can be asserted deterministically
        margin=-1.0,
        inter_class_margin=-1.0,
        pull_weight=0.0,
    )
    loss_bg_only = NeckContrastiveLoss(
        **base_kwargs,
        inter_class_weight=0.0,
        bg_weight=1.0,
    )
    loss_inter_only = NeckContrastiveLoss(
        **base_kwargs,
        inter_class_weight=1.0,
        bg_weight=0.0,
    )
    loss_weighted = NeckContrastiveLoss(
        **base_kwargs,
        inter_class_weight=0.5,
        bg_weight=2.0,
    )
    feat = torch.zeros(1, 1, 4, 4)
    feat[0, 0, 0:2, 0:2] = 1.0  # class 0 region
    feat[0, 0, 2:4, 2:4] = 1.0  # class 1 region
    batch2 = {
        "batch_idx": torch.tensor([0, 0], dtype=torch.long),
        "cls": torch.tensor([[0], [1]], dtype=torch.long),
        "bboxes": torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]], dtype=torch.float32),
    }
    bg_term = loss_bg_only([feat], batch2)
    inter_term = loss_inter_only([feat], batch2)
    got = loss_weighted([feat], batch2)
    expected = 0.5 * inter_term + 2.0 * bg_term
    assert torch.allclose(got, expected, atol=1e-5), f"expected {expected.item()}, got {got.item()}"
    assert bg_term > 0 and inter_term > 0
    print("neck_contrastive test passed")


if __name__ == "__main__":
    main()
