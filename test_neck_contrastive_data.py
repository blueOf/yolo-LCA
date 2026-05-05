from unittest.mock import patch

import torch

from ultralytics.tools.neck_contrastive import NeckContrastiveYOLOData


def main():
    fake_data = {"train": "/abs/train/images", "val": "/abs/val/images", "nc": 2, "names": {0: "a", 1: "b"}}
    fake_cfg = object()

    with (
        patch("ultralytics.tools.neck_contrastive.data.check_det_dataset", return_value=fake_data) as p_check,
        patch("ultralytics.tools.neck_contrastive.data.get_cfg", return_value=fake_cfg) as p_cfg,
        patch("ultralytics.tools.neck_contrastive.data.build_yolo_dataset", return_value="dataset") as p_build_ds,
        patch("ultralytics.tools.neck_contrastive.data.build_dataloader", return_value="loader") as p_build_dl,
    ):
        data = NeckContrastiveYOLOData(data="dummy.yaml", imgsz=320, batch=4, workers=2, autodownload=False)
        ds = data.build_dataset(mode="train", stride=16)
        dl = data.build_dataloader(mode="train", stride=16)

    assert ds == "dataset"
    assert dl == "loader"
    p_check.assert_called_once_with("dummy.yaml", autodownload=False)
    p_cfg.assert_called_once()
    assert p_build_ds.call_count == 2
    p_build_ds.assert_any_call(
        cfg=fake_cfg,
        img_path="/abs/train/images",
        batch=4,
        data=fake_data,
        mode="train",
        rect=False,
        stride=16,
    )
    p_build_dl.assert_called_once_with(
        dataset="dataset",
        batch=4,
        workers=2,
        shuffle=True,
        rank=-1,
        drop_last=False,
        pin_memory=True,
    )

    batch = {"img": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8), "cls": torch.zeros(1, 1, dtype=torch.long)}
    out = NeckContrastiveYOLOData.preprocess_batch(batch)
    assert out["img"].dtype == torch.float32
    assert float(out["img"].max()) == 1.0
    print("neck_contrastive_data test passed")


if __name__ == "__main__":
    main()
