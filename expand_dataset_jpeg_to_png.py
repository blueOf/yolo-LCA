"""Generate JPEG variants (q10/q25/q50) into per-split q-images folders.

For each image in datasets/SVRDD/datasets_{n}/{train,val}/images, write
JPEG copies with qualities 10, 25, 50 into a sibling folder q-images.
Labels stay untouched (no label duplication). Default target is
datasets/SVRDD/datasets_1, but datasets list is configurable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install Pillow") from exc


def save_variants(
    image_path: Path,
    dst_q_images: Path,
    qualities: Sequence[int],
) -> int:
    """Save JPEG variants for a single image and copy matching labels.

    Returns the number of new images written.
    """

    stem = image_path.stem
    image = Image.open(image_path).convert("RGB")

    written = 0
    for q in qualities:
        out_img = dst_q_images / f"{stem}_q{q}.jpg"
        image.save(
            out_img, format="JPEG", quality=q, optimize=True, subsampling="4:2:0"
        )
        written += 1

    return written


def process_split(
    src_split: Path,
    dst_split: Path,
    qualities: Sequence[int],
    exts: Iterable[str] = (".jpg", ".jpeg", ".png"),
) -> tuple[int, int]:
    """Process a split directory; returns (images_processed, images_written)."""

    src_images = src_split / "images"
    dst_q_images = dst_split / "q-images"
    dst_q_images.mkdir(parents=True, exist_ok=True)

    imgs = [
        p for p in src_images.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]
    processed = 0
    written = 0

    for img_path in imgs:
        try:
            written += save_variants(img_path, dst_q_images, qualities)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed on {img_path}: {exc}")
            continue
        processed += 1

    return processed, written


def process_dataset(
    src_root: Path,
    dst_root: Path,
    datasets: Sequence[str],
    splits: Sequence[str],
    qualities: Sequence[int],
) -> None:
    for ds in datasets:
        for split in splits:
            src_split = src_root / ds / split
            dst_split = dst_root / ds / split
            if not src_split.exists():
                print(f"[WARN] Skip missing split: {src_split}")
                continue
            processed, written = process_split(src_split, dst_split, qualities)
            print(
                f"[OK] {ds}/{split}: processed {processed} images -> wrote {written} outputs"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JPEG variants for SVRDD datasets"
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("datasets/SVRDD"),
        help="Source datasets root",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["datasets_1"],
        help="Dataset folders to process",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], help="Splits to process"
    )
    parser.add_argument(
        "--qualities", nargs="+", type=int, default=[10, 25, 50], help="JPEG qualities"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        args.src_root,
        args.src_root,
        args.datasets,
        args.splits,
        args.qualities,
    )


if __name__ == "__main__":
    main()
