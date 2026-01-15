# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
import math
import random
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Albumentations,
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
    RandomHSV,
    RandomPerspective,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (
            self.use_segments and self.use_keypoints
        ), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for (
                im_file,
                lb,
                shape,
                segments,
                keypoint,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(
                f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}"
            )
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = (
                load_dataset_cache_file(cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(
                self.label_files + self.im_files
            )  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(
                f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = (
            (len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels
        )
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(
                f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}"
            )
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose(
                [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(
                resample_segments(segments, n=segment_resamples), axis=0
            )
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(
            bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized
        )
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class RandomHSV_MultiChannel(RandomHSV):
    """
    Randomly adjusts HSV channels of a multi-channel image (e.g. 12 channels).
    Applies the SAME random augmentation to every 3-channel block.
    """

    def __call__(self, labels):
        img = labels["img"]
        # Check if augmentations are enabled
        if self.hgain or self.sgain or self.vgain:
            r = (
                np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            )  # random gains

            # Create LUTs once
            dtype = img.dtype  # uint8
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            # Process in 3-channel chunks
            num_channels = img.shape[2]

            # Use original implementation if just 3 channels
            if num_channels == 3:
                return super().__call__(labels)

            if num_channels % 3 != 0:
                LOGGER.warning(
                    f"RandomHSV_MultiChannel: Image channels {num_channels} not divisible by 3. Skipping HSV."
                )
                return labels

            # Apply identical transformation to each 3-channel block
            for i in range(0, num_channels, 3):
                chunk = img[:, :, i : i + 3]

                # Convert to HSV (OpenCV expects BGR)
                # Note: cv2.split/merge might be slow, but safe
                hue, sat, val = cv2.split(cv2.cvtColor(chunk, cv2.COLOR_BGR2HSV))

                # Apply LUT
                hue = cv2.LUT(hue, lut_hue)
                sat = cv2.LUT(sat, lut_sat)
                val = cv2.LUT(val, lut_val)

                # Convert back to BGR and update
                im_hsv = cv2.merge((hue, sat, val))
                # cv2.cvtColor in newer versions might not like non-contiguous dst
                chunk_bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
                img[:, :, i : i + 3] = chunk_bgr

            # labels["img"] is updated in-place because it references the array
        return labels


class RandomPerspective_MultiChannel(RandomPerspective):
    """
    Applies RandomPerspective to multiple 3-channel blocks consistently.
    Ensures padding (borderValue) is (114,114,114) for EACH block.
    """

    def affine_transform(self, img, border):
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(
            -self.perspective, self.perspective
        )  # x perspective (about y)
        P[2, 1] = random.uniform(
            -self.perspective, self.perspective
        )  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # x shear (deg)
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        )  # x translation (pixels)
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]
        )  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        # Affine image
        if (
            (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any()
        ):  # image changed

            num_channels = img.shape[2]
            if num_channels > 3 and num_channels % 3 == 0:
                chunks = []
                for i in range(0, num_channels, 3):
                    chunk = img[:, :, i : i + 3]
                    if self.perspective:
                        chunk = cv2.warpPerspective(
                            chunk, M, dsize=self.size, borderValue=(114, 114, 114)
                        )
                    else:
                        chunk = cv2.warpAffine(
                            chunk, M[:2], dsize=self.size, borderValue=(114, 114, 114)
                        )
                    chunks.append(chunk)
                img = np.concatenate(chunks, axis=2)
            else:
                if self.perspective:
                    img = cv2.warpPerspective(
                        img, M, dsize=self.size, borderValue=(114, 114, 114)
                    )
                else:  # affine
                    img = cv2.warpAffine(
                        img, M[:2], dsize=self.size, borderValue=(114, 114, 114)
                    )

        return img, M, s


class Albumentations_MultiChannel:
    """
    Albumentations transformations for multi-channel (e.g., 12-channel) images.
    Applies the SAME transform parameters to each 3-channel block using ReplayCompose.
    """

    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A
            from ultralytics.utils.checks import check_version

            check_version(A.__version__, "1.0.3", hard=True)

            # Spatial transforms list to check if bboxes need processing
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }

            # Define transforms (Standard YOLO set)
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                # A.ImageCompression(quality_lower=75, p=0.0),
            ]

            self.contains_spatial = any(
                transform.__class__.__name__ in spatial_transforms for transform in T
            )

            # Use ReplayCompose to ensure consistency across channels
            if self.contains_spatial:
                self.transform = A.ReplayCompose(
                    T,
                    bbox_params=A.BboxParams(
                        format="yolo", label_fields=["class_labels"]
                    ),
                )
            else:
                self.transform = A.ReplayCompose(T)

            LOGGER.info(
                prefix
                + ", ".join(
                    f"{x}".replace("always_apply=False, ", "") for x in T if x.p
                )
            )

        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        if self.transform is None or random.random() > self.p:
            return labels

        img = labels["img"]
        cls = labels["cls"]

        h, w, c = img.shape
        if c % 3 != 0:
            return labels

        num_blocks = c // 3
        imgs = np.split(img, num_blocks, axis=2)

        import albumentations as A  # Ensure A is available

        # Prepare for first block
        if self.contains_spatial:
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(w, h)
            bboxes = labels["instances"].bboxes

            # Apply to first block
            # Note: Albumentations expects bboxes as list of lists usually, but numpy array might work depending on version
            # Standard YOLO passes numpy array? Let's check original.
            # Original: new = self.transform(image=im, bboxes=bboxes, class_labels=cls)
            # "bboxes" in YOLO are numpy.

            data = self.transform(image=imgs[0], bboxes=bboxes, class_labels=cls)

            # Replay for other blocks
            out_imgs = [data["image"]]
            for i in range(1, num_blocks):
                if self.contains_spatial:
                    # Pass dummy class_labels and bboxes to satisfy validation, ignore outputs
                    res = A.ReplayCompose.replay(
                        data["replay"], image=imgs[i], bboxes=[], class_labels=[]
                    )
                else:
                    res = A.ReplayCompose.replay(data["replay"], image=imgs[i])
                out_imgs.append(res["image"])

            # Reassemble
            if len(data["class_labels"]) > 0:
                labels["img"] = np.concatenate(out_imgs, axis=2)
                labels["cls"] = np.array(data["class_labels"])
                bboxes = np.array(data["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            # Non-spatial
            data = self.transform(image=imgs[0])
            out_imgs = [data["image"]]
            for i in range(1, num_blocks):
                res = A.ReplayCompose.replay(data["replay"], image=imgs[i])
                out_imgs.append(res["image"])

            labels["img"] = np.concatenate(out_imgs, axis=2)

        return labels


class Format_MultiChannel(Format):
    """
    Format class for Multi-Channel images (e.g. 12 channels).
    Maintains channel order for each 3-channel block when flipping (BGR <-> RGB).
    """

    def _format_img(self, img):
        """
        Formats an image for YOLO from a Numpy array to a PyTorch tensor.
        Handles multi-channel flip correctly.
        """
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)

        # img is HWC. Transpose to CHW.
        img = img.transpose(2, 0, 1)

        # Check probability to return BGR (if > bgr, we flip to RGB)
        # self.bgr is "probability to return BGR"
        # if r > self.bgr -> flip to RGB (if input is BGR)
        if random.uniform(0, 1) > self.bgr:
            c = img.shape[0]
            if c % 3 == 0:
                # Flip each 3-channel block (BGR -> RGB)
                for i in range(0, c, 3):
                    # Reverses the 3 channels in this block
                    img[i : i + 3] = img[i : i + 3][::-1]
            else:
                # Fallback to global flip
                img = img[::-1]

        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)


class YOLOJFDataset(YOLODataset):
    """
    Extended YOLO Dataset for loading multiple quality images.
    """

    def __init__(self, *args, data=None, task="detect", q_list=None, **kwargs):
        """Initializes the YOLOJFDataset with quality list for multi-image loading."""
        self.q_list = q_list if q_list is not None else [10, 25, 50]
        print(f"[YOLOJFDataset] Initialized with q_list: {self.q_list}")
        super().__init__(*args, data=data, task=task, **kwargs)

        # Calculate output channels
        self.output_channels = (len(self.q_list) + 1) * 3
        print(f"[YOLOJFDataset] Output channels: {self.output_channels}")

    def build_transforms(self, hyp=None):
        """Builds transforms and customizes for multi-channel support."""
        if self.augment:
            transforms = super().build_transforms(hyp)
            # transforms is a Compose object. transforms.transforms is the list.

            new_list = []
            for t in transforms.transforms:
                if isinstance(t, RandomHSV):
                    # Replace with MultiChannel version
                    new_list.append(
                        RandomHSV_MultiChannel(
                            hgain=t.hgain, sgain=t.sgain, vgain=t.vgain
                        )
                    )
                elif isinstance(t, Albumentations):
                    # Use MultiChannel Albumentations
                    new_list.append(Albumentations_MultiChannel(p=t.p))
                elif isinstance(t, Format):
                    # Use MultiChannel Format
                    new_list.append(
                        Format_MultiChannel(
                            bbox_format=t.bbox_format,
                            normalize=t.normalize,
                            return_mask=t.return_mask,
                            return_keypoint=t.return_keypoint,
                            return_obb=t.return_obb,
                            mask_ratio=t.mask_ratio,
                            mask_overlap=t.mask_overlap,
                            batch_idx=t.batch_idx,
                            bgr=t.bgr,
                        )
                    )
                else:
                    new_list.append(t)

            transforms.transforms = new_list
            return transforms
        else:
            # Validation/Test transforms also use Format. We need to patch it too!
            transforms = super().build_transforms(hyp)
            # Even in validation, Format is used.
            # Standard build_transforms returns Compose([LetterBox, Format])

            new_list = []
            for t in transforms.transforms:
                if isinstance(t, Format):
                    new_list.append(
                        Format_MultiChannel(
                            bbox_format=t.bbox_format,
                            normalize=t.normalize,
                            return_mask=t.return_mask,
                            return_keypoint=t.return_keypoint,
                            return_obb=t.return_obb,
                            mask_ratio=t.mask_ratio,
                            mask_overlap=t.mask_overlap,
                            batch_idx=t.batch_idx,
                            bgr=t.bgr,
                        )
                    )
                else:
                    new_list.append(t)
            transforms.transforms = new_list
            return transforms

    def get_q_image_path(self, original_path, q_value):
        """Get the path of quality-compressed image."""
        original_path = Path(original_path)

        # Replace 'images' with 'q-images'
        parent_dir = original_path.parent
        q_parent_dir = parent_dir.parent / "q-images"

        # Construct filename with quality suffix
        stem = original_path.stem
        ext = original_path.suffix
        q_filename = f"{stem}_q{q_value}{ext}"

        q_image_path = q_parent_dir / q_filename

        return q_image_path

    def load_image(self, i, rect_mode=True):
        """
        Loads multiple quality images and concatenates them.
        """
        # Load original image
        im, (h0, w0), (h, w) = super().load_image(i, rect_mode)

        # Store all images
        images = [im]

        # Get original image path
        original_path = self.im_files[i]

        # Load all quality versions
        for q_value in self.q_list:
            q_path = self.get_q_image_path(original_path, q_value)
            q_npy_path = q_path.with_suffix(".npy")

            q_im = None
            # Prioritize loading .npy
            if q_npy_path.exists():
                try:
                    q_im = np.load(str(q_npy_path))
                except Exception as e:
                    LOGGER.warning(f"Failed to load npy: {q_npy_path}, error: {e}")

            # Fallback to image loading if npy not found or failed
            if q_im is None:
                if q_path.exists():
                    q_im = cv2.imread(str(q_path))
                    if q_im is None:
                        LOGGER.warning(f"Failed to load:  {q_path}")
                        q_im = im.copy()
                    else:
                        # Resize to match original
                        if q_im.shape[:2] != (h, w):
                            q_im = cv2.resize(
                                q_im, (w, h), interpolation=cv2.INTER_LINEAR
                            )
                else:
                    LOGGER.warning(f"Not found: {q_path}")
                    q_im = im.copy()
            else:
                # Check resize for npy loaded data just in case
                if q_im.shape[:2] != (h, w):
                    q_im = cv2.resize(q_im, (w, h), interpolation=cv2.INTER_LINEAR)

            images.append(q_im)

        # Concatenate along channel dimension
        im_concat = np.concatenate(images, axis=2)

        return im_concat, (h0, w0), (h, w)


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes a dataset object for object detection tasks with optional specifications."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Add texts information for multi-modal model training."""
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]
        return labels

    def build_transforms(self, hyp=None):
        """Enhances data transformations with optional text augmentation for multi-modal training."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(
                -1, RandomLoadText(max_samples=min(self.data["nc"], 80), padding=True)
            )
        return transforms


class GroundingDataset(YOLODataset):
    """Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes a GroundingDataset for object detection, loading annotations from a specified JSON file."""
        assert (
            task == "detect"
        ), "`GroundingDataset` only support `detect` task for now!"
        self.json_file = json_file
        super().__init__(*args, task=task, data={}, **kwargs)

    def get_img_files(self, img_path):
        """The image files would be read in `get_labels` function, return empty list here."""
        return []

    def get_labels(self):
        """Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image."""
        labels = []
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f'{x["id"]:d}': x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(
            img_to_anns.items(), desc=f"Reading annotations {self.json_file}"
        ):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                cat_name = " ".join(
                    [img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]]
                )
                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
            lb = (
                np.array(bboxes, dtype=np.float32)
                if len(bboxes)
                else np.zeros((0, 5), dtype=np.float32)
            )
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        return labels

    def build_transforms(self, hyp=None):
        """Configures augmentations for training with optional text loading; `hyp` adjusts augmentation intensity."""
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            transforms.insert(-1, RandomLoadText(max_samples=80, padding=True))
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if (
            TORCHVISION_0_18
        ):  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = (
            args.cache is True or str(args.cache).lower() == "ram"
        )  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "WARNING ⚠️ Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = (
            str(args.cache).lower() == "disk"
        )  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [
            list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples
        ]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[
            i
        ]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if (
                im is None
            ):  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(
                [x[0] for x in self.samples]
            )  # identical hash
            nf, nc, n, samples = cache.pop(
                "results"
            )  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(
                    func=verify_image, iterable=zip(self.samples, repeat(self.prefix))
                )
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
