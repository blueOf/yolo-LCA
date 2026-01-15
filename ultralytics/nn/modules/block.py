# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "SqueezeExcite",
    "ConvBnAct",
    "DepthwiseSeparableConv",
    "InvertedResidual",
    "CondConvResidual",
    "EdgeResidual",
    "UniversalInvertedResidual",
    "MobileAttention",
    "CKAS_DAF",
    "NonLocalBlockND",
    "C3k2_HKA",
    "C2PSA_1",
    "JpegFusionModule",
    "JpegFusionModule12C",
)

# region
##  ---------------------------------------------------修改------------------------------------------------##
from timm.layers import (
    create_conv2d,
    DropPath,
    create_act_layer,
    create_aa,
    to_2tuple,
    LayerType,
    ConvNormAct,
    get_norm_act_layer,
    MultiQueryAttention2d,
    Attention2d,
)
from typing import Callable, Dict, Optional, Type
from PIL import Image
from typing import Optional, Union
from io import BytesIO
import numpy as np
import nvjpeg
import cv2


ModuleType = Type[nn.Module]

ArrayOrTensor = Union[np.ndarray, "torch.Tensor"]


def reencode_via_jpeg_return_tensor(
    img: ArrayOrTensor,
    quality: int = 75,
    subsampling: int = 2,
    torch_device: Optional[str] = None,
) -> "torch.Tensor":
    """
    将输入 (3,H,W) 或 (H,W,3) 的图像先编码为 JPEG（给定 quality），再从 JPEG 解码并返回 torch.Tensor。
    返回: torch.Tensor shape=(3,H,W), dtype=torch.uint8（位于 torch_device 如果提供）
    """
    # convert torch -> numpy
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    # normalize dims: allow (C,H,W) or (H,W,C)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))  # -> H,W,3
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError(
            "Input must have shape (3,H,W) or (H,W,3) with 3 channels (RGB)."
        )

    # convert to uint8
    if np.issubdtype(arr.dtype, np.floating):
        mx = arr.max() if arr.size > 0 else 0.0
        if mx <= 1.0:
            arr = (arr * 255.0).round()
        else:
            arr = arr.round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # PIL image
    pil_img = Image.fromarray(arr, mode="RGB")

    # save as JPEG to memory
    jbuf = BytesIO()
    pil_img.save(
        jbuf,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=False,
        progressive=False,
    )
    jbuf.seek(0)

    # read back and convert
    pil_from_jpeg = Image.open(jbuf).convert("RGB")
    arr_jpeg = np.array(pil_from_jpeg)  # H,W,3 uint8

    # to torch tensor C,H,W uint8
    t = torch.from_numpy(np.transpose(arr_jpeg, (2, 0, 1))).to(dtype=torch.uint8)
    if torch_device:
        t = t.to(torch_device)

    jbuf.close()
    return t


# class JpegFusionModule(nn.Module):
#    """
#    img -> q1 -> conv ---\
#    img -> q2 -> conv ---- concat -> 1x1 conv ----\
#    img -> q3 -> conv ---------------------------- concat -> 1x1 conv ----\
#    img -> origin -> conv --------------------------------------------- concat -> 1x1 conv -> out
#    """

#    def __init__(
#        self,
#        input_channels: int,
#        output_channels: int,
#        conv_channels: int = 64,
#        conv_stride: int = 2,
#        q1: int = 10,
#        q2: int = 20,
#        q3: int = 50,
#        subsampling: int = 2,
#    ):
#        super().__init__()
#        self.q1 = int(q1)
#        self.q2 = int(q2)
#        self.q3 = int(q3)
#        self.subsampling = int(subsampling)
#        self.conv_stride = int(conv_stride)

#        # 四路分支 conv：用你的 Conv，k=3, s=conv_stride
#        self.conv_q1 = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)
#        self.conv_q2 = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)
#        self.conv_q3 = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)
#        self.conv_ori = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)

#        # 1x1 融合 conv（也用你的 Conv，k=1）
#        self.fuse_1_2 = Conv(2 * conv_channels, conv_channels, k=1, s=1)
#        self.fuse_prev_3 = Conv(2 * conv_channels, conv_channels, k=1, s=1)
#        self.fuse_prev_ori_out = Conv(
#            2 * conv_channels, output_channels, k=1, s=1, act=False
#        )

#    @torch.no_grad()
#    def _jpeg_batch(self, x: torch.Tensor, quality: int) -> torch.Tensor:
#        if x.ndim != 4:
#            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}")
#        b, c, h, w = x.shape
#        device = x.device

#        if x.dtype == torch.uint8:
#            x_u8 = x
#        else:
#            xf = x.detach()
#            if xf.max() <= 1.0:
#                xf = xf * 255.0
#            x_u8 = xf.round().clamp(0, 255).to(torch.uint8)

#        outs = []
#        for i in range(b):
#            t_u8 = reencode_via_jpeg_return_tensor(
#                x_u8[i],
#                quality=quality,
#                subsampling=self.subsampling,
#                torch_device=None,
#            )
#            outs.append(t_u8)

#        y_u8 = torch.stack(outs, dim=0)  # CPU uint8
#        y = y_u8.to(device=device, dtype=torch.float32) / 255.0
#        return y

#    def forward(self, img):
#        if img.ndim != 4:
#            raise ValueError(f"Expected img (B,C,H,W), got {tuple(img.shape)}")
#        if img.shape[1] != 3:
#            raise ValueError(
#                f"This implementation assumes RGB input (C==3) for JPEG branches, got C={img.shape[1]}."
#            )

#        # origin 分支（不做 JPEG）
#        x_ori = img if img.dtype.is_floating_point else img.float() / 255.0
#        f_ori = self.conv_ori(x_ori)

#        # JPEG 分支
#        x1 = self._jpeg_batch(img, quality=self.q1)
#        x2 = self._jpeg_batch(img, quality=self.q2)
#        x3 = self._jpeg_batch(img, quality=self.q3)

#        f1 = self.conv_q1(x1)
#        f2 = self.conv_q2(x2)
#        f3 = self.conv_q3(x3)

#        f_1_2 = self.fuse_1_2(torch.cat([f1, f2], dim=1))
#        f_1_2_3 = self.fuse_prev_3(torch.cat([f_1_2, f3], dim=1))
#        out = self.fuse_prev_ori_out(torch.cat([f_1_2_3, f_ori], dim=1))
#        return out


class JpegFusionModule(nn.Module):
    """
    img -> q1 -> conv ---\
    img -> q2 -> conv ---- concat -> 1x1 conv ----\
    img -> q3 -> conv ---------------------------- concat -> 1x1 conv ----\
    img -> origin -> conv --------------------------------------------- concat -> 1x1 conv -> out
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv_channels: int = 64,
        conv_stride: int = 2,
        q1: int = 10,
        q2: int = 20,
        q3: int = 50,
        subsampling: int = 2,
    ):
        super().__init__()
        self.q1 = int(q1)
        self.q2 = int(q2)
        self.q3 = int(q3)
        self.subsampling = int(subsampling)
        self.conv_stride = int(conv_stride)

        # 四路分支 conv：用你的 Conv，k=3, s=conv_stride
        self.conv_q1 = Conv(input_channels, conv_channels // 4, k=3, s=self.conv_stride)
        self.conv_q2 = Conv(input_channels, conv_channels // 3, k=3, s=self.conv_stride)
        self.conv_q3 = Conv(input_channels, conv_channels // 2, k=3, s=self.conv_stride)
        self.conv_ori = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)

        # 1x1 融合 conv（也用你的 Conv，k=1）
        self.fuse_1_2 = Conv(
            (conv_channels // 4 + conv_channels // 3), conv_channels // 2, k=1, s=1
        )
        self.fuse_prev_3 = Conv(
            (conv_channels // 2 + conv_channels // 2), conv_channels, k=1, s=1
        )
        self.fuse_prev_ori_out = Conv(
            2 * conv_channels, output_channels, k=1, s=1, act=False
        )

    @torch.no_grad()
    def _jpeg_batch(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        device = x.device

        if x.dtype == torch.uint8:
            x_u8 = x
        else:
            xf = x.detach()
            if xf.max() <= 1.0:
                xf = xf * 255.0
            x_u8 = xf.round().clamp(0, 255).to(torch.uint8)

        outs = []
        for i in range(b):
            t_u8 = reencode_via_jpeg_return_tensor(
                x_u8[i],
                quality=quality,
                subsampling=self.subsampling,
                torch_device=None,
            )
            outs.append(t_u8)

        y_u8 = torch.stack(outs, dim=0)  # CPU uint8
        y = y_u8.to(device=device, dtype=torch.float32) / 255.0
        return y

    def forward(self, img):
        if img.ndim != 4:
            raise ValueError(f"Expected img (B,C,H,W), got {tuple(img.shape)}")
        if img.shape[1] != 3:
            raise ValueError(
                f"This implementation assumes RGB input (C==3) for JPEG branches, got C={img.shape[1]}."
            )

        # origin 分支（不做 JPEG）
        x_ori = img if img.dtype.is_floating_point else img.float() / 255.0
        f_ori = self.conv_ori(x_ori)

        # JPEG 分支
        x1 = self._jpeg_batch(img, quality=self.q1)
        x2 = self._jpeg_batch(img, quality=self.q2)
        x3 = self._jpeg_batch(img, quality=self.q3)

        f1 = self.conv_q1(x1)
        f2 = self.conv_q2(x2)
        f3 = self.conv_q3(x3)

        f_1_2 = self.fuse_1_2(torch.cat([f1, f2], dim=1))
        f_1_2_3 = self.fuse_prev_3(torch.cat([f_1_2, f3], dim=1))
        out = self.fuse_prev_ori_out(torch.cat([f_1_2_3, f_ori], dim=1))
        return out


class DW_JpegFusionModule(nn.Module):
    """
    img -> q1 -> conv ---\
    img -> q2 -> conv ---- concat -> 1x1 conv ----\
    img -> q3 -> conv ---------------------------- concat -> 1x1 conv ----\
    img -> origin -> conv --------------------------------------------- concat -> 1x1 conv -> out
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv_channels: int = 64,
        conv_stride: int = 2,
        q1: int = 10,
        q2: int = 20,
        q3: int = 50,
        subsampling: int = 2,
    ):
        super().__init__()
        self.q1 = int(q1)
        self.q2 = int(q2)
        self.q3 = int(q3)
        self.subsampling = int(subsampling)
        self.conv_stride = int(conv_stride)

        # 四路分支 conv：用你的 Conv，k=3, s=conv_stride
        self.conv_q1 = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)
        self.conv_q2 = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)
        self.conv_q3 = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)
        self.conv_ori = Conv(input_channels, conv_channels, k=3, s=self.conv_stride)

        # 1x1 融合 conv（也用你的 Conv，k=1）
        self.fuse_1_2 = Conv(2 * conv_channels, conv_channels, k=1, s=1)
        self.fuse_prev_3 = Conv(2 * conv_channels, conv_channels, k=1, s=1)
        self.fuse_prev_ori_out = Conv(
            2 * conv_channels, output_channels, k=1, s=1, act=False
        )

    @torch.no_grad()
    def _jpeg_batch(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        device = x.device

        if x.dtype == torch.uint8:
            x_u8 = x
        else:
            xf = x.detach()
            if xf.max() <= 1.0:
                xf = xf * 255.0
            x_u8 = xf.round().clamp(0, 255).to(torch.uint8)

        outs = []
        for i in range(b):
            t_u8 = reencode_via_jpeg_return_tensor(
                x_u8[i],
                quality=quality,
                subsampling=self.subsampling,
                torch_device=None,
            )
            outs.append(t_u8)

        y_u8 = torch.stack(outs, dim=0)  # CPU uint8
        y = y_u8.to(device=device, dtype=torch.float32) / 255.0
        return y

    def forward(self, img):
        if img.ndim != 4:
            raise ValueError(f"Expected img (B,C,H,W), got {tuple(img.shape)}")
        if img.shape[1] != 3:
            raise ValueError(
                f"This implementation assumes RGB input (C==3) for JPEG branches, got C={img.shape[1]}."
            )

        # origin 分支（不做 JPEG）
        x_ori = img if img.dtype.is_floating_point else img.float() / 255.0
        f_ori = self.conv_ori(x_ori)

        # JPEG 分支
        x1 = self._jpeg_batch(img, quality=self.q1)
        x2 = self._jpeg_batch(img, quality=self.q2)
        x3 = self._jpeg_batch(img, quality=self.q3)

        f1 = self.conv_q1(x1)
        f2 = self.conv_q2(x2)
        f3 = self.conv_q3(x3)

        f_1_2 = self.fuse_1_2(torch.cat([f1, f2], dim=1))
        f_1_2_3 = self.fuse_prev_3(torch.cat([f_1_2, f3], dim=1))
        out = self.fuse_prev_ori_out(torch.cat([f_1_2_3, f_ori], dim=1))
        return out


class JpegFusionModule12C(nn.Module):
    """
    Simplified JpegFusionModule for 12-channel input (Orig + Q1 + Q2 + Q3).
    Removes internal JPEG compression logic as the dataloader provides pre-processed channels.
    Assumes input shape (B, 12, H, W) where channels are concatenated: [Original, Q1, Q2, Q3].
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv_channels: int = 64,
        conv_stride: int = 2,
    ):
        super().__init__()
        self.conv_stride = int(conv_stride)

        # Determine step size (should be 3 if input is 12)
        c_step = input_channels // 4

        # Branch Convolutions
        # Original branch
        self.conv_ori = Conv(c_step, conv_channels, k=3, s=self.conv_stride)
        # Quality branches
        self.conv_q1 = Conv(c_step, conv_channels // 4, k=3, s=self.conv_stride)
        self.conv_q2 = Conv(c_step, conv_channels // 3, k=3, s=self.conv_stride)
        self.conv_q3 = Conv(c_step, conv_channels // 2, k=3, s=self.conv_stride)

        # 1x1 Fusion Convs
        self.fuse_1_2 = Conv(
            (conv_channels // 4 + conv_channels // 3), conv_channels // 2, k=1, s=1
        )
        self.fuse_prev_3 = Conv(
            (conv_channels // 2 + conv_channels // 2), conv_channels, k=1, s=1
        )
        self.fuse_prev_ori_out = Conv(
            2 * conv_channels, output_channels, k=1, s=1, act=False
        )

    def forward(self, x):
        """
        x: (B, 12, H, W)
        Splits into 4 groups of 3 channels: Orin, Q1, Q2, Q3.
        """
        # Split input
        print(x.shape)
        step = x.shape[1] // 4
        x_ori = x[:, :step, :, :]
        x_q1 = x[:, step : 2 * step, :, :]
        x_q2 = x[:, 2 * step : 3 * step, :, :]
        x_q3 = x[:, 3 * step :, :, :]

        # Forward pass branches
        f_ori = self.conv_ori(x_ori)
        f1 = self.conv_q1(x_q1)
        f2 = self.conv_q2(x_q2)
        f3 = self.conv_q3(x_q3)

        # Fusion
        f_1_2 = self.fuse_1_2(torch.cat([f1, f2], dim=1))
        f_1_2_3 = self.fuse_prev_3(torch.cat([f_1_2, f3], dim=1))
        out = self.fuse_prev_ori_out(torch.cat([f_1_2_3, f_ori], dim=1))
        return out


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
        self,
        in_chs: int,
        rd_ratio: float = 0.25,
        rd_channels: Optional[int] = None,
        act_layer: LayerType = nn.ReLU,
        gate_layer: LayerType = nn.Sigmoid,
        force_act_layer: Optional[LayerType] = None,
        rd_round_fn: Optional[Callable] = None,
    ):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """Conv + Norm Layer + Activation w/ optional skip connection"""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 0,
        pad_type: str = "",
        skip: bool = False,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        drop_path_rate: float = 0.0,
    ):
        super(ConvBnAct, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        self.conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation,
            groups=groups,
            padding=pad_type,
        )
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride, enable=use_aa)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # output of conv after act, same as block coutput
            return dict(
                module="bn1", hook_type="forward", num_chs=self.conv.out_channels
            )
        else:  # location == 'bottleneck', block output
            return dict(module="", num_chs=self.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.aa(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 1,
        pad_type: str = "",
        noskip: bool = False,
        pw_kernel_size: int = 1,
        pw_act: bool = False,
        s2d: int = 0,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        se_layer: Optional[ModuleType] = None,
        drop_path_rate: float = 0.0,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(
                in_chs, sd_chs, kernel_size=2, stride=2, padding="same"
            )
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = "same" if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
            use_aa = False  # disable AA
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        groups = num_groups(group_size, in_chs)

        self.conv_dw = create_conv2d(
            in_chs,
            in_chs,
            dw_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation,
            padding=dw_pad_type,
            groups=groups,
        )
        self.bn1 = norm_act_layer(in_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PW
            return dict(
                module="conv_pw",
                hook_type="forward_pre",
                num_chs=self.conv_pw.in_channels,
            )
        else:  # location == 'bottleneck', block output
            return dict(module="", num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 1,
        pad_type: str = "",
        noskip: bool = False,
        exp_ratio: float = 1.0,
        exp_kernel_size: int = 1,
        pw_kernel_size: int = 1,
        s2d: int = 0,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        se_layer: Optional[ModuleType] = None,
        conv_kwargs: Optional[Dict] = None,
        drop_path_rate: float = 0.0,
    ):
        super(InvertedResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(
                in_chs, sd_chs, kernel_size=2, stride=2, padding="same"
            )
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = "same" if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
            use_aa = False  # disable AA
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)

        # Point-wise expansion
        self.conv_pw = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs,
            mid_chs,
            dw_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation,
            groups=groups,
            padding=dw_pad_type,
            **conv_kwargs,
        )
        self.bn2 = norm_act_layer(mid_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs
        )
        self.bn3 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PWL
            return dict(
                module="conv_pwl",
                hook_type="forward_pre",
                num_chs=self.conv_pwl.in_channels,
            )
        else:  # location == 'bottleneck', block output
            return dict(module="", num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class UniversalInvertedResidual(nn.Module):
    """Universal Inverted Residual Block (aka Universal Inverted Bottleneck, UIB)

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L778
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size_start: int = 0,
        dw_kernel_size_mid: int = 3,
        dw_kernel_size_end: int = 0,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 1,
        noskip: bool = False,
        exp_ratio: float = 1.0,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        se_layer: Optional[ModuleType] = None,
        conv_kwargs: Optional[Dict] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
    ):
        super(UniversalInvertedResidual, self).__init__()
        pad_type = ""
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        # FIXME dilation isn't right w/ extra ks > 1 convs
        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(
                in_chs,
                in_chs,
                dw_kernel_size_start,
                stride=dw_start_stride,
                dilation=dilation,  # FIXME
                groups=dw_start_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            self.dw_start = nn.Identity()

        # Point-wise expansion
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(
            in_chs,
            mid_chs,
            1,
            padding=pad_type,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        # Middle depth-wise convolution
        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(
                mid_chs,
                mid_chs,
                dw_kernel_size_mid,
                stride=stride,
                dilation=dilation,  # FIXME
                groups=groups,
                padding=pad_type,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            # keeping mid as identity so it can be hooked more easily for features
            self.dw_mid = nn.Identity()

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.pw_proj = ConvNormAct(
            mid_chs,
            out_chs,
            1,
            padding=pad_type,
            apply_act=False,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        if dw_kernel_size_end:
            dw_end_stride = (
                stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            )
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = ConvNormAct(
                out_chs,
                out_chs,
                dw_kernel_size_end,
                stride=dw_end_stride,
                dilation=dilation,
                groups=dw_end_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **conv_kwargs,
            )
        else:
            self.dw_end = nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PWL
            return dict(
                module="pw_proj.conv",
                hook_type="forward_pre",
                num_chs=self.pw_proj.conv.in_channels,
            )
        else:  # location == 'bottleneck', block output
            return dict(module="", num_chs=self.pw_proj.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class MobileAttention(nn.Module):
    """Mobile Attention Block

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L1504
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        dw_kernel_size: int = 3,
        dilation: int = 1,
        group_size: int = 1,
        pad_type: str = "",
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        use_multi_query: bool = False,
        query_strides: int = (1, 1),
        kv_stride: int = 1,
        cpe_dw_kernel_size: int = 3,
        noskip: bool = False,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        drop_path_rate: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        use_bias: bool = False,
        use_cpe: bool = False,
    ):
        super(MobileAttention, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_stride = any([s > 1 for s in self.query_strides])

        # This CPE is different than the one suggested in the original paper.
        # https://arxiv.org/abs/2102.10882
        # 1. Rather than adding one CPE before the attention blocks, we add a CPE
        #    into every attention block.
        # 2. We replace the expensive Conv2D by a Seperable DW Conv.
        if use_cpe:
            self.conv_cpe_dw = create_conv2d(
                in_chs,
                in_chs,
                kernel_size=cpe_dw_kernel_size,
                dilation=dilation,
                depthwise=True,
                bias=True,
            )
        else:
            self.conv_cpe_dw = None

        self.norm = norm_act_layer(in_chs, apply_act=False)

        if num_heads is None:
            assert in_chs % key_dim == 0
            num_heads = in_chs // key_dim

        if use_multi_query:
            self.attn = MultiQueryAttention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                query_strides=query_strides,
                kv_stride=kv_stride,
                dilation=dilation,
                padding=pad_type,
                dw_kernel_size=dw_kernel_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                # bias=use_bias, # why not here if used w/ mhsa?
            )
        else:
            self.attn = Attention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                bias=use_bias,
            )

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, input to PW
            return dict(
                module="conv_pw",
                hook_type="forward_pre",
                num_chs=self.conv_pw.in_channels,
            )
        else:  # location == 'bottleneck', block output
            return dict(module="", num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        if self.conv_cpe_dw is not None:
            x_cpe = self.conv_cpe_dw(x)
            x = x + x_cpe

        shortcut = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut

        return x


class CondConvResidual(InvertedResidual):
    """Inverted residual block w/ CondConv routing"""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 1,
        pad_type: str = "",
        noskip: bool = False,
        exp_ratio: float = 1.0,
        exp_kernel_size: int = 1,
        pw_kernel_size: int = 1,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        se_layer: Optional[ModuleType] = None,
        num_experts: int = 0,
        drop_path_rate: float = 0.0,
    ):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)
        super(CondConvResidual, self).__init__(
            in_chs,
            out_chs,
            dw_kernel_size=dw_kernel_size,
            stride=stride,
            dilation=dilation,
            group_size=group_size,
            pad_type=pad_type,
            noskip=noskip,
            exp_ratio=exp_ratio,
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            conv_kwargs=conv_kwargs,
            drop_path_rate=drop_path_rate,
        )
        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EdgeResidual(nn.Module):
    """Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        exp_kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 0,
        force_in_chs: int = 0,
        noskip: bool = False,
        exp_ratio: float = 1.0,
        pw_kernel_size: int = 1,
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[LayerType] = None,
        se_layer: Optional[ModuleType] = None,
        drop_path_rate: float = 0.0,
    ):
        super(EdgeResidual, self).__init__()
        pad_type = ""
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(
            group_size, mid_chs
        )  # NOTE: Using out_chs of conv_exp for groups calc
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs,
            mid_chs,
            exp_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation,
            groups=groups,
            padding=pad_type,
        )
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type
        )
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == "expansion":  # after SE, before PWL
            return dict(
                module="conv_pwl",
                hook_type="forward_pre",
                num_chs=self.conv_pwl.in_channels,
            )
        else:  # location == 'bottleneck', block output
            return dict(module="", num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class CKAS_DAF(nn.Module):
    def __init__(self, dims, c2, k, s):
        super().__init__()
        scale = 4
        group = 2
        self.dims = dims
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)

        self.w_attention_7 = ConvWeightAttention_CKAS(dims, kernel_size=7)
        self.w_attention_5 = ConvWeightAttention_CKAS(dims, kernel_size=5)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(group * dims),
            nn.Conv2d(group * dims, scale * dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(scale * dims, dims, kernel_size=1),
        )

        self.conv1 = nn.Conv2d(
            in_channels=2 * dims, out_channels=dims, kernel_size=1, stride=1, padding=0
        )
        self.daf = DAFusion(dims, dims)
        self.conv4 = Conv(dims, c2, k, s)
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=1)

    def forward(self, x):
        r3 = self.conv2(x) + x

        w5 = F.interpolate(
            self.conv2.weight.data, size=5, mode="bilinear", align_corners=False
        )
        w5 = self.w_attention_5(w5)

        w7 = F.interpolate(
            self.conv2.weight.data, size=7, mode="bilinear", align_corners=False
        )
        w7 = self.w_attention_7(w7)

        r5 = F.conv2d(x, w5, padding=2, groups=self.dims) + x
        r7 = F.conv2d(x, w7, padding=3, groups=self.dims) + x
        r3 = self.conv3(r3)
        r5 = self.conv3(r5)
        r7 = self.conv3(r7)
        rl = [r5, r7]
        r4 = self.daf(rl)

        r = torch.cat([r3, r4], dim=1)
        r_end = self.conv(r)
        r_end = self.conv4(r_end)

        return r_end


class C2f_HKA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_HKA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_HKA(C2f_HKA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            (
                C3k(self.c, self.c, 2, shortcut, g)
                if c3k
                else Bottleneck_HKA(self.c, self.c, shortcut, g)
            )
            for _ in range(n)
        )


class ConvWeightAttention_CKAS(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )
        self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])

    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        return self.ln(res + weight)


class DAFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.conv_align1 = Conv(inc, ouc, 1)
        self.conv_align2 = Conv(inc, ouc, 1)

        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()

        self.x1_param = nn.Parameter(
            torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True
        )
        self.x2_param = nn.Parameter(
            torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True
        )

        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)

        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight

        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


class Bottleneck_HKA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CKAConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DynamicTanh_ConvWeight(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None, None] + self.bias[:, None, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class CKAConv(nn.Module):
    def __init__(self, dims, c2):
        super().__init__()
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.w_attention = ConvWeightAttention_CKA(dims, 3)
        self.se = SEAtt(dims)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(dims),
            nn.SiLU(),
        )
        self.conv2 = nn.Conv2d(dims, c2, kernel_size=1)

    def forward(self, x):
        w3 = self.w_attention(self.conv3.weight.data)
        w3 = self.se(w3)
        r3 = F.conv2d(x, w3, padding=1, groups=self.dims)
        # r = r3 + x
        r = self.conv(r3)
        r = self.conv2(r)
        return r


class SEAtt(nn.Module):
    def __init__(self, dims):
        super(SEAtt, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dims, dims // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims // 16, dims, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, weight):
        b, _, h, w = weight.shape
        weight = weight.view(1, b, h, w)
        att = self.se(weight)
        weight = weight * att
        weight = weight.view(b, 1, h, w)
        return weight


class ConvWeightAttention_CKA(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        embed_dim = 1
        self.q = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.k = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.v = nn.Conv2d(dims, dims, kernel_size=kernel_size, groups=dims)
        self.qw = nn.Parameter(self.q.weight.data)
        self.kw = nn.Parameter(self.k.weight.data)
        self.vw = nn.Parameter(self.v.weight.data)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, batch_first=True
        )
        # self.ln = nn.LayerNorm([embed_dim, kernel_size, kernel_size])
        self.ln = DynamicTanh_ConvWeight(dims, False)

    def forward(self, weight):
        b, c, h, w = weight.shape
        q = torch.matmul(weight, self.qw)
        k = torch.matmul(weight, self.kw)
        v = torch.matmul(weight, self.vw)
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()
        k = k.permute(0, 2, 1).contiguous()
        v = v.permute(0, 2, 1).contiguous()
        res = self.attention(query=q, value=v, key=k, need_weights=False)
        res = res[0].permute(0, 2, 1).contiguous()
        res = res.view(b, c, h, w)
        res = self.ln(res + weight)
        return res


class PSABlock_1(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention_1(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_1(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(
            *(
                PSABlock_1(self.c, attn_ratio=0.5, num_heads=self.c // 64)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Attention_1(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = CKAConv(dim, dim)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            B, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(
            v.reshape(B, C, H, W)
        )
        x = self.proj(x)
        return x


class NonLocalBlockND(nn.Module):
    def __init__(
        self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True
    ):
        super(NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample  # 是否进行下采样

        self.in_channels = in_channels  # 输入通道数
        self.inter_channels = inter_channels  # 中间通道数

        # 如果未指定中间通道数，默认为输入通道数的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 定义 g、theta、phi 的卷积层
        self.g = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # 定义 W 层，可选择是否使用批归一化
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(self.in_channels),
            )
            # nn.init.constant(self.W[1].weight, 0)  # 初始化权重
            # nn.init.constant(self.W[1].bias, 0)    # 初始化偏置
        else:
            self.W = nn.Conv2d(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            # nn.init.constant(self.W.weight, 0)  # 初始化权重
            # nn.init.constant(self.W.bias, 0)    # 初始化偏置

        # 定义 theta 和 phi 的卷积层
        self.theta = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # 如果进行下采样，则对 g 和 phi 添加池化层
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)  # 获取批量大小

        # 计算 g(x)
        g_x = self.g(x).view(
            batch_size, self.inter_channels, -1
        )  # 变形为 (b, inter_channels, N)
        g_x = g_x.permute(0, 2, 1)  # 调整维度顺序

        # 计算 theta(x) 和 phi(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # 调整维度顺序
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算注意力权重
        f = torch.matmul(theta_x, phi_x)  # 矩阵乘法
        f_div_C = F.softmax(f, dim=-1)  # 归一化

        # 加权聚合
        y = torch.matmul(f_div_C, g_x)  # 使用权重对 g_x 加权
        y = y.permute(0, 2, 1).contiguous()  # 调整维度顺序
        y = y.view(
            batch_size, self.inter_channels, *x.size()[2:]
        )  # 变形为 (b, inter_channels, t, h, w)

        # 融合输入和输出
        W_y = self.W(y)  # 通过 W 层
        z = W_y + x  # 残差连接

        return z  # 返回最终输出


##---------------------------end blocks---------------------------##
# endregion


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(
            c_, c_, 2, 2, 0, bias=True
        )  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()
    ):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(
            *(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(
                Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(
                Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1)
                for _ in range(n)
            )
        )


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = (
            nn.Sequential(Conv(c1, c3, k=1, s=s, act=False))
            if s != 1 or c1 != c3
            else nn.Identity()
        )

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = (
            nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        )
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch]
        )
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [
            pool(proj(x)).view(bs, -1, num_patches)
            for (x, proj, pool) in zip(x, self.projections, self.im_pools)
        ]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [
            F.interpolate(x[self.idx[i]], size=target_size, mode="nearest")
            for i, x in enumerate(xs[:-1])
        ]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            (
                C3k(self.c, self.c, 2, shortcut, g)
                if c3k
                else Bottleneck(self.c, self.c, shortcut, g)
            )
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n)
        )


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            B, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(
            v.reshape(B, C, H, W)
        )
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(
            *(
                PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)
                for _ in range(n)
            )
        )

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(
            PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)
        )


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))
