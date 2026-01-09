import ctypes
import torch
import numpy
from typing import Tuple, Union

import nvidia.dali as dali
import nvidia.dali.types as types
from nvidia.dali.backend import TensorGPU, TensorCPU
from nvidia.dali.plugin.pytorch import to_torch_type
from nvjpeg_wrapper import (
    NvJpegExecption,
    NvJpeg,
    ChromaSubsampling,
    InputFormat,
    OutputFormat,
)

__version__ = "1.0.0"
__author__ = "xyswjtu@163.com"

exception = NvJpegExecption


def to_tensor(dali_tensor, cuda_stream=None):
    """将DALI Tensor拷贝为PyTorch的Tensor."""
    torch_type = to_torch_type[dali_tensor.dtype]

    assert isinstance(
        dali_tensor, (TensorGPU, TensorCPU)
    ), "to_tensor仅支持Tensor形式(TensorGPU, TensorCPU), 不支持TensorList."

    if isinstance(dali_tensor, TensorGPU):
        cuda_stream = types._raw_cuda_stream(cuda_stream)
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        tensor = torch.empty(dali_tensor.shape(), dtype=torch_type, device="cuda")
        c_type_pointer = ctypes.c_void_p(tensor.data_ptr())
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
        return tensor
    else:
        tensor = torch.empty(dali_tensor.shape(), dtype=torch_type, device="cpu")
        c_type_pointer = ctypes.c_void_p(tensor.data_ptr())
        dali_tensor.copy_to_external(c_type_pointer)
        return tensor


@dali.pipeline_def(
    batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False
)
def dali_decoder_bgri_pipe():
    images = dali.fn.external_source(device="cpu", name="bytes")
    images = dali.fn.decoders.image(
        images, device="mixed", output_type=types.DALIImageType.BGR, use_fast_idct=True
    )
    return images


@dali.pipeline_def(
    batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False
)
def dali_decoder_rgbi_pipe():
    images = dali.fn.external_source(device="cpu", name="bytes")
    images = dali.fn.decoders.image(
        images, device="mixed", output_type=types.DALIImageType.RGB, use_fast_idct=True
    )
    return images


@dali.pipeline_def(
    batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False
)
def dali_decoder_bgrp_pipe():
    images = dali.fn.external_source(device="cpu", name="bytes")
    images = dali.fn.decoders.image(
        images, device="mixed", output_type=types.DALIImageType.BGR, use_fast_idct=True
    )
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    return images


@dali.pipeline_def(
    batch_size=1, num_threads=4, device_id=0, exec_pipelined=False, exec_async=False
)
def dali_decoder_rgbp_pipe():
    images = dali.fn.external_source(device="cpu", name="bytes")
    images = dali.fn.decoders.image(
        images, device="mixed", output_type=types.DALIImageType.RGB, use_fast_idct=True
    )
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    return images


class nvjpegChromaSubsampling:
    CSS_444 = ChromaSubsampling.CSS_444
    CSS_422 = ChromaSubsampling.CSS_422
    CSS_420 = ChromaSubsampling.CSS_420
    CSS_440 = ChromaSubsampling.CSS_440
    CSS_411 = ChromaSubsampling.CSS_411
    CSS_410 = ChromaSubsampling.CSS_410
    CSS_GRAY = ChromaSubsampling.CSS_GRAY


class nvjpegInputFormat:
    RGBP = InputFormat.RGB
    BGRP = InputFormat.BGR
    RGB = InputFormat.RGBI
    BGR = InputFormat.BGRI


class nvjpegOutputFormat:
    RGBP = OutputFormat.RGB
    BGRP = OutputFormat.BGR
    RGB = OutputFormat.RGBI
    BGR = OutputFormat.BGRI


class JpegCoder:
    """
    Summary:
        JpegCoder类使用Pybind11对nvjpeg库进行了封装.
    """

    def __init__(self) -> None:
        self.nvjpeg_coder = NvJpeg()
        self._bgri_pipe = dali_decoder_bgri_pipe()
        self._rgbi_pipe = dali_decoder_rgbi_pipe()
        self._bgrp_pipe = dali_decoder_bgrp_pipe()
        self._rgbp_pipe = dali_decoder_rgbp_pipe()
        self._bgri_pipe.build()
        self._rgbi_pipe.build()
        self._bgrp_pipe.build()
        self._rgbp_pipe.build()

    def imread(
        self, filename: str, format: nvjpegOutputFormat = nvjpegOutputFormat.BGR
    ) -> torch.Tensor:
        """读取图像数据并解码为torch.Tensor(GPU, torch.uint8)."""
        buffer = numpy.fromfile(filename, dtype=numpy.uint8)
        image = None
        if format == nvjpegOutputFormat.BGR:
            self._bgri_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._bgri_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        elif format == nvjpegOutputFormat.RGB:
            self._rgbi_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._rgbi_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        elif format == nvjpegOutputFormat.BGRP:
            self._bgrp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._bgrp_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        elif format == nvjpegOutputFormat.RGBP:
            self._rgbp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._rgbp_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return None if image is None else to_tensor(image[0])

    def imdecode(
        self,
        data: Union[bytes, numpy.ndarray],
        format: nvjpegOutputFormat = nvjpegOutputFormat.BGR,
    ) -> torch.Tensor:
        """读取图像数据buffer并解码为torch.Tensor(GPU, torch.uint8)."""
        if isinstance(data, bytes):
            buffer = numpy.frombuffer(data, dtype=numpy.uint8)
        else:
            buffer = data
            assert (
                buffer.dtype == numpy.uint8
            ), f"numpy ndarray expected uint8 but got {buffer.dtype}"
        image = None
        if format == nvjpegOutputFormat.BGR:
            self._bgri_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._bgri_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        elif format == nvjpegOutputFormat.RGB:
            self._rgbi_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._rgbi_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        elif format == nvjpegOutputFormat.BGRP:
            self._bgrp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._bgrp_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        elif format == nvjpegOutputFormat.RGBP:
            self._rgbp_pipe.feed_input("bytes", numpy.expand_dims(buffer, axis=0))
            outputs = self._rgbp_pipe.run()
            image = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return None if image is None else to_tensor(image[0])

    def imencode(
        self,
        data: torch.Tensor,
        quality: int = 75,
        format: nvjpegInputFormat = nvjpegInputFormat.BGR,
        subsampling: nvjpegChromaSubsampling = nvjpegChromaSubsampling.CSS_420,
    ) -> torch.Tensor:
        """将GPU Torch Tensor图像数据编码."""
        assert quality > 0 and quality <= 100, "quality should be between 0 and 100"
        return self.nvjpeg_coder.imencode(data, quality, format, subsampling)

    def imwrite(
        self,
        filename: str,
        data: torch.Tensor,
        quality: int = 75,
        format: nvjpegInputFormat = nvjpegInputFormat.BGR,
        subsampling: nvjpegChromaSubsampling = nvjpegChromaSubsampling.CSS_420,
    ) -> bool:
        """将GPU Torch Tensor图像数据编码且写到指定的路径."""
        assert quality > 0 and quality <= 100, "quality should be between 0 and 100"
        return self.nvjpeg_coder.imwrite(filename, data, quality, format, subsampling)

    def shape(self, data: Union[str, torch.Tensor, numpy.ndarray]) -> Tuple[int]:
        """获取图像的形状信息，格式为(H, W, C)."""
        if isinstance(data, str):
            return self.nvjpeg_coder.shape_filename(data)
        elif isinstance(data, numpy.ndarray):
            return self.nvjpeg_coder.shape_numpy(data)
        else:
            return self.nvjpeg_coder.shape_torch(data)

    def version(self) -> str:
        """获取nvjepg的版本信息."""
        return self.nvjpeg_coder.version()

    def __repr__(self) -> str:
        return self.nvjpeg_coder.__repr__()
