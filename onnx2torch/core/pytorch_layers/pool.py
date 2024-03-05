import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .pad_layer import is_symmetric_padding, pad_onnx_to_torch

from .utils import get_value_by_key


class Pool(nn.Module):
    @classmethod
    def from_onnx(cls, onnx_node):
        input_shape = onnx_node.inputs[0].shape
        input_dim = len(input_shape) if input_shape is not None else None
        if onnx_node.op == "GlobalAveragePool":
            if input_dim == 3:
                pool = nn.AdaptiveAvgPool1d((1))
            elif input_dim == 4:
                pool = nn.AdaptiveAvgPool2d((1, 1))
        elif onnx_node.op == "MaxPool":
            padding = onnx_node.attrs["pads"]
            if is_symmetric_padding(padding):
                padding = padding[: len(padding) // 2]
                pool = nn.MaxPool2d(
                    kernel_size=onnx_node.attrs["kernel_shape"],
                    stride=onnx_node.attrs["strides"],
                    padding=padding,
                    ceil_mode=bool(get_value_by_key(onnx_node, "ceil_mode", 0)),
                )
            else:
                torch_padding = pad_onnx_to_torch(padding)
                pad = nn.ConstantPad2d(torch_padding, 0)
                pool = nn.MaxPool2d(
                    kernel_size=onnx_node.attrs["kernel_shape"],
                    stride=onnx_node.attrs["strides"],
                    padding=0,
                    ceil_mode=bool(get_value_by_key(onnx_node, "ceil_mode", 0)),
                )
                pool = nn.Sequential(pad, pool)
        elif onnx_node.op == "AveragePool":
            pool = nn.AvgPool2d(
                kernel_size=onnx_node.attrs["kernel_shape"],
                stride=onnx_node.attrs["strides"],
                padding=get_value_by_key(onnx_node, "pads", [0, 0, 0, 0])[2:],
                ceil_mode=bool(get_value_by_key(onnx_node, "ceil_mode", 0)),
                count_include_pad=bool(
                    get_value_by_key(onnx_node, "count_include_pad", 0)
                ),
            )
        return pool
