import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

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
            pool = nn.MaxPool2d(
                kernel_size=onnx_node.attrs["kernel_shape"],
                stride=onnx_node.attrs["strides"],
                padding=onnx_node.attrs["pads"][2:],
                ceil_mode=bool(get_value_by_key(onnx_node, "ceil_mode", 0)),
            )
        elif onnx_node.op == "AveragePool":
            pool = nn.AvgPool2d(
                kernel_size=onnx_node.attrs["kernel_shape"],
                stride=onnx_node.attrs["strides"],
                padding=get_value_by_key(onnx_node, "pads", [0, 0, 0, 0])[2:],
                ceil_mode=bool(get_value_by_key(onnx_node, "ceil_mode", 0)),
            )
        return pool
