import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def is_symmetric_padding(pads):
    half = len(pads) // 2
    return pads[:half] == pads[half:]


def pad_onnx_to_torch(onnx_intervals):
    num_intervals = len(onnx_intervals) // 2
    torch_intervals = []
    onnx_intervals_begin, onnx_intervals_end = (
        onnx_intervals[:num_intervals],
        onnx_intervals[num_intervals:],
    )
    onnx_intervals_begin, onnx_intervals_end = (
        onnx_intervals_begin[::-1],
        onnx_intervals_end[::-1],
    )

    torch_intervals = []
    for begin, end in zip(onnx_intervals_begin, onnx_intervals_end):
        torch_intervals.extend([begin, end])

    return torch_intervals


class Pad(nn.Module):
    @classmethod
    def from_onnx(cls, onnx_node):
        padding = onnx_node.inputs[1].values.tolist()
        torch_padding = pad_onnx_to_torch(padding)
        pad = nn.ConstantPad1d(torch_padding, 0)

        return pad
