import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Conv(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        weight = Parameter(torch.from_numpy(mod.inputs[1].values))
        bias = (
            None
            if len(mod.inputs) < 3
            else Parameter(torch.from_numpy(mod.inputs[2].values))
        )
        groups = mod.attrs["group"]
        padding = mod.attrs["pads"][0:2]
        conv = nn.Conv2d(
            in_channels=weight.shape[1] * groups,
            out_channels=weight.shape[0],
            kernel_size=weight.shape[2:],
            stride=mod.attrs["strides"],
            padding=padding,
            dilation=mod.attrs["dilations"],
            groups=groups,
            bias=bias is not None,
        )
        conv.weight = weight
        conv.bias = bias

        return conv
