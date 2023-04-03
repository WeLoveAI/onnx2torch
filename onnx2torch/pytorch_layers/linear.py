import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        weight = Parameter(torch.from_numpy(mod.inputs[1].values))
        bias = (
            None
            if len(mod.inputs) < 3
            else Parameter(torch.from_numpy(mod.inputs[2].values))
        )
        linear = nn.Linear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bias=bias is not None,
        )
        linear.weight = weight
        linear.bias = bias
        return linear
