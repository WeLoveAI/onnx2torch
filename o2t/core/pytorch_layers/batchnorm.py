import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNorm(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        bn = nn.BatchNorm2d(mod.inputs[1].shape[0], eps=mod.attrs["epsilon"])
        bn.scale = Parameter(torch.from_numpy(mod.inputs[1].values))
        bn.bias = Parameter(torch.from_numpy(mod.inputs[2].values))
        bn.running_mean = Parameter(
            torch.from_numpy(mod.inputs[3].values), requires_grad=False
        )
        bn.running_var = Parameter(
            torch.from_numpy(mod.inputs[4].values), requires_grad=False
        )

        return bn
