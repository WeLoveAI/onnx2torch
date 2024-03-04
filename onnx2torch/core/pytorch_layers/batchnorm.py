import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BatchNorm(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        input_shape = mod.inputs[0].shape
        input_dim = len(input_shape) if input_shape is not None else None
        if input_dim == 2 or input_dim == 3:
            bn = nn.BatchNorm1d(
                mod.inputs[1].shape[0],
                eps=mod.attrs["epsilon"],
                momentum=mod.attrs["momentum"],
            )
            bn.weight = Parameter(
                torch.from_numpy(mod.inputs[1].values), requires_grad=False
            )
            bn.bias = Parameter(
                torch.from_numpy(mod.inputs[2].values), requires_grad=False
            )
            bn.running_mean = Parameter(
                torch.from_numpy(mod.inputs[3].values), requires_grad=False
            )
            bn.running_var = Parameter(
                torch.from_numpy(mod.inputs[4].values), requires_grad=False
            )
            return bn
        elif input_dim == 4:
            bn = nn.BatchNorm2d(
                mod.inputs[1].shape[0],
                eps=mod.attrs["epsilon"],
                momentum=mod.attrs["momentum"],
            )
            bn.weight = Parameter(
                torch.from_numpy(mod.inputs[1].values), requires_grad=False
            )
            bn.bias = Parameter(
                torch.from_numpy(mod.inputs[2].values), requires_grad=False
            )
            bn.running_mean = Parameter(
                torch.from_numpy(mod.inputs[3].values), requires_grad=False
            )
            bn.running_var = Parameter(
                torch.from_numpy(mod.inputs[4].values), requires_grad=False
            )

            return bn
        else:
            raise
