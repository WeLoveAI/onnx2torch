import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class InstanceNormalization(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        input_shape = mod.inputs[0].shape
        input_dim = len(input_shape) if input_shape is not None else None
        eps = mod.attrs.get("epsilon", 1e-5)
        if input_dim == 3:
            if len(mod.inputs) == 1:
                normalized_shape = mod.inputs[0].shape
                return nn.InstanceNorm1d(normalized_shape, eps, False, False)
            else:
                normalized_shape = mod.inputs[1].shape
                scale = Parameter(torch.from_numpy(mod.inputs[1].values))
                bias = Parameter(torch.from_numpy(mod.inputs[2].values))

                instancenorm = nn.InstanceNorm1d(num_features=scale.size()[0], eps=eps)
                instancenorm.weight = scale
                instancenorm.bias = bias

                return instancenorm
        elif input_dim == 4:
            if len(mod.inputs) == 1:
                normalized_shape = mod.inputs[0].shape
                return nn.InstanceNorm2d(normalized_shape, eps, False, False)
            else:
                normalized_shape = mod.inputs[1].shape
                scale = Parameter(torch.from_numpy(mod.inputs[1].values))
                bias = Parameter(torch.from_numpy(mod.inputs[2].values))

                instancenorm = nn.InstanceNorm2d(num_features=scale.size()[0], eps=eps)
                instancenorm.weight = scale
                instancenorm.bias = bias

                return instancenorm
        else:
            raise
