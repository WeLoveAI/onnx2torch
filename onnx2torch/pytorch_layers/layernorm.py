import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LayerNorm(nn.Module):
    @classmethod
    def from_onnx(cls, mod):
        eps = mod.attrs.get("epsilon", 1e-5)
        if len(mod.inputs) == 1:
            normalized_shape = mod.inputs[0].shape
            return nn.LayerNorm(normalized_shape, eps, False, False)
        else:
            normalized_shape = mod.inputs[1].shape
            weight = Parameter(torch.from_numpy(mod.inputs[1].values))
            bias = (
                None
                if len(mod.inputs) < 3
                else Parameter(torch.from_numpy(mod.inputs[2].values))
            )

            layernorm = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=True,
                bias=bias is not None,
            )
            layernorm.weight = weight
            layernorm.bias = bias

            return layernorm
