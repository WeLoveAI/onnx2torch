import torch
import torch.nn as nn
from onnx_graphsurgeon import Constant


class Arithmetic(nn.Module):
    @classmethod
    def from_onnx(cls, mod, env):
        def get_input_node(input, env):
            if isinstance(input, Constant):
                return torch.nn.Parameter(
                    torch.from_numpy(input.values), requires_grad=False
                )
            else:
                return env[input.name]

        inputs = []
        for input in mod.inputs:
            if isinstance(input, Constant):
                inputs.append(get_input_node(input, env))
            else:
                for feed in input.inputs:
                    inputs.append(get_input_node(feed, env))

        return tuple(inputs)
