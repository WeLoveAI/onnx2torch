import torch
import torch.nn as nn
from onnx_graphsurgeon import Constant


class Arithmetic(nn.Module):
    @classmethod
    def from_onnx(cls, mod, env):
        def get_input_node(input, env):
            if isinstance(input, Constant):
                return torch.from_numpy(input.values)
            else:
                return env[input.name]

        inputs = []
        for input in mod.inputs:
            inputs.append(get_input_node(input, env))

        return tuple(inputs)
