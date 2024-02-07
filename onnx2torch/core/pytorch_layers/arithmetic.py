import torch
import torch.nn as nn
from onnx_graphsurgeon import Constant


class Arithmetic(nn.Module):
    @classmethod
    def from_onnx(cls, mod):  # arithmetic can be different
        def get_input_node(input):
            if isinstance(input, Constant):
                return torch.nn.Parameter(
                    torch.from_numpy(input.values), requires_grad=False
                )
            else:
                return input

        inputs = []
        for input in mod.inputs:
            if len(input.inputs) == 0 and not isinstance(
                input, Constant
            ):  # input is real input
                inputs.append(input)
            elif isinstance(input, Constant):
                inputs.append(get_input_node(input))
            for feed in input.inputs:
                if (
                    feed.op == "Split"
                ):  # for split node, we need to get the output of split node
                    inputs.append(input)
                else:
                    inputs.append(get_input_node(feed))

        return tuple(inputs)
