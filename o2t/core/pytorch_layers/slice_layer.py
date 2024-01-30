import torch
import torch.nn as nn
from onnx_graphsurgeon import Constant


class Slice(nn.Module):
    @classmethod
    def from_onnx(cls, mod, env):
        def get_input_node(input, env):
            if isinstance(input, Constant):
                return int(input.values)
            else:
                return env[input.name]

        start = mod.inputs[1]
        end = mod.inputs[2]
        axes = mod.inputs[3] if len(mod.inputs) > 3 else None
        steps = mod.inputs[4] if len(mod.inputs) > 4 else None

        slice_inputs = [slice(None, None, None)] * len(mod.inputs[0].shape)
        if axes is not None:
            for idx in range(len(axes.values)):
                axes_idx = axes.values[idx]
                steps_ = steps.values[idx] if steps is not None else 1
                slice_inputs[axes_idx] = slice(
                    start.values[idx], end.values[idx], steps_
                )
        else:
            raise NotImplementedError

        return tuple(slice_inputs)
