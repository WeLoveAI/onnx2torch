import torch
import torch.nn as nn
from onnx_graphsurgeon import Constant


class Slice(nn.Module):
    @classmethod
    def from_onnx(cls, mod, env):
        def get_input_node(input, env, idx, default_value=-1):
            if isinstance(input, Constant):
                return input.values[idx]
            elif input is None:
                return default_value
            else:
                for feed in input.inputs:  # user is a Node
                    if (
                        feed.op == "Split"
                    ):  # for split node, we need to get the output of split node
                        return env[input.name]
                    else:
                        return env[feed.name]

        start = mod.inputs[1]
        end = mod.inputs[2]
        axes = mod.inputs[3] if len(mod.inputs) > 3 else None
        steps = mod.inputs[4] if len(mod.inputs) > 4 else None

        if max(axes.values) > 0:
            slice_inputs = (
                [slice(None, None, None)] * (max(axes.values) + 1)
                if axes is not None
                else [slice(None, None, None)] * len(start.values)
            )
            if axes is not None:
                for idx in range(len(axes.values)):
                    axes_idx = axes.values[idx]

                    start_ = get_input_node(start, env, idx)
                    end_ = get_input_node(end, env, idx)
                    steps_ = get_input_node(steps, env, idx, 1)

                    slice_inputs[axes_idx] = slice(start_, end_, steps_)
            else:
                raise NotImplementedError

            return tuple(slice_inputs)
        else:
            slice_inputs = [...] + [slice(None, None, None)] * (abs(max(axes.values)))
            for idx in range(len(axes.values)):
                axes_idx = axes.values[idx]

                start_ = get_input_node(start, env, idx)
                end_ = get_input_node(end, env, idx)
                steps_ = get_input_node(steps, env, idx, 1)

                slice_inputs[axes_idx] = slice(start_, end_, steps_)
            return tuple(slice_inputs)
