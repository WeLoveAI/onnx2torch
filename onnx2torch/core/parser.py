import _operator
import re

import numpy as np

import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, GraphModule
from .pytorch_layers import *
from ..utils.utils import (
    gen_onnxruntime_input_data,
    numpy_dtype_to_torch,
    onnxruntime_inference,
)


class OnnxPytorchParser:
    def __init__(self, model, block_info=None):
        super(OnnxPytorchParser, self).__init__()
        self.model = model
        self.block_info = block_info
        if isinstance(model, str):
            self.onnx_model = onnx.load(model)
        else:
            self.onnx_model = model

        self.graph = gs.import_onnx(self.onnx_model)
        self.graph.fold_constants().cleanup().toposort()
        self.pytorch_graph = Graph()
        self.pytorch_graph_module = GraphModule(torch.nn.Module(), self.pytorch_graph)
        self.env = {}
        self._illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")

    def convert(self):
        self.gen_pytorch_graph_module()

    def create_arg(self, a):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.pytorch_graph_module.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.pytorch_graph_module.named_buffers():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.pytorch_graph_module.named_modules():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})

        if isinstance(a, tuple) and hasattr(a, "_fields"):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        qualname = None
        if isinstance(a, (torch.Tensor)):
            if not qualname:
                i = 0
                while True:
                    qualname = f"_tensor_constant{i}"
                    if not hasattr(self.pytorch_graph_module, qualname):
                        break
                    i += 1
                setattr(self.pytorch_graph_module, qualname, a)

            return self.pytorch_graph.create_node("get_attr", qualname, (), {})

    def process_inputs(self, inputs):
        inputs = list(inputs)
        for idx in range(len(inputs)):
            input = self.create_arg(inputs[idx])
            if input:
                inputs[idx] = input
            else:
                inputs[idx] = self.env[inputs[idx].name]

        inputs = tuple(inputs)

        return inputs

    def get_node_users(self, node):
        users = []
        for output in node.outputs:  # output is a Variable
            for user in output.outputs:  # user is a Node
                users.append(user)
        return users

    def get_node_feeds(self, node):
        feeds = []
        for input in node.inputs:  # input is a Variable
            for feed in input.inputs:  # user is a Node
                if (
                    feed.op == "Split"
                ):  # for split node, we need to get the output of split node
                    feeds.append(input)
                else:
                    feeds.append(feed)
        return feeds

    def find_block_id(self, node_name, block_info):
        if block_info is None:
            return None

        for block_id, block_data in block_info.items():
            if node_name in block_data["nodes"]:
                return block_id
        # Return None if the node is not found in any block
        return None

    def get_value_by_key_or_index(self, node, key, index, default=None):
        if key in node.attrs:
            return node.attrs[key]
        elif index < len(node.inputs):
            if isinstance(node.inputs[index], gs.Constant):
                return node.inputs[index].values
            else:
                return default
        else:
            return default

    def gen_pytorch_graph_module(self):
        for input in self.graph.inputs:
            node = self.pytorch_graph.placeholder(
                self._illegal_char_regex.sub("_", input.name)
            )
            self.env[input.name] = node

        for onnx_node in self.graph.nodes:
            node_name = onnx_node.name
            target_name = node_name
            block_id = self.find_block_id(node_name, self.block_info)
            if block_id is not None:
                target_name = f"{block_id}.{node_name}"
            node_feeds = self.get_node_feeds(onnx_node)
            if len(node_feeds) == 0:
                node_feeds = self.graph.inputs[0]
            elif len(node_feeds) == 1:
                node_feeds = node_feeds[0]

            if onnx_node.op == "Conv":
                module = Conv.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "LayerNormalization":
                module = LayerNorm.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Relu":
                module = nn.ReLU()
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Clip":
                module = nn.ReLU6()
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Add":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.add,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Sub":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.sub,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Div":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.div,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Mul":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.mul,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "MatMul":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.matmul,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Gelu":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    F.gelu,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "LogSoftmax":
                axis = onnx_node.attrs.get("axis", None)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    F.log_softmax,
                    (self.env[node_feeds.name], axis),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "GlobalAveragePool":
                module = Pool.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "MaxPool":
                module = Pool.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "AveragePool":
                module = Pool.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Flatten":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.flatten,
                    (self.env[node_feeds.name],),
                    {"start_dim": onnx_node.attrs["axis"]},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Concat":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.cat,
                    (inputs,),
                    {"dim": onnx_node.attrs["axis"]},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Reshape":
                if isinstance(onnx_node.inputs[1], gs.Constant):
                    node = self.pytorch_graph.create_node(
                        "call_method",
                        "reshape",
                        (
                            self.env[node_feeds.name],
                            onnx_node.inputs[1].values.tolist(),
                        ),
                        {},
                        node_name,
                    )
                    self.env[node_name] = node
                else:
                    cpu_node_name = node_name + "_cpu"
                    node = self.pytorch_graph.create_node(
                        "call_method",
                        "cpu",
                        (self.env[node_feeds[1].name],),
                        {},
                        cpu_node_name,
                    )
                    self.env[cpu_node_name] = node
                    numpy_node_name = node_name + "_numpy"
                    node = self.pytorch_graph.create_node(
                        "call_method",
                        "numpy",
                        (self.env[cpu_node_name],),
                        {},
                        numpy_node_name,
                    )
                    self.env[numpy_node_name] = node
                    to_list_node_name = node_name + "_tolist"
                    node = self.pytorch_graph.create_node(
                        "call_method",
                        "tolist",
                        (self.env[numpy_node_name],),
                        {},
                        to_list_node_name,
                    )
                    self.env[to_list_node_name] = node
                    module = Reshape.from_onnx()
                    self.pytorch_graph_module.add_submodule(target_name, module)
                    node = self.pytorch_graph.create_node(
                        "call_module",
                        target_name,
                        (self.env[node_feeds[0].name], self.env[to_list_node_name]),
                        {},
                        node_name,
                    )
                    self.env[node_name] = node
            elif onnx_node.op == "Transpose":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.permute,
                    (
                        self.env[node_feeds.name],
                        onnx_node.attrs["perm"],
                    ),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Split":
                chunk = self.get_value_by_key_or_index(
                    onnx_node, "split", 1, len(onnx_node.outputs)
                )
                if len(onnx_node.inputs) > 1:
                    if isinstance(onnx_node.inputs[1], gs.Constant):
                        chunk = onnx_node.inputs[1].values
                    else:
                        chunk = self.env[node_feeds[1].name]
                    func = torch.split
                else:
                    chunk = len(onnx_node.outputs)
                    func = torch.tensor_split

                if isinstance(chunk, np.ndarray):
                    chunk = chunk.tolist()
                dim = self.get_value_by_key_or_index(onnx_node, "axis", 2, 0)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    func,
                    (
                        self.env[node_feeds.name],
                        chunk,
                        dim,
                    ),
                    {},
                    node_name,
                )
                self.env[node_name] = node
                for i, output in enumerate(onnx_node.outputs):
                    node = self.pytorch_graph.create_node(
                        "call_function",
                        _operator.getitem,
                        (
                            self.env[node_name],
                            i,
                        ),
                        {},
                        output.name,
                    )
                    self.env[output.name] = node
            elif onnx_node.op == "Slice":
                if isinstance(onnx_node.inputs[0], gs.Constant):
                    node_feeds = torch.nn.Parameter(
                        torch.from_numpy(onnx_node.inputs[0].values),
                        requires_grad=False,
                    )
                    node_feeds = self.process_inputs([node_feeds])[0]
                elif isinstance(node_feeds, list):
                    node_feeds = node_feeds[0]
                    node_feeds = self.env[node_feeds.name]
                else:
                    node_feeds = self.env[node_feeds.name]
                inputs = Slice.from_onnx(onnx_node, self.env)
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    _operator.getitem,
                    (
                        node_feeds,
                        inputs,
                    ),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Gemm":
                module = Linear.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "BatchNormalization":
                module = BatchNorm.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Softmax":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.softmax,
                    (self.env[node_feeds.name],),
                    {"dim": -1},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Sigmoid":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.sigmoid,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "HardSwish":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.hardswish,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "LeakyRelu":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.leaky_relu,
                    (self.env[node_feeds.name], onnx_node.attrs["alpha"]),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Resize":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.interpolate,
                    (self.env[node_feeds.name],),
                    {
                        "scale_factor": onnx_node.inputs[2].values.tolist()[2:],
                        "mode": onnx_node.attrs["mode"],
                    },
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Pow":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.pow,
                    (self.env[node_feeds.name], float(onnx_node.inputs[1].values)),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Sqrt":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.sqrt,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Erf":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.sqrt,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Squeeze":
                dim = self.get_value_by_key_or_index(onnx_node, "axes", 1, None)
                if isinstance(dim, np.ndarray):
                    dim = dim.tolist()
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.squeeze,
                    (self.env[node_feeds.name], dim),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Unsqueeze":
                axes = onnx_node.attrs.get("axes")[0]
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.unsqueeze,
                    (self.env[node_feeds.name], int(axes)),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Neg":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.neg,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "ConstantOfShape":
                size_name = node_name + "_size"
                module = Size.from_onnx()
                self.pytorch_graph_module.add_submodule(size_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    size_name,
                    (self.env[node_feeds.name],),
                    {},
                    size_name,
                )
                self.env[size_name] = node
                module = Full.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[size_name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "ReduceMean":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_method",
                    "mean",
                    (self.env[node_feeds.name],),
                    {
                        "dim": onnx_node.attrs["axes"],
                        "keepdim": bool(onnx_node.attrs.get("keepdims", 1)),
                    },
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Cast":
                torch_dtype = numpy_dtype_to_torch(
                    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_node.attrs["to"]]
                )
                node = self.pytorch_graph_module.graph.create_node(
                    "call_method",
                    "to",
                    (self.env[node_feeds.name], torch_dtype),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "ReduceSum":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    torch.sum,
                    (self.env[node_feeds.name],),
                    {
                        "dim": onnx_node.attrs["axes"],
                        "keepdim": bool(onnx_node.attrs.get("keepdims", 1)),
                    },
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "ReduceMax":
                if onnx_node.attrs.get("axes"):
                    node = self.pytorch_graph_module.graph.create_node(
                        "call_function",
                        torch.max,
                        (self.env[node_feeds.name],),
                        {
                            "dim": onnx_node.attrs.get("axes"),
                            "keepdim": bool(onnx_node.attrs.get("keepdims", 1)),
                        },
                        node_name,
                    )
                    self.env[node_name] = node
                else:
                    node = self.pytorch_graph_module.graph.create_node(
                        "call_function",
                        torch.max,
                        (self.env[node_feeds.name],),
                        {},
                        node_name,
                    )
                    self.env[node_name] = node
            elif onnx_node.op == "Shape":
                shape_node_name = node_name + "_shape"
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    getattr,
                    (self.env[node_feeds.name], "shape"),
                    {},
                    shape_node_name,
                )
                self.env[shape_node_name] = node
                module = Tensor.from_onnx()
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[shape_node_name],),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Range":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                module = Arange.from_onnx()
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Equal":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.eq,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "LessOrEqual":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.less_equal,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "GreaterOrEqual":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.greater_equal,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "And":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.logical_and,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Not":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.logical_not,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Where":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.where,
                    inputs,
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Expand":
                inputs = Arithmetic.from_onnx(onnx_node)
                inputs = self.process_inputs(inputs)

                size_name = node_name + "_size"
                module = Size.from_onnx()
                self.pytorch_graph_module.add_submodule(size_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    size_name,
                    (self.env[node_feeds[1].name],),
                    {},
                    size_name,
                )
                self.env[size_name] = node

                ones_node_name = node_name + "_ones"
                module = Ones.from_onnx()
                self.pytorch_graph_module.add_submodule(ones_node_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    ones_node_name,
                    (self.env[size_name],),
                    {},
                    ones_node_name,
                )
                self.env[ones_node_name] = node

                node = self.pytorch_graph.create_node(
                    "call_function",
                    _operator.mul,
                    (self.env[node_feeds[0].name], self.env[ones_node_name]),
                    {},
                    node_name,
                )
                self.env[node_name] = node
            elif onnx_node.op == "Gather":
                if isinstance(onnx_node.inputs[0], gs.Constant):
                    module = Embedding.from_onnx(onnx_node)
                    self.pytorch_graph_module.add_submodule(target_name, module)
                    node = self.pytorch_graph.create_node(
                        "call_module",
                        target_name,
                        (self.env[node_feeds.name],),
                        {},
                        node_name,
                    )
                    self.env[node_name] = node
                elif all(isinstance(input, gs.Variable) for input in onnx_node.inputs):
                    axis = onnx_node.attrs.get("axis", 0)
                    index = self.env[node_feeds[1].name]
                    if axis == 0:
                        index_all = index
                    else:
                        index_all = [slice(None, None, None)] * (axis)
                        index_all.append(index)
                    node = self.pytorch_graph_module.graph.create_node(
                        "call_function",
                        _operator.getitem,
                        (
                            self.env[node_feeds[0].name],
                            index_all,
                        ),
                        {},
                        node_name,
                    )
                    self.env[node_name] = node
                else:
                    axis = onnx_node.attrs.get("axis", 0)
                    index = torch.nn.Parameter(
                        torch.from_numpy(onnx_node.inputs[1].values),
                        requires_grad=False,
                    )
                    index = self.process_inputs([index])[0]
                    if axis == 0:
                        index_all = index
                    else:
                        index_all = [slice(None, None, None)] * (axis)
                        index_all.append(index)
                    node = self.pytorch_graph_module.graph.create_node(
                        "call_function",
                        _operator.getitem,
                        (
                            self.env[node_feeds.name],
                            index_all,
                        ),
                        {},
                        node_name,
                    )
                    self.env[node_name] = node
            elif onnx_node.op == "QuantizeLinear":
                dequant_node = onnx_node.o(0)
                assert dequant_node.op == "DequantizeLinear"

                module = Observer(
                    float(onnx_node.inputs[1].values), float(onnx_node.inputs[2].values)
                )
                self.pytorch_graph_module.add_submodule(target_name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    target_name,
                    (self.env[node_feeds.name],),
                    {},
                    node_name,
                )
                self.env[dequant_node.outputs[0].name] = node
            elif onnx_node.op == "DequantizeLinear":
                pass
            else:
                raise NotImplementedError(
                    "Operator {} is not supported.".format(onnx_node.op)
                )

        if len(self.graph.outputs) == 1:
            if self.graph.outputs[0].inputs[0].op == "Split":
                graph_output = self.env[self.graph.outputs[0].name]
            else:
                graph_output = self.env[self.graph.outputs[0].inputs[0].name]

            node = self.pytorch_graph.output(graph_output)
        else:
            graph_output = []
            for output in self.graph.outputs:
                if output.inputs[0].op == "Split":
                    graph_output.append(self.env[output.name])
                else:
                    graph_output.append(self.env[output.inputs[0].name])

            node = self.pytorch_graph.output(graph_output)

        self.pytorch_graph_module.graph.lint()
        self.pytorch_graph_module.recompile()

    def save(self, output_model):
        torch.save(self.pytorch_graph_module, output_model)

    def check(self):
        input_data_dict = gen_onnxruntime_input_data(self.onnx_model)
        onnx_output_dict = onnxruntime_inference(self.onnx_model, input_data_dict)
        torch_dict = {
            self._illegal_char_regex.sub("_", k): torch.from_numpy(v)
            for k, v in input_data_dict.items()
        }
        with torch.no_grad():
            self.pytorch_graph_module.eval()
            torch_output = self.pytorch_graph_module(**torch_dict)

        if isinstance(torch_output, torch.Tensor):
            torch_output = [torch_output]

        onnx_output = list(onnx_output_dict.values())
        assert len(torch_output) == len(
            onnx_output
        ), f"({len(torch_output)}, {len(onnx_output)})"

        for idx in range(len(onnx_output)):
            np.testing.assert_allclose(
                torch_output[idx].detach().cpu().numpy(),
                onnx_output[idx],
                rtol=1e-7,
                atol=1e-3,
            )

        print("accuracy test passed")
