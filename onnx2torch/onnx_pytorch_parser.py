import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, GraphModule

import onnx
import onnx_graphsurgeon as gs
import _operator
from .pytorch_layers import *


class OnnxPytorchParser:
    def __init__(self, model, fuse=False, dynamic_batch=False):
        super(OnnxPytorchParser, self).__init__()
        self.model = model
        self.onnx_model = onnx.load(model)
        self.graph = gs.import_onnx(self.onnx_model)
        self.graph.fold_constants().cleanup().toposort()
        self.pytorch_graph = Graph()
        self.pytorch_graph_module = GraphModule(torch.nn.Module(), self.pytorch_graph)
        self.env = {}
        self._illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")

    def convert(self):
        self.gen_pytorch_graph_module()
        return self.pytorch_graph_module

    def gen_pytorch_graph_module(self):
        for input in self.graph.inputs:
            node = self.pytorch_graph.placeholder(
                self._illegal_char_regex.sub("_", input.name)
            )
            self.env[input.name] = node

        for onnx_node in self.graph.nodes:
            if onnx_node.op == "Conv":
                module = Conv.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(onnx_node.outputs[0].name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    onnx_node.outputs[0].name,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Relu":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    F.relu,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Add":
                inputs = Arithmetic.from_onnx(onnx_node, self.env)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.add,
                    inputs,
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Div":
                inputs = Arithmetic.from_onnx(onnx_node, self.env)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.div,
                    inputs,
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Mul":
                inputs = Arithmetic.from_onnx(onnx_node, self.env)
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.mul,
                    inputs,
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "GlobalAveragePool":
                module = Pool.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(onnx_node.outputs[0].name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    onnx_node.outputs[0].name,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "MaxPool":
                module = Pool.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(onnx_node.outputs[0].name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    onnx_node.outputs[0].name,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "AveragePool":
                module = Pool.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(onnx_node.outputs[0].name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    onnx_node.outputs[0].name,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Flatten":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.flatten,
                    (self.env[onnx_node.inputs[0].name],),
                    {"start_dim": onnx_node.attrs["axis"]},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Concat":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.cat,
                    ([self.env[input_node.name] for input_node in onnx_node.inputs],),
                    {"dim": onnx_node.attrs["axis"]},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Reshape":
                node = self.pytorch_graph.create_node(
                    "call_method",
                    "reshape",
                    (
                        self.env[onnx_node.inputs[0].name],
                        onnx_node.inputs[1].values.tolist(),
                    ),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Transpose":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.permute,
                    (
                        self.env[onnx_node.inputs[0].name],
                        onnx_node.attrs["perm"],
                    ),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Split":
                node = self.pytorch_graph.create_node(
                    "call_function",
                    torch.chunk,
                    (
                        self.env[onnx_node.inputs[0].name],
                        len(onnx_node.inputs[1].values.tolist()),
                    ),
                    {"dim": onnx_node.attrs["axis"]},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
                for i, output in enumerate(onnx_node.outputs):
                    node = self.pytorch_graph.create_node(
                        "call_function",
                        _operator.getitem,
                        (
                            self.env[onnx_node.outputs[0].name],
                            i,
                        ),
                        {},
                        output.name,
                    )
                    self.env[output.name] = node
            elif onnx_node.op == "Slice":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    _operator.getitem,
                    (
                        self.env[onnx_node.inputs[0].name],
                        (slice(1, self.env[onnx_node.inputs[2].name], 1)),
                    ),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Gemm":
                module = Linear.from_onnx(onnx_node)
                self.pytorch_graph_module.add_submodule(onnx_node.outputs[0].name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    onnx_node.outputs[0].name,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Softmax":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.softmax,
                    (self.env[onnx_node.inputs[0].name],),
                    {"dim": -1},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Sigmoid":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    F.sigmoid,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "ReduceMean":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_method",
                    "mean",
                    (self.env[onnx_node.inputs[0].name],),
                    {
                        "dim": onnx_node.attrs["axes"],
                        "keepdim": bool(onnx_node.attrs["keepdims"]),
                    },
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Shape":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    getattr,
                    (self.env[onnx_node.inputs[0].name], "shape"),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "Gather":
                node = self.pytorch_graph_module.graph.create_node(
                    "call_function",
                    _operator.getitem,
                    (
                        self.env[onnx_node.inputs[0].name],
                        int(onnx_node.inputs[1].values),
                    ),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[onnx_node.outputs[0].name] = node
            elif onnx_node.op == "QuantizeLinear":
                dequant_node = onnx_node.o(0)
                assert(dequant_node.op == "DequantizeLinear")

                module = Observer(float(onnx_node.inputs[1].values), float(onnx_node.inputs[2].values))
                self.pytorch_graph_module.add_submodule(onnx_node.outputs[0].name, module)
                node = self.pytorch_graph.create_node(
                    "call_module",
                    onnx_node.outputs[0].name,
                    (self.env[onnx_node.inputs[0].name],),
                    {},
                    onnx_node.outputs[0].name,
                )
                self.env[dequant_node.outputs[0].name] = node
            elif onnx_node.op == "DequantizeLinear":
                pass                            
            else:
                raise NotImplementedError(
                    "Operator {} is not supported.".format(onnx_node.op)
                )

        for output in self.graph.outputs:
            node = self.pytorch_graph.output(self.env[output.name])
            self.env[output.name] = node

        self.pytorch_graph_module.graph.lint()
        self.pytorch_graph_module.recompile()
