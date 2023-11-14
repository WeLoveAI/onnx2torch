import os

import pytest
import warnings
import torch
import torchvision.models as models

from brocolli.converter.pytorch_onnx_parser import PytorchOnnxParser
from onnx2torch.onnx_pytorch_parser import OnnxPytorchParser

FUSE = True
PRETRAINED = False

os.makedirs("tmp", exist_ok=True)


class TestTorchVisionClass:
    @pytest.mark.parametrize("use_onnx_export", (True, False))
    @pytest.mark.parametrize(
        "model",
        (
            models.resnet18,
            models.alexnet,
            models.squeezenet1_0,
            models.googlenet,
        ),
    )
    def test_torchvision(
        self, request, use_onnx_export, model, shape=(1, 3, 224, 224), fuse=FUSE
    ):
        model = model(pretrained=PRETRAINED)
        x = torch.rand(shape)
        runner = PytorchOnnxParser(model, x, fuse)
        if use_onnx_export:
            runner.export_onnx("tmp/" + request.node.name + ".onnx")
        else:
            runner.convert()
            runner.save("tmp/" + request.node.name + ".onnx")
            runner.check_result()

        pytorch_parser = OnnxPytorchParser("tmp/" + request.node.name + ".onnx")
        graph_module = pytorch_parser.convert()

        pytorch_out = model(x)
        graph_module_out = graph_module(x)
        tol = 1e-5
        torch.testing.assert_close(pytorch_out, graph_module_out, rtol=tol, atol=tol)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "test/test_onnx_nets.py"])
