import os
import warnings

import pytest
import timm
import torch
import torchvision.models as models

from onnx2torch import convert


FUSE = True
PRETRAINED = False

os.makedirs("tmp", exist_ok=True)


class TestTorchVisionClass:
    @pytest.mark.parametrize(
        "model",
        (
            models.resnet18,
            models.alexnet,
            models.squeezenet1_0,
            models.googlenet,
        ),
    )
    def test_torchvision(self, request, model, shape=(1, 3, 224, 224)):
        model = model(pretrained=PRETRAINED)
        x = torch.rand(shape)
        torch.onnx.export(model, x, "tmp/" + request.node.name + ".onnx")

        convert(
            "tmp/" + request.node.name + ".onnx",
            "tmp/" + request.node.name + ".gm",
            check=True,
        )


class TestTimmClass:
    @pytest.fixture(params=timm.list_models())
    def model_name(self, request):
        yield request.param

    def test_timm(self, request, model_name):
        model = timm.create_model(model_name, pretrained=PRETRAINED)
        input_size = model.default_cfg.get("input_size")
        x = torch.randn((1,) + input_size)
        torch.onnx.export(model, x, "tmp/" + request.node.name + ".onnx")

        convert(
            "tmp/" + request.node.name + ".onnx",
            "tmp/" + request.node.name + ".gm",
            check=True,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "tests/test_onnx_nets.py"])
