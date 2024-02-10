import os
import warnings

import pytest
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
    def test_torchvision(self, request, model, shape=(1, 3, 224, 224), fuse=FUSE):
        model = model(pretrained=PRETRAINED)
        x = torch.rand(shape)
        torch.onnx.export(model, x, "tmp/" + request.node.name + ".onnx")

        convert(
            "tmp/" + request.node.name + ".onnx",
            "tmp/" + request.node.name + ".gm",
            check=True,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "tests/test_onnx_nets.py"])
