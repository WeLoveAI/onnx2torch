import torch
import torch.nn as nn


class Shape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tensor(x.shape)

    @classmethod
    def from_onnx(cls, mod):

        return cls()
