import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Observer(nn.Module):
    def __init__(self, scale, zero_point, input_shape=None):
        super(Observer, self).__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.input_shape = input_shape

    def calculate_qparams(self):
        return self.scale, self.zero_point

    def forward(self, x):
        return x
