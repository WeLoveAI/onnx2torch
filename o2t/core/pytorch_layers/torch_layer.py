import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_buffer", torch.Tensor(), persistent=False)


class Tensor(BaseModule):
    # Tensor module is needed, because torch.tensor is not supported in torch.fx
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tensor(x).to(self.dummy_buffer.device)

    @classmethod
    def from_onnx(cls):
        return cls()


class Size(BaseModule):
    # Size module is needed, because torch.Size is not supported in torch.fx
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.Size(x)

    @classmethod
    def from_onnx(cls):
        return cls()


class Full(BaseModule):
    # Full module is needed, because torch.full is not supported in torch.fx
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        return torch.full(x, self.fill_value, device=self.dummy_buffer.device)

    @classmethod
    def from_onnx(cls, onnx_node):
        value = onnx_node.attrs.get("value")
        return cls(int(value.values))


class Reshape(nn.Module):
    # Reshape module is needed, because torch.reshape is not supported in torch.fx
    def __init__(self):
        super().__init__()

    def forward(self, x, shape):
        return torch.reshape(x, shape)

    @classmethod
    def from_onnx(cls):
        return cls()


class Arange(BaseModule):
    # Arange module is needed, because torch.reshape is not supported in torch.fx
    def __init__(self):
        super().__init__()

    def forward(self, start, end, step=1):
        return torch.arange(start, end, step, device=self.dummy_buffer.device)

    @classmethod
    def from_onnx(cls):
        return cls()


class Ones(BaseModule):
    # Ones module is needed, because torch.ones is not supported in torch.fx
    def __init__(self):
        super().__init__()

    def forward(self, size):
        return torch.ones(size, device=self.dummy_buffer.device)

    @classmethod
    def from_onnx(cls):
        return cls()
