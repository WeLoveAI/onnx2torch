import torch


class InspectorModule(torch.nn.Module):
    def __init__(self):
        super(InspectorModule, self).__init__()

    def forward(self, *inputs):
        print(inputs)

        return inputs
