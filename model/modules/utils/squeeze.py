import torch.nn as nn


class Squeeze(nn.Module):
    def __init__(self, position):
        super(Squeeze, self).__init__()

        self.position = position

    def forward(self, input):
        return input.squeeze(self.position)
