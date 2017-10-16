import torch.nn as nn


class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()

        self.size = args

    def forward(self, input):
        return input.view(*self.size)
