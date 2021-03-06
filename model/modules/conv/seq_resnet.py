import torch.nn as nn
import torch.nn.functional as F


class SeqResNet(nn.Module):
    def __init__(self, size, num_layers, dim, transpose=False):
        super(SeqResNet, self).__init__()

        self.num_layers = num_layers
        self.size = size

        self.conv = nn.ModuleList([
            nn.Sequential(
                self.conv3x3(1, transpose),
                nn.SELU(),

                self.conv3x3(1, transpose),
            )

            for _ in range(num_layers)
        ])

    def forward(self, input):

        batch_size = input.size(0)
        input = input.view(-1, 1, self.size)

        for layer in self.conv:
            input = layer(input) + input
            input = F.elu(input)

        return input.view(batch_size, -1, self.size)

    @staticmethod
    def conv3x3(size, transpose):

        if transpose:
            return nn.utils.weight_norm(
                nn.ConvTranspose1d(size, size, kernel_size=3, stride=1, padding=1, bias=False)
            )

        return nn.utils.weight_norm(
            nn.Conv1d(size, size, kernel_size=3, stride=1, padding=1, bias=False)
        )