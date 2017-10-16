import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, size, num_layers):
        super(ResNet, self).__init__()

        self.num_layers = num_layers
        self.size = size

        self.conv = nn.ModuleList([
            nn.Sequential(
                self.conv3x3(size),
                nn.ELU(),

                self.conv3x3(size),
            )

            for _ in range(num_layers)
        ])

    def forward(self, input):

        batch_size = input.size(0)

        should_view = False
        if len(input.size()) == 2:
            input = input.view(batch_size, self.size, -1)
            should_view = True

        for layer in self.conv:
            input = layer(input) + input
            input = F.elu(input)

        return input.view(batch_size, -1) if should_view else input

    @staticmethod
    def conv3x3(size):
        return nn.utils.weight_norm(
            nn.Conv1d(size, size, kernel_size=3, stride=1, padding=1, bias=False)
        )
