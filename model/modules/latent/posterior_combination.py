import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm

from model.modules.conv.resnet import ResNet
from model.modules.utils.squeeze import Squeeze


class PosteriorCombination(nn.Module):
    def __init__(self, size):
        super(PosteriorCombination, self).__init__()

        self.mu_combination = nn.Sequential(
            weight_norm(nn.Conv1d(4, 3, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)),
            nn.SELU(),

            ResNet(3, num_layers=2),

            weight_norm(nn.Conv1d(3, 1, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)),
            nn.SELU(),

            Squeeze(1),

            weight_norm(nn.Linear(size, size))
        )

        self.std_combination = nn.Sequential(
            weight_norm(nn.Conv1d(4, 3, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)),
            nn.SELU(),

            ResNet(3, num_layers=2),

            weight_norm(nn.Conv1d(3, 1, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)),
            nn.SELU(),

            Squeeze(1),

            weight_norm(nn.Linear(size, size))

        )

    def forward(self, posterior):
        input = t.stack(posterior, 1)

        mu = self.mu_combination(input)
        std = (0.5 * self.std_combination(input)).exp()

        return mu, std
