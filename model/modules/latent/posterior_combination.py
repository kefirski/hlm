import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm

from model.modules.conv.resnet import ResNet
from model.modules.utils.squeeze import Squeeze


class PosteriorCombination(nn.Module):
    def __init__(self, size):
        super(PosteriorCombination, self).__init__()

        self.mu_weight = nn.Sequential(
            weight_norm(nn.Linear(4 * size, size)),
            nn.SELU(),

            weight_norm(nn.Linear(size, size)),
            nn.Sigmoid()
        )

        self.std_weight = nn.Sequential(
            weight_norm(nn.Linear(4 * size, size)),
            nn.SELU(),

            weight_norm(nn.Linear(size, size)),
            nn.Sigmoid()
        )

    def forward(self, posterior):
        [mu_1, mu_2, std_1, std_2] = posterior

        input = t.cat(posterior, 1)

        mu_weight = self.mu_weight(input)
        std_weight = self.std_weight(input)

        mu = mu_1 * mu_weight + mu_2 * (1 - mu_weight)
        std = std_1 * std_weight + std_2 * (1 - std_weight)

        return mu, std
