import torch as t
import torch.nn as nn
from torch.nn.utils import weight_norm

from model.modules.conv.resnet import ResNet
from model.modules.utils.squeeze import Squeeze


class PosteriorCombination(nn.Module):
    def __init__(self, size):
        super(PosteriorCombination, self).__init__()

    def forward(self, posterior):
        [mu_1, mu_2, std_1, std_2] = posterior

        return mu_1 + mu_2, std_1 + std_2
