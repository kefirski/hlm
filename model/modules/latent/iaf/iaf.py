import torch as t
import torch.nn as nn

from model.modules.conv.resnet import ResNet
from .autoregressive_linear import AutoregressiveLinear


class IAF(nn.Module):
    def __init__(self, latent_size, h_size, depth):
        super(IAF, self).__init__()

        self.z_size = latent_size
        self.h_size = h_size
        self.depth = depth

        self.h = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(self.h_size, self.h_size * 2)),
            ResNet(1, 4, dim=1),
            nn.utils.weight_norm(nn.Linear(self.h_size * 2, self.h_size)),
            ResNet(1, 4, dim=1)
        )

        self.m = nn.ModuleList([
            nn.Sequential(
                AutoregressiveLinear(self.z_size + self.h_size, self.z_size),
                nn.ELU(),
                AutoregressiveLinear(self.z_size, self.z_size)
            )

            for _ in range(depth)
        ])

        self.s = nn.ModuleList([
            nn.Sequential(
                AutoregressiveLinear(self.z_size + self.h_size, self.z_size),
                nn.ELU(),
                AutoregressiveLinear(self.z_size, self.z_size)
            )

            for _ in range(depth)
        ])

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the iaf mapping Jacobian
        """

        h = self.h(h)

        log_det = 0
        for i in range(self.depth):
            input = t.cat([z, h], 1)

            m = self.m[i](input)
            s = self.s[i](input)

            z = s.exp() * z + m

            log_det += s.sum(1)

        return z, log_det
