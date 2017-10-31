import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.utils.weight_norm(nn.Linear(size, size)) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ f(G(x)) + (1 - σ(x)) ⨀ Q(x) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for i in range(self.num_layers):

            gate = F.sigmoid(self.gate[i](F.alpha_dropout(x, 0.3, self.training)))

            nonlinear = self.f(self.nonlinear[i](F.alpha_dropout(x, 0.3, self.training)))
            linear = self.linear[i](F.alpha_dropout(x, 0.3, self.training))

            x = gate * nonlinear + (1 - gate) * linear

        return x
