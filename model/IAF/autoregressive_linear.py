import math

import torch as t
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class AutoregressiveLinear(nn.Module):
    def __init__(self, input_size, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.weight = Parameter(t.Tensor(self.input_size, self.out_size))

        if bias:
            self.bias = Parameter(t.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size]
        """

        if self.bias is not None:
            return t.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1) + self.bias
        return output
