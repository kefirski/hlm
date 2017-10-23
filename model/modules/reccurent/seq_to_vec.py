import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from model.modules.conv.resnet import ResNet


class SeqToVec(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SeqToVec, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(input_size, input_size, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.SELU(),

            ResNet(input_size, num_layers=3),

            nn.ConvTranspose1d(input_size, hidden_size, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.SELU(),

            ResNet(input_size, num_layers=3),

            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        """

        if isinstance(input, PackedSequence):
            input, _ = pad_packed_sequence(input, batch_first=True)

        input = input.transpose(1, 2)
        return self.net(input).mean(2)
