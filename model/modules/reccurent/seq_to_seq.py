import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from model.modules.conv.resnet import ResNet


class SeqToSeq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, out=None):
        super(SeqToSeq, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_size * 2, hidden_size * 2,
                               kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.SELU(),

            ResNet(hidden_size * 2, 2),

            nn.ConvTranspose1d(hidden_size * 2, hidden_size,
                               kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.SELU()
        )

        self.out = out

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        :return: An float tensor with shape of [batch_size, seq_len, out_size]
        """

        result, _ = self.rnn(input)

        lengths = None
        is_packed_sequence = isinstance(result, PackedSequence)
        if is_packed_sequence:
            result, lengths = pad_packed_sequence(result, batch_first=True)

        result = result.transpose(1, 2)
        result = self.net(result).transpose(1, 2)

        result = result if not is_packed_sequence else pack_padded_sequence(result, lengths, batch_first=True)

        return result if self.out is None else self.out(result)
