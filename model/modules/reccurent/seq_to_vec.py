import torch.nn as nn


class SeqToVec(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super(SeqToVec, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        :return: An float tensor with shape of [batch_size, hidden_size * 2]
        """

        _, result = self.rnn(input)

        batch_size = result.size(1)
        return result.transpose(0, 1).contiguous().view(batch_size, -1)
