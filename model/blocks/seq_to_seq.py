import torch.nn as nn


class SeqToVec(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(SeqToVec, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        :return: An float tensor with shape of [batch_size, seq_len, out_size]
        """

        result, _ = self.rnn(input)
        return result