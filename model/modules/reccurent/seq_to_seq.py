import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class SeqToSeq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, out=None):
        super(SeqToSeq, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.out = out

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        :return: An float tensor with shape of [batch_size, seq_len, out_size]
        """

        result, _ = self.rnn(input)

        if self.out is None:
            return result

        lengths = None
        is_packed_sequence = isinstance(result, PackedSequence)
        if is_packed_sequence:
            result, lengths = pad_packed_sequence(result, batch_first=True)

        result = self.out(result)
        return result if not is_packed_sequence else pack_padded_sequence(result, lengths, batch_first=True)
