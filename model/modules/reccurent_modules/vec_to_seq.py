import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class VecToSeq(nn.Module):
    def __init__(self, input_size, z_size, hidden_size, num_layers, out=None):
        super(VecToSeq, self).__init__()

        self.input_size = input_size
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=self.input_size + self.z_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.out = out

    def forward(self, z, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, input_size]
        :param z: An float tensor with shape of [batch_size, z_size]
        :return: An float tensor with shape of [batch_size, seq_len, hidden_size]
        """

        is_packed_seq = isinstance(input[0], PackedSequence)

        lengths = None
        if is_packed_seq:
            tmp = [pad_packed_sequence(var, batch_first=True) for var in input]

            input = [var[0] for var in tmp]
            lengths = [var[1] for var in tmp]

        [_, seq_len, _] = input[0].size()
        z = z.unsqueeze(1).repeat(1, seq_len, 1)

        input = t.cat(input + [z], 2)

        if is_packed_seq:
            input = pack_padded_sequence(input, lengths[0], batch_first=True)

        result, _ = self.rnn(input)

        if self.out is None:
            return result

        if is_packed_seq:
            result, lengths = pad_packed_sequence(result, batch_first=True)

        result = self.out(result)
        return result if not is_packed_seq else pack_padded_sequence(result, lengths, batch_first=True)

