import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.nn.utils.rnn import pack_padded_sequence


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.weight = xavier_uniform(self.embeddings.weight)

    def forward(self, input, lengths=None):
        result = self.embeddings(input)

        if lengths is not None:
            result = pack_padded_sequence(result, lengths, batch_first=True)

        return result
