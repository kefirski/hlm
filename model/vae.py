from math import log, pi

import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.blocks import *


class VAE(nn.Module):
    def __init__(self, vocab_size):
        super(VAE, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = Embedding(self.vocab_size, embedding_size=15)

        self.inference = nn.ModuleList([
            InferenceBlock(
                input=SeqToSeq(input_size=15, hidden_size=15, num_layers=1, bidirectional=True),
                posterior=nn.Sequential(
                    SeqToVec(input_size=30, hidden_size=30, num_layers=1),
                    ParametersInference(input_size=60, latent_size=55, h_size=60)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=15 + 30, hidden_size=45, num_layers=1, bidirectional=True),
                posterior=nn.Sequential(
                    SeqToVec(input_size=90, hidden_size=70, num_layers=1),
                    ParametersInference(input_size=140, latent_size=20, h_size=120)
                )
            )
        ])

        self.iaf = nn.ModuleList(
            [
                IAF(latent_size=55, h_size=60),
                IAF(latent_size=20, h_size=120)
            ]
        )

        self.vae_length = len(self.inference)

    def forward(self, input):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :return: An float tensor with shape of [batch_size, seq_len, vocab_size]
        """

        [batch_size, seq_len] = input.size()
        cuda = input.is_cuda

        posterior_parameters = []

        '''
        Here we perform top-down inference.
        parameters array is filled with posterior parameters [mu, std, h]
        '''
        for i in range(self.vae_length):

            if i < self.vae_length - 1:
                out, parameters = self.inference[i](input)

                [out, lengths] = pad_packed_sequence(out, batch_first=True)
                [input, _] = pad_packed_sequence(input, batch_first=True)

                input = t.cat([input, out], 2)
                input = pack_padded_sequence(input, lengths, batch_first=True)

            else:
                parameters = self.inference[i](input)

            posterior_parameters += [parameters]

    @staticmethod
    def kl_divergence(**kwargs):

        log_p_z_x = VAE.log_gauss(kwargs['z_gauss'], kwargs['posterior']) - kwargs['log_det']

        if kwargs.get('prior') is None:
            kwargs['prior'] = [Variable(t.zeros(*kwargs['z'].size())),
                               Variable(t.ones(*kwargs['z'].size()))]

        lambda_par = Variable(t.FloatTensor([1]))

        if kwargs['z'].is_cuda:
            lambda_par = lambda_par.cuda()
            for var in kwargs['prior']:
                var.cuda()
        log_p_z = VAE.log_gauss(kwargs['z'], kwargs['prior'])

        result = log_p_z_x - log_p_z
        return t.max(t.stack([result.mean(), lambda_par]), 0)[0]

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)
