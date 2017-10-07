from math import log, pi

import torch as t
import torch.nn as nn
from torch.autograd import Variable

from model.blocks import *
from model.blocks.abstract_vae_blocks.inference import InferenceBlock


class VAE(nn.Module):
    def __init__(self, vocab_size):
        super(VAE, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_size=15)

        self.inference = nn.ModuleList([
            InferenceBlock(
                input=SeqToSeq(input_size=15, hidden_size=15, num_layers=1, bidirectional=True),
                posterior=nn.Sequential(
                    SeqToVec(input_size=30, hidden_size=30, num_layers=1),
                    ParametersInference(input_size=60, latent_size=40, h_size=50)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=15 + 30, hidden_size=45, num_layers=1, bidirectional=True),
                posterior=nn.Sequential(
                    SeqToVec(input_size=90, hidden_size=70, num_layers=1),
                    ParametersInference(input_size=140, latent_size=15, h_size=100)
                ),
            )
        ])

    def forward(self):
        pass

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
