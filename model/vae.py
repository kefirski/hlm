from math import log, pi

import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.modules import *


class VAE(nn.Module):
    def __init__(self, vocab_size):
        super(VAE, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = Embedding(self.vocab_size, embedding_size=15)

        self.inference = nn.ModuleList([
            InferenceBlock(
                input=SeqToSeq(input_size=15, hidden_size=15, num_layers=1),
                posterior=nn.Sequential(
                    SeqToVec(input_size=30, hidden_size=30, num_layers=1),
                    ParametersInference(input_size=60, latent_size=55, h_size=60)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=15 + 30, hidden_size=45, num_layers=1),
                posterior=nn.Sequential(
                    SeqToVec(input_size=90, hidden_size=70, num_layers=1),
                    ParametersInference(input_size=140, latent_size=20, h_size=120)
                )
            )
        ])

        self.iaf = nn.ModuleList([
            IAF(latent_size=55, h_size=60),
            IAF(latent_size=20, h_size=120)
        ])

        self.generation = nn.ModuleList([
            GenerativeBlock(
                posterior=nn.Sequential(
                    SeqToVec(input_size=vocab_size, hidden_size=25, num_layers=1),
                    ParametersInference(input_size=50, latent_size=55)
                ),
                input=SeqToSeq(input_size=vocab_size, hidden_size=25, num_layers=1),
                prior=nn.Sequential(
                    SeqToVec(input_size=50, hidden_size=25, num_layers=1),
                    ParametersInference(input_size=50, latent_size=55)
                ),
                out=VecToSeq(50 + 15, z_size=55, hidden_size=40, num_layers=1,
                             out=weight_norm(nn.Linear(40, vocab_size))),
            ),

            GenerativeBlock(
                out=VecToSeq(15, z_size=20, hidden_size=30, num_layers=2, out=weight_norm(nn.Linear(30, vocab_size))),
            )
        ])

        self.out = SeqToSeq(15 + 15, 15, 1, bidirectional=True, out=weight_norm(nn.Linear(30, vocab_size))),

        self.latent_size = [55, 20]
        self.vae_length = len(self.inference)

    def forward(self, input, generator_input, lengths, generator_lengths):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :param generator_input: An long tensor with shape of [batch_size, seq_len]
        :param lengths: An list with length of batch_size with lengths of every batch sequence
        :param generator_lengths: An list with length of batch_size with lengths of every batch sequence
        :return: An float tensor with shape of [batch_size, seq_len, vocab_size]
        """

        cuda = input.is_cuda

        posterior_parameters = []

        input = self.embedding(input, lengths)
        generator_input = self.embedding(generator_input)
        packed_generator_input = pack_padded_sequence(generator_input, generator_lengths, True)

        for i in range(self.vae_length):

            if i < self.vae_length - 1:
                out, parameters = self.inference[i](input)

                [out, lengths] = pad_packed_sequence(out, batch_first=True)
                [input, _] = pad_packed_sequence(input, batch_first=True)

                input = t.cat([input, out], 2)
                input = pack_padded_sequence(input, lengths, batch_first=True)

            else:
                parameters = self.inference[i](input)

            posterior_parameters.append(parameters)

        [mu, std, h] = posterior_parameters[-1]

        prior = Variable(t.randn(*mu.size()))
        eps = Variable(t.randn(*mu.size()))

        if cuda:
            prior, eps = prior.cuda(), eps.cuda()

        posterior_gauss = eps * std + mu
        posterior, log_det = self.iaf[-1](posterior_gauss, h)

        kld = self.kl_divergence(z=posterior,
                                 z_gauss=posterior_gauss,
                                 log_det=log_det,
                                 posterior=[mu, std])

        posterior = self.generation[-1](posterior, [packed_generator_input])
        prior = self.generation[-1](prior, [packed_generator_input])

        for i in range(self.vae_length - 2, -1, -1):

            posterior_determenistic = self.generation[i].input(posterior)
            prior_determenistic = self.generation[i].input(prior)

            [top_down_mu, top_down_std, _] = self.generation[i].inference(posterior, 'posterior')
            [bottom_up_mu, bottom_up_std, h] = posterior_parameters[i]

            posterior_mu = top_down_mu + bottom_up_mu
            posterior_std = top_down_std + bottom_up_std

            eps = Variable(t.randn(*posterior_mu.size()))
            if cuda:
                eps.cuda()

            posterior_gauss = eps * posterior_std + posterior_mu
            posterior, log_det = self.iaf[i](posterior_gauss, h)

            prior_mu, prior_std, _ = self.generation[i].inference(prior_determenistic, 'prior')

            kld += VAE.kl_divergence(z=posterior,
                                     z_gauss=posterior_gauss,
                                     log_det=log_det,
                                     posterior=[posterior_mu, posterior_std],
                                     prior=[prior_mu, prior_std])

            posterior = self.generation[i].out(posterior, [posterior_determenistic, packed_generator_input])

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
