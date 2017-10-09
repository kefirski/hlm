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
                input=SeqToSeq(input_size=15, hidden_size=40, num_layers=1),
                posterior=nn.Sequential(
                    SeqToVec(input_size=80, hidden_size=80, num_layers=1),
                    ParametersInference(input_size=160, latent_size=100, h_size=150)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=15 + 80, hidden_size=100, num_layers=1),
                posterior=nn.Sequential(
                    SeqToVec(input_size=200, hidden_size=50, num_layers=1),
                    ParametersInference(input_size=100, latent_size=50, h_size=100)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=15 + 200, hidden_size=50, num_layers=1),
                posterior=nn.Sequential(
                    SeqToVec(input_size=100, hidden_size=25, num_layers=1),
                    ParametersInference(input_size=50, latent_size=10, h_size=50)
                )
            )
        ])

        self.iaf = nn.ModuleList([
            IAF(latent_size=100, h_size=150),
            IAF(latent_size=50, h_size=100),
            IAF(latent_size=10, h_size=50),
        ])

        self.generation = nn.ModuleList([
            GenerativeBlock(
                posterior=ParametersInference(100, latent_size=100),
                input=nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(100, 100)),
                    nn.SELU()
                ),
                prior=ParametersInference(100, latent_size=100),
                out=nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(100 + 100, 220)),
                    nn.SELU(),
                    nn.utils.weight_norm(nn.Linear(220, 240)),
                    nn.SELU()
                )
            ),

            GenerativeBlock(
                posterior=ParametersInference(30, latent_size=50),
                input=nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(30, 40)),
                    nn.SELU()
                ),
                prior=ParametersInference(40, latent_size=50),
                out=nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(50 + 40, 90)),
                    nn.SELU(),
                    nn.utils.weight_norm(nn.Linear(90, 100)),
                    nn.SELU()
                )
            ),

            GenerativeBlock(
                out=nn.Sequential(
                    weight_norm(nn.Linear(10, 15)),
                    nn.SELU(),
                    nn.utils.weight_norm(nn.Linear(15, 30)),
                    nn.SELU()
                )
            )
        ])

        self.out = VecToSeq(15, 240, hidden_size=140, num_layers=2, out=weight_norm(nn.Linear(140, vocab_size)))

        self.latent_size = [100, 50, 10]
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

        '''
        Pickup embeddings for input sequences
        
        Residual input is used in order to 
        concat determenistic output from every layer with input sequence
        '''
        input = self.embedding(input)
        residual = input
        input = pack_padded_sequence(input, lengths, True)

        generator_input = self.embedding(generator_input, generator_lengths)

        '''
        Bottom-up inference
        Input goes through input op in order to obtain determenistic features
        that will be used to get parameters of posterior
        and to concat with residual input
        '''
        for i in range(self.vae_length):

            if i < self.vae_length - 1:
                out, parameters = self.inference[i](input)

                [out, _] = pad_packed_sequence(out, batch_first=True)

                input = t.cat([residual, out], 2)
                input = pack_padded_sequence(input, lengths, batch_first=True)

            else:
                parameters = self.inference[i](input)

            posterior_parameters.append(parameters)

        '''
        Generation on top-most layer
        
        After sampling latent variables from diagonal gaussian, 
        they are passed through IAF and then KLD is approximated with Monte Carlo method
        '''

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

        posterior = self.generation[-1](posterior)
        prior = self.generation[-1](prior)

        '''
        Top-down inference
        
        Quite similar to generation on top-most layer, 
        but for now parameters of posterior ar combined with bottom-up information
        '''
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

            posterior = self.generation[i].out(t.cat([posterior, posterior_determenistic], 1))

            if i != 0:
                '''
                Since there no level below bottom-most, 
                there no reason to pass prior through out operation
                '''

                eps = Variable(t.randn(*prior_mu.size()))
                if cuda:
                    eps = eps.cuda()

                prior = eps * prior_std + prior_mu

                prior = self.generation[i].out(t.cat([prior, prior_determenistic], 1))

        return self.out(posterior, generator_input)

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
