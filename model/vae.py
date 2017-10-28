from math import log, pi

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.modules import *


class VAE(nn.Module):
    def __init__(self, vocab_size, embeddings_path):
        super(VAE, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = 300
        self.embedding = Embedding(embeddings_path, self.vocab_size, embedding_size=self.embedding_size)

        self.inference = nn.ModuleList([
            InferenceBlock(
                input=SeqToSeq(input_size=self.embedding_size, hidden_size=100, num_layers=2),
                posterior=nn.Sequential(
                    SeqToVec(input_size=200, hidden_size=80, num_layers=2),
                    ParametersInference(input_size=320, latent_size=80, h_size=350)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=self.embedding_size + 200, hidden_size=80, num_layers=2),
                posterior=nn.Sequential(
                    SeqToVec(input_size=160, hidden_size=50, num_layers=2),
                    ParametersInference(input_size=200, latent_size=60, h_size=150)
                ),
                out=lambda x: x
            ),

            InferenceBlock(
                input=SeqToSeq(input_size=self.embedding_size + 160, hidden_size=40, num_layers=2),
                posterior=nn.Sequential(
                    SeqToVec(input_size=80, hidden_size=30, num_layers=2),
                    ParametersInference(input_size=120, latent_size=20, h_size=100)
                )
            )
        ])

        self.posterior_combination = nn.ModuleList([
            PosteriorCombination(80),
            PosteriorCombination(60)
        ])

        self.iaf = nn.ModuleList([
            IAF(latent_size=80, h_size=350),
            IAF(latent_size=60, h_size=150),
            IAF(latent_size=20, h_size=100),
        ])

        self.generation = nn.ModuleList([
            GenerativeBlock(
                posterior=ParametersInference(735, latent_size=80),
                input=nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(735, 120)),
                    nn.SELU(),
                    ResNet(1, 3),
                    nn.utils.weight_norm(nn.Linear(120, 100)),
                    nn.SELU()
                ),
                prior=ParametersInference(100, latent_size=80),
                out=nn.Sequential(
                    nn.utils.weight_norm(nn.ConvTranspose1d(10, 15, kernel_size=3, stride=1, padding=0, dilation=2)),
                    nn.SELU(),
                    ResNet(15, 3),
                    nn.utils.weight_norm(nn.ConvTranspose1d(15, 20, kernel_size=3, stride=1, padding=0, dilation=2)),
                    nn.SELU(),
                )
            ),

            GenerativeBlock(
                posterior=ParametersInference(255, latent_size=60),
                input=nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(255, 200)),
                    nn.SELU(),
                    ResNet(1, 3),
                    nn.utils.weight_norm(nn.Linear(200, 150)),
                    nn.SELU()
                ),
                prior=ParametersInference(150, latent_size=60),
                out=nn.Sequential(
                    nn.utils.weight_norm(nn.ConvTranspose1d(10, 12, kernel_size=3, stride=2, padding=0, dilation=2)),
                    nn.SELU(),
                    ResNet(12, 3, transpose=True),
                    nn.utils.weight_norm(nn.ConvTranspose1d(12, 15, kernel_size=3, stride=1, padding=0, dilation=2)),
                    nn.SELU(),
                )
            ),

            GenerativeBlock(
                out=nn.Sequential(
                    nn.utils.weight_norm(nn.ConvTranspose1d(10, 12, kernel_size=3, stride=2, padding=0, dilation=2)),
                    nn.SELU(),
                    ResNet(12, 3, transpose=True),
                    nn.utils.weight_norm(nn.ConvTranspose1d(12, 15, kernel_size=3, stride=2, padding=0, dilation=2)),
                    nn.SELU(),
                    ResNet(15, 4, transpose=True)
                )
            )
        ])

        self.out = VecToSeq(self.embedding_size, 520, hidden_size=140, num_layers=3,
                            out=nn.Sequential(
                                Highway(140, 3, nn.SELU()),
                                weight_norm(nn.Linear(140, vocab_size))
                            ))

        self.latent_size = [80, 60, 20]
        self.vae_length = len(self.inference)

    def forward(self, input, generator_input, lengths, generator_lengths, lambda_par):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :param generator_input: An long tensor with shape of [batch_size, seq_len]
        :param lengths: An list with length of batch_size with lengths of every batch sequence
        :param generator_lengths: An list with length of batch_size with lengths of every batch sequence
        :return: An float tensor with shape of [batch_size, seq_len, vocab_size]
        """

        batch_size = input.size(0)

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
                                 posterior=[mu, std],
                                 lambda_par=lambda_par)

        posterior = posterior.view(batch_size, 10, -1)
        prior = prior.view(batch_size, 10, -1)

        posterior = self.generation[-1](posterior)
        prior = self.generation[-1](prior)

        posterior = posterior.view(batch_size, -1)
        prior = prior.view(batch_size, -1)

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

            posterior_mu, posterior_std = self.posterior_combination[i]([
                top_down_mu, bottom_up_mu, top_down_std, bottom_up_std
            ])

            eps = Variable(t.randn(*posterior_mu.size()))
            if cuda:
                eps = eps.cuda()

            posterior_gauss = eps * posterior_std + posterior_mu
            posterior, log_det = self.iaf[i](posterior_gauss, h)

            prior_mu, prior_std, _ = self.generation[i].inference(prior_determenistic, 'prior')

            kld += VAE.kl_divergence(z=posterior,
                                     z_gauss=posterior_gauss,
                                     log_det=log_det,
                                     posterior=[posterior_mu, posterior_std],
                                     prior=[prior_mu, prior_std],
                                     lambda_par=lambda_par)

            posterior = self.generation[i].out(t.cat([posterior, posterior_determenistic], 1).view(batch_size, 10, -1))
            posterior = posterior.view(batch_size, -1)

            if i != 0:
                '''
                Since there no level below bottom-most, 
                there no reason to pass prior through out operation
                '''

                eps = Variable(t.randn(*prior_mu.size()))
                if cuda:
                    eps = eps.cuda()

                prior = eps * prior_std + prior_mu

                prior = self.generation[i].out(t.cat([prior, prior_determenistic], 1).view(batch_size, 10, -1))
                prior = prior.view(batch_size, -1)

        return self.out(posterior, generator_input)[0], kld

    @staticmethod
    def kl_divergence(**kwargs):

        log_p_z_x = VAE.log_gauss(kwargs['z_gauss'], kwargs['posterior']) - kwargs['log_det']

        if kwargs.get('prior') is None:
            kwargs['prior'] = [Variable(t.zeros(*kwargs['z'].size())),
                               Variable(t.ones(*kwargs['z'].size()))]

        lambda_par = Variable(t.FloatTensor([kwargs['lambda_par']]))

        if kwargs['z'].is_cuda:
            lambda_par = lambda_par.cuda()
            kwargs['prior'] = [var.cuda() for var in kwargs['prior']]

        log_p_z = VAE.log_gauss(kwargs['z'], kwargs['prior'])

        result = log_p_z_x - log_p_z
        return t.max(t.stack([result.mean(), lambda_par]), 0)[0]

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + log(2 * pi)).sum(1)

    def sample(self, max_seq_len, cuda, batch_loader, z=None):
        """
        :param z: An array of variables from normal distribution each with shape of [batch_size, latent_size[i]]
        :return: Sample from generative model
        """

        if z is None:
            z = [Variable(t.randn(1, size)) for size in self.latent_size]

            if cuda:
                z = [var.cuda() for var in z]

        top_variable = z[-1].view(1, 10, -1)

        out = self.generation[-1].out(top_variable)
        out = out.view(1, -1)

        for i in range(self.vae_length - 2, -1, -1):
            determenistic = self.generation[i].input(out)

            [mu, std, _] = self.generation[i].prior(determenistic)
            prior = z[i] * std + mu
            out = self.generation[i].out(t.cat([prior, determenistic], 1).view(1, 10, -1))
            out = out.view(1, -1)

        z = out
        del out

        initial_state = None

        input = batch_loader.go_input(1, cuda)
        input = self.embedding(input)

        result = ''

        for i in range(max_seq_len):

            logits, initial_state = self.out(z, input, initial_state=initial_state)

            logits = logits.view(-1, self.vocab_size)
            prediction = F.softmax(logits).data.cpu().numpy()[-1]

            idx, char = batch_loader.sample_char(prediction)

            if char == batch_loader.stop_token:
                break

            result += char

            input = Variable(t.from_numpy(np.array([[idx]])))
            if cuda:
                input = input.cuda()
            input = self.embedding(input)

        return result

    def loss(self, input, generator_input, lengths, generator_lengths, target, criterion, lambda_par, average=True):

        out, kld = self(input, generator_input, lengths, generator_lengths, lambda_par)
        out = pad_packed_sequence(out, batch_first=True)[0]

        [batch_size, _] = target.size()

        out = out.contiguous().view(-1, self.vocab_size)
        target = target.view(-1)
        likelihood = criterion(out, target) / (batch_size if average else 1)

        return likelihood, kld

    def learnable_parameters(self):
        for par in self.parameters():
            if par.requires_grad:
                yield par
