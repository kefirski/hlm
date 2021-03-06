import argparse
from math import tanh

import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Adam

from model.vae import VAE
from utils.ptb_dataloader.ptb_loader import PTBLoader


def kl_coef(x):
    return tanh(x / 8000)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='hlm')
    parser.add_argument('--num-iterations', type=int, default=150_000, metavar='NI',
                        help='num iterations (default: 150_000)')
    parser.add_argument('--batch-size', type=int, default=40, metavar='BS',
                        help='batch size (default: 40)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                        help='dropout rate (default: 0.0)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    writer = SummaryWriter(args.tensorboard)

    dataloader = PTBLoader('utils/ptb_dataloader/data/')

    vae = VAE(vocab_size=dataloader.vocab_size)
    if args.use_cuda:
        vae = vae.cuda()

    optimizer = Adam(vae.learnable_parameters(), args.learning_rate, eps=1e-6)

    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    kl_par = lambda x: 18 if x < 10000 else 12

    for iteration in range(args.num_iterations):

        (input, lengths), (gen_input, gen_lengths), target = \
            dataloader.torch_batch(args.batch_size, 'train', args.use_cuda, args.dropout)

        optimizer.zero_grad()

        likelihood, kld = vae.loss(input, gen_input, lengths, gen_lengths, target, criterion,
                                   kl_par(iteration))
        loss = likelihood + kld * kl_coef(iteration)

        loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            (input, lengths), (gen_input, gen_lengths), target = \
                dataloader.torch_batch(args.batch_size, 'valid', args.use_cuda, 0., volatile=True)

            likelihood, kld = vae.loss(input, gen_input, lengths, gen_lengths, target, criterion,
                                       kl_par(iteration), eval=True, average=False)

            likelihood = likelihood.cpu().data
            kld = kld.cpu().data

            writer.add_scalar('likelihood', likelihood / sum(lengths), iteration)
            writer.add_scalar('kld', kld, iteration)
            writer.add_scalar('NLL', likelihood / args.batch_size, iteration)

            print('iteration {}, likelihood {} likelihood seq {} kld {}'.format(iteration, likelihood.numpy() / args.batch_size,
                                                                                likelihood.numpy() / sum(lengths), kld.numpy()))


            sampling = vae.sample(200, args.use_cuda, dataloader)
            print('_________')
            print(sampling)
