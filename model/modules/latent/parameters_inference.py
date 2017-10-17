import torch.nn as nn

from ..conv.resnet import ResNet


class ParametersInference(nn.Module):
    def __init__(self, input_size, latent_size, h_size=None):
        super(ParametersInference, self).__init__()

        self.mu = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size, input_size)),
            ResNet(1, num_layers=4),
            nn.utils.weight_norm(nn.Linear(input_size, latent_size))
        )

        self.std = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size, input_size)),
            ResNet(1, num_layers=4),
            nn.utils.weight_norm(nn.Linear(input_size, latent_size))
        )

        self.h = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_size, input_size)),
            ResNet(1, num_layers=4),
            nn.utils.weight_norm(nn.Linear(input_size, h_size)),
            nn.ELU()
        ) if h_size is not None else None

    def forward(self, input):
        mu = self.mu(input)
        std = (0.5 * self.std(input)).exp()
        h = self.h(input) if self.h is not None else None

        return [mu, std, h]
