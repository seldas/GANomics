import torch
import torch.nn as nn
import math

class FullyConnectedGenerator(nn.Module):
    """
    Generator using 1x1 convolutions to emulate fully connected layers over gene features.
    This behaves as a dense mapping over the feature axis, suitable for transcriptomics.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(FullyConnectedGenerator, self).__init__()
        # Use 1x1 convolutions to act as dense layers on the channel dimension
        # Input shape expected: (batch, genes, 1, 1) or (batch, genes)
        self.C1 = nn.Conv2d(input_dim, math.floor(input_dim / 2), kernel_size=1)
        self.C3 = nn.Conv2d(math.floor(input_dim / 2), output_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Ensure input is 4D (batch, channels, height, width) for Conv2d
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            x = x.unsqueeze(-1)
            
        x = self.relu(self.C1(x))
        x = self.C3(x)
        return x

class FullyConnectedDiscriminator(nn.Module):
    """
    Discriminator using 1x1 convolutions to classify real vs synthetic genomic profiles.
    """
    def __init__(self, input_dim: int):
        super(FullyConnectedDiscriminator, self).__init__()
        self.C1 = nn.Conv2d(input_dim, math.floor(input_dim / 2), kernel_size=1)
        self.C3 = nn.Conv2d(math.floor(input_dim / 2), 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            x = x.unsqueeze(-1)
            
        x = self.relu(self.C1(x))
        x = self.C3(x)
        # Output is typically passed through a GAN loss which might include Sigmoid
        return x

class GANLoss(nn.Module):
    """
    Helper class to calculate GAN losses (LSGAN or Vanilla).
    """
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
