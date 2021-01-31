import torch
from torch import nn
import torch.nn.functional as F

class ResizeConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 # scale_factor,
                 padding_mode="valid",
                 activation=torch.relu,
                 stride=1,
                 mode='nearest'):
        super().__init__()

        self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                       kernel_size, stride=stride,
                                       padding=(kernel_size //
                                                2) if padding_mode == "same" else 0,
                                       bias=False
                                       )
        self.bn = nn.BatchNorm3d(out_channels,eps=0.001)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3Layer(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride=1,
                 padding_mode="valid",
                 activation=torch.relu):
        super().__init__()

        self.activation = activation

        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=filter_size,
                              stride=stride,
                              padding=(filter_size //
                                       2) if padding_mode == "same" else 0,
                              bias=False
                              )
        self.bn = nn.BatchNorm3d(out_channels,eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_latents=100):
        super().__init__()
        self.conv1 = Conv3Layer(1, 8, 3,
                                padding_mode="valid",
                                stride=1,
                                activation=torch.nn.ELU())

        self.conv2 = Conv3Layer(8, 16, 3,
                                padding_mode="same",
                                stride=2,
                                activation=torch.nn.ELU())
        self.conv3 = Conv3Layer(16, 32, 3,
                                padding_mode="valid",
                                stride=1,
                                activation=torch.nn.ELU())
        self.conv4 = Conv3Layer(32, 64, 3,
                                padding_mode="same",
                                stride=2,
                                activation=torch.nn.ELU())

        self.dense = nn.Sequential(
            torch.nn.Linear(7**3*64, 343, bias=False),
            torch.nn.BatchNorm1d(343,eps=0.001),
            torch.nn.ELU(),
        )
        self.mu = nn.Sequential(
            torch.nn.Linear(343, num_latents, bias=False),
            torch.nn.BatchNorm1d(num_latents,eps=0.001)
        )
        self.logsigma = nn.Sequential(
            torch.nn.Linear(343, num_latents, bias=False),
            torch.nn.BatchNorm1d(num_latents,eps=0.001)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        return mu, logsigma
        #return self.dense2(x)


class Decoder(nn.Module):
    def __init__(self, num_latents=100):
        super().__init__()
        self.dense = nn.Sequential(
            torch.nn.Linear(num_latents, 343, bias=False),
            torch.nn.BatchNorm1d(343,eps=0.001),
            torch.nn.ELU(),
        )
        self.deconv1 = Conv3Layer(1, 64, 3,
                                  padding_mode="same",
                                  stride=1,
                                  activation=torch.nn.ELU())
        self.deconv2 = ResizeConv3d(64, 32, 3,
                                    padding_mode="valid",
                                    stride=2,
                                    activation=torch.nn.ELU())
        self.deconv3 = Conv3Layer(32, 16, 3,
                                  padding_mode="same",
                                  stride=1,
                                  activation=torch.nn.ELU())
        self.deconv4 = ResizeConv3d(16, 8, 4,
                                    padding_mode="valid",
                                    stride=2,
                                    activation=torch.nn.ELU())
        self.deconv5 = Conv3Layer(8, 1, 3,
                                  padding_mode="same",
                                  stride=1,
                                  activation=None)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), 1, 7, 7, 7)
        #x = F.interpolate(x,size=(15,15,15),mode="trilinear",align_corners=False)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return torch.sigmoid(x)


class VAE(nn.Module):

    def __init__(self, z_dim=100):
        super().__init__()
        self.encoder = Encoder(num_latents=z_dim)
        self.decoder = Decoder(num_latents=z_dim)

    def forward(self, x):
        #z = self.encoder(x)
        mean, logsigma = self.encoder(x)
        z = self.reparameterize(mean, logsigma)
        x = self.decoder(z)
        return x, mean, logsigma

    @staticmethod
    def reparameterize(mean, logsigma):
        std = torch.exp(logsigma)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std
