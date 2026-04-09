import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
    # spectral convolution in fourier domain for low-mode interaction
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # complex multiplication in fourier space for each mode
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # apply spectral conv to selected low-frequency modes
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # adapt to available modes to support multiple resolutions
        modes1_actual = min(self.modes1, x.size(-2) // 2)
        modes2_actual = min(self.modes2, x.size(-1) // 2 + 1) # considering nyquist limitations (modes cant be bigger than the grid resolution)

        out_ft[:, :, :modes1_actual, :modes2_actual] = self.compl_mul2d(
            x_ft[:, :, :modes1_actual, :modes2_actual],
            self.weights1[:, :, :modes1_actual, :modes2_actual]
        )
        out_ft[:, :, -modes1_actual:, :modes2_actual] = self.compl_mul2d(
            x_ft[:, :, -modes1_actual:, :modes2_actual],
            self.weights2[:, :, :modes1_actual, :modes2_actual]
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    # fourier neural operator with iterative spectral layers and local skip connections
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # input: 4 physical channels + 2 grid channels
        self.fc0 = nn.Linear(6, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, size_y])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y).repeat([batchsize, size_x, 1])
        return torch.stack((gridx, gridy), dim=3)

    def forward(self, x):
        # convert input to [batch, nx, nt, channels] and append grid channels
        grid = self.get_grid(x.shape, x.device)
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
