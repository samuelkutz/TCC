import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv2d(nn.Module):
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
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # adapt to available modes to support multiple resolutions
        modes1_actual = min(self.modes1, x.size(-2) // 2)
        modes2_actual = min(self.modes2, x.size(-1) // 2 + 1) # considering Nyquist limitaions (modes cant be bigger than the grid resolution)

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


class PINO2d(nn.Module):
    def __init__(self, modes1, modes2, width, out_channels=1):
        super(PINO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.out_channels = out_channels

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
        self.fc2 = nn.Linear(128, self.out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, size_y])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y).repeat([batchsize, size_x, 1])
        return torch.stack((gridx, gridy), dim=3)

    def forward(self, x):
        """Forward pass for PINO2d.

        Input x has shape [batch, channels, nx, nt]. Output has shape
        [batch, out_channels, nx, nt], where channel 0 is eta and channel 1 is u.
        """
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


def spectral_spatial_derivatives(u, dx):
    """Compute spectral spatial derivatives for a 2D field u(x,t) along x."""
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    Nx = u.shape[-2]
    u_ft = torch.fft.fft(u, dim=-2)
    kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=u.device).view(1, Nx, 1)

    ux = torch.fft.ifft(1j * kx * u_ft, dim=-2).real
    uxx = torch.fft.ifft(-(kx ** 2) * u_ft, dim=-2).real

    return ux.unsqueeze(1), uxx.unsqueeze(1)


def finite_time_derivative(u, dt, order=1):
    """Compute finite-difference time derivatives for a 2D field u(x,t) along t."""
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    ut = torch.zeros_like(u)
    utt = torch.zeros_like(u)

    # First-order derivative
    ut[..., 0] = (u[..., 1] - u[..., 0]) / dt
    ut[..., -1] = (u[..., -1] - u[..., -2]) / dt
    if u.shape[-1] > 2:
        ut[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dt)

    # Second-order derivative
    utt[..., 0] = (u[..., 2] - 2.0 * u[..., 1] + u[..., 0]) / (dt ** 2)
    utt[..., -1] = (u[..., -1] - 2.0 * u[..., -2] + u[..., -3]) / (dt ** 2)
    if u.shape[-1] > 2:
        utt[..., 1:-1] = (u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]) / (dt ** 2)

    if order == 1:
        return ut.unsqueeze(1)
    if order == 2:
        return utt.unsqueeze(1)
    raise ValueError('order must be 1 or 2')


def pde_residual_boussinesq(eta, u, dx, dt, alpha, beta):
    """Return the Boussinesq PDE residuals for (eta,u)(x,t)."""
    eta_x, eta_xx = spectral_spatial_derivatives(eta, dx)
    u_x, u_xx = spectral_spatial_derivatives(u, dx)

    eta_t = finite_time_derivative(eta, dt, order=1)
    u_t = finite_time_derivative(u, dt, order=1)
    u_xxt = finite_time_derivative(u_xx, dt, order=1)
    eta_u = eta * u
    eta_u_x, _ = spectral_spatial_derivatives(eta_u, dx)
    nonlinear = u * u_x

    alpha = alpha.reshape(-1, 1, 1, 1)
    beta = beta.reshape(-1, 1, 1, 1)
    res_eq_1 = eta_t + u_x + alpha * eta_u_x
    res_eq_2 = u_t - (beta / 3.0) * u_xxt + eta_x + alpha * nonlinear

    return res_eq_1, res_eq_2


def pino_loss(model, batch_x, batch_y, dx, dt,
              phys_weight=1.0, ic_weight=0.1, data_weight=0.01):
    """Compute a PINO loss combining PDE residual, IC, and relative data terms."""
    pred = model(batch_x)
    if pred.shape[1] == 2:
        eta_pred = pred[:, 0:1, :, :]
        u_pred = pred[:, 1:2, :, :]
    else:
        raise ValueError('PINO models must output two channels: [eta, u].')

    alpha = batch_x[:, 2:3, :, :].mean(dim=[2, 3], keepdim=True)
    beta = batch_x[:, 3:4, :, :].mean(dim=[2, 3], keepdim=True)
    res_eq_1, res_eq_2 = pde_residual_boussinesq(eta_pred, u_pred, dx, dt, alpha, beta)
    loss_pde = torch.mean(res_eq_1 ** 2 + res_eq_2 ** 2)

    eta0 = batch_x[:, 0:1, :, 0:1]
    u0 = batch_x[:, 1:2, :, 0:1]
    eta_pred0 = eta_pred[:, :, :, 0:1]
    u_pred0 = u_pred[:, :, :, 0:1]
    loss_ic = torch.mean((u_pred0 - u0) ** 2 + (eta_pred0 - eta0) ** 2)

    diff = pred - batch_y
    sq_sum = torch.sqrt(torch.sum(diff * diff, dim=[1, 2, 3]) + 1e-12)
    target_norm = torch.sqrt(torch.sum(batch_y * batch_y, dim=[1, 2, 3]) + 1e-12)
    loss_data = torch.mean(sq_sum / (target_norm + 1e-12))

    loss = phys_weight * loss_pde + ic_weight * loss_ic + data_weight * loss_data
    return loss, loss_pde, loss_ic, loss_data


class L2_loss(object):
    def __init__(self, eps=1e-10):
        super(L2_loss, self).__init__()
        self.eps = eps

    def __call__(self, x, y):
        # Flattened L2 norm: sqrt(sum((x - y)^2)) per sample
        diff = x - y
        loss = torch.sqrt(torch.sum(diff * diff, dim=[1, 2, 3]) + self.eps)
        return torch.mean(loss)


class RelativeL2_loss(object):
    def __init__(self, eps=1e-10):
        super(RelativeL2_loss, self).__init__()
        self.eps = eps

    def __call__(self, x, y):
        diff = x - y
        numerator = torch.sqrt(torch.sum(diff * diff, dim=[1, 2, 3]) + self.eps)
        denominator = torch.sqrt(torch.sum(y * y, dim=[1, 2, 3]) + self.eps)
        return torch.mean(numerator / (denominator + self.eps))
