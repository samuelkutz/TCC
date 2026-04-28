import torch
import torch.nn as nn
from FNO.FNO import SpectralConv2d
from tools import normalize_tensor, unnormalize_tensor


class PINO2d(nn.Module):
    def __init__(self, modes1, modes2, width, out_channels):
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
        gridx = torch.linspace(-1, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, size_y])
        gridy = torch.linspace(-1, 1, size_y, dtype=torch.float, device=device)
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
        x = torch.nn.functional.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = torch.nn.functional.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = torch.nn.functional.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = torch.nn.functional.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


def spectral_spatial_derivatives(u, dx):
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    Nx = u.shape[-2]
    u_ft = torch.fft.fft(u, dim=-2)
    kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=u.device).view(1, Nx, 1)

    ux = torch.fft.ifft(1j * kx * u_ft, dim=-2).real
    uxx = torch.fft.ifft(-(kx ** 2) * u_ft, dim=-2).real

    return ux.unsqueeze(1), uxx.unsqueeze(1)


def finite_time_derivative(u, dt, order):
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    ut = torch.zeros_like(u)
    utt = torch.zeros_like(u)

    ut[..., 0] = (u[..., 1] - u[..., 0]) / dt
    ut[..., -1] = (u[..., -1] - u[..., -2]) / dt
    if u.shape[-1] > 2:
        ut[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dt)

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
    eta_x, eta_xx = spectral_spatial_derivatives(eta, dx)
    u_x, u_xx = spectral_spatial_derivatives(u, dx)

    eta_t = finite_time_derivative(eta, dt, order=1)
    u_t = finite_time_derivative(u, dt, order=1)
    u_xxt = finite_time_derivative(u_xx, dt, order=1)
    eta_u = eta * u
    eta_u_x, _ = spectral_spatial_derivatives(eta_u, dx)
    nonlinear = u * u_x

    # Keep alpha and beta in their original batched field shape so they broadcast correctly
    # against the spatial derivatives. They are constant per sample in the current dataset,
    # but this also supports spatially varying coefficients if used later.
    res_eq_1 = eta_t + u_x + alpha * eta_u_x
    res_eq_2 = u_t - (beta / 3.0) * u_xxt + eta_x + alpha * nonlinear

    return res_eq_1, res_eq_2


def pino_loss(model, batch_x, batch_y, dx, dt, norm_stats,
              phys_weight=1.0, ic_weight=0.1, data_weight=0.01):
    pred = model(batch_x)
    if pred.shape[1] == 2:
        eta_pred_norm = pred[:, 0:1, :, :]
        u_pred_norm = pred[:, 1:2, :, :]
    else:
        raise ValueError('PINO models must output two channels: [eta, u].')

    alpha_norm = batch_x[:, 2:3, :, :]
    beta_norm = batch_x[:, 3:4, :, :]
    alpha = unnormalize_tensor(
        alpha_norm,
        norm_stats['input_min'][:, 2:3, :, :],
        norm_stats['input_max'][:, 2:3, :, :],
        norm_stats['eps'],
    )
    beta = unnormalize_tensor(
        beta_norm,
        norm_stats['input_min'][:, 3:4, :, :],
        norm_stats['input_max'][:, 3:4, :, :],
        norm_stats['eps'],
    )

    eta_pred = unnormalize_tensor(
        eta_pred_norm,
        norm_stats['output_min'][:, 0:1, :, :],
        norm_stats['output_max'][:, 0:1, :, :],
        norm_stats['eps'],
    )
    u_pred = unnormalize_tensor(
        u_pred_norm,
        norm_stats['output_min'][:, 1:2, :, :],
        norm_stats['output_max'][:, 1:2, :, :],
        norm_stats['eps'],
    )

    # pde residual is evaluated on physical units, but the model is trained with normalized fields.
    res_eq_1, res_eq_2 = pde_residual_boussinesq(eta_pred, u_pred, dx, dt, alpha, beta)
    loss_pde = torch.mean(res_eq_1 ** 2 + res_eq_2 ** 2)

    eta0 = batch_x[:, 0:1, :, 0:1]
    u0 = batch_x[:, 1:2, :, 0:1]
    eta0_phys = unnormalize_tensor(
        eta0,
        norm_stats['input_min'][:, 0:1, :, :],
        norm_stats['input_max'][:, 0:1, :, :],
        norm_stats['eps'],
    )
    u0_phys = unnormalize_tensor(
        u0,
        norm_stats['input_min'][:, 1:2, :, :],
        norm_stats['input_max'][:, 1:2, :, :],
        norm_stats['eps'],
    )
    eta_pred0_phys = eta_pred[:, :, :, 0:1]
    u_pred0_phys = u_pred[:, :, :, 0:1]
    loss_ic = torch.mean((u_pred0_phys - u0_phys) ** 2 + (eta_pred0_phys - eta0_phys) ** 2)

    batch_y_phys = unnormalize_tensor(
        batch_y,
        norm_stats['output_min'],
        norm_stats['output_max'],
        norm_stats['eps'],
    )
    diff = torch.cat((eta_pred, u_pred), dim=1) - batch_y_phys
    loss_data = torch.mean(diff * diff)

    loss = phys_weight * loss_pde + ic_weight * loss_ic + data_weight * loss_data
    return loss, loss_pde, loss_ic, loss_data
