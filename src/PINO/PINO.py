import torch
import torch.nn as nn
from FNO.FNO import FNO2d
from tools import unnormalize_tensor


class PINO2d(FNO2d):
    """physics-informed neural operator with an fno backbone.

    references:
    - li et al., 2021, fourier neural operator for parametric pdes.
    - li et al., 2023, physics-informed neural operator for learning pdes.
    - neuraloperator repository: https://github.com/neuraloperator/neuraloperator
    """
    def __init__(self, modes1, modes2, width, out_channels):
        super(PINO2d, self).__init__(modes1=modes1, modes2=modes2, width=width)
        self.out_channels = out_channels

        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, out_channels),
        )


def spectral_spatial_derivatives(u, dx):
    """compute spatial derivatives via fft using forward normalization.

    dx is spatial spacing, kx = 2*pi*fftfreq
    ux = ifft(i*kx*fft(u)), uxx = ifft(-kx^2*fft(u))
    """
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    Nx = u.shape[-2]
    u_ft = torch.fft.fft(u, dim=-2, norm='forward')
    # spectral derivative factor for spatial dimension
    kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=u.device).view(1, Nx, 1)

    ux = torch.fft.ifft(1j * kx * u_ft, dim=-2, norm='forward').real
    uxx = torch.fft.ifft(-(kx ** 2) * u_ft, dim=-2, norm='forward').real

    return ux.unsqueeze(1), uxx.unsqueeze(1)


def finite_time_derivative(u, dt, order):
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    ut = torch.zeros_like(u)
    utt = torch.zeros_like(u)

    # first-order time derivative using forward/backward difference at boundaries
    ut[..., 0] = (u[..., 1] - u[..., 0]) / dt
    ut[..., -1] = (u[..., -1] - u[..., -2]) / dt
    if u.shape[-1] > 2:
        ut[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dt)

    # second-order time derivative with one-sided boundary approximations
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
    """compute boussinesq continuity and momentum residuals.

    res_eq_1 = eta_t + u_x + alpha*partial_x(eta*u)
    res_eq_2 = u_t - (beta/3)*u_xxt + eta_x + alpha*u*u_x
    """
    eta_x, eta_xx = spectral_spatial_derivatives(eta, dx)
    u_x, u_xx = spectral_spatial_derivatives(u, dx)

    eta_t = finite_time_derivative(eta, dt, order=1)
    u_t = finite_time_derivative(u, dt, order=1)
    u_xxt = finite_time_derivative(u_xx, dt, order=1)
    eta_u = eta * u
    eta_u_x, _ = spectral_spatial_derivatives(eta_u, dx)
    nonlinear = u * u_x

    # continuity residual: eta_t + u_x + alpha * (eta*u)_x
    res_eq_1 = eta_t + u_x + alpha * eta_u_x
    # momentum residual: u_t - (beta/3) u_xxt + eta_x + alpha u u_x
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

    # pde residual is computed in physical units for physics-consistent constraints.
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

    # weighted objective follows the standard pino decomposition (physics + ic + data).
    loss = phys_weight * loss_pde + ic_weight * loss_ic + data_weight * loss_data
    return loss, loss_pde, loss_ic, loss_data