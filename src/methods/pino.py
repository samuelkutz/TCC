import torch
import torch.nn as nn
from methods.fno import FNO2d
from tools import from_symmetric


class PINO2d(FNO2d):
    """FNO backbone with a configurable output width (Li et al., 2023)."""
    def __init__(self, modes1, modes2, width, out_channels):
        super(PINO2d, self).__init__(modes1=modes1, modes2=modes2, width=width)
        self.out_channels = out_channels

        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, out_channels),
        )


def spectral_spatial_derivatives(u, dx):
    """u_x = ifft(i kx fft(u)) and u_xx = ifft(-kx^2 fft(u)) along the space axis (-2).

    Both are returned with the channel axis restored: (B, 1, Nx, Nt).
    """
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    Nx = u.shape[-2]
    u_ft = torch.fft.fft(u, dim=-2, norm='forward')
    kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx, device=u.device).view(1, Nx, 1)

    ux = torch.fft.ifft(1j * kx * u_ft, dim=-2, norm='forward').real
    uxx = torch.fft.ifft(-(kx ** 2) * u_ft, dim=-2, norm='forward').real

    return ux.unsqueeze(1), uxx.unsqueeze(1)


def finite_time_derivative(u, dt):
    """d/dt along the last axis: central in the interior, one-sided at the ends."""
    if u.ndim == 4 and u.shape[1] == 1:
        u = u[:, 0, :, :]

    out = torch.zeros_like(u)
    out[..., 0] = (u[..., 1] - u[..., 0]) / dt
    out[..., -1] = (u[..., -1] - u[..., -2]) / dt
    if u.shape[-1] > 2:
        out[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dt)
    return out.unsqueeze(1)


def _dealias_23_space(field):
    """Orszag 2/3 rule: zero the top third of spatial modes of a quadratic product,
    so aliasing from the nonlinear terms does not pollute the residual."""
    Nx = field.shape[-2]
    k_cut = Nx // 3
    field_ft = torch.fft.fft(field, dim=-2)
    mask = torch.ones(Nx, device=field.device, dtype=field_ft.dtype)
    if Nx - k_cut > k_cut + 1:
        mask[k_cut + 1:Nx - k_cut] = 0.0
    shape = [1] * field.ndim
    shape[-2] = Nx
    field_ft = field_ft * mask.view(shape)
    return torch.fft.ifft(field_ft, dim=-2).real


def _fourier_interp_space(field, n_target):
    """Spectral interpolation along the space axis (-2) to n_target points.

    Exact for band-limited inputs, so the smooth initial profiles resample without
    interpolation error."""
    n_src = field.shape[-2]
    if n_target == n_src:
        return field
    field_ft = torch.fft.rfft(field, dim=-2)
    n_keep = min(field_ft.shape[-2], n_target // 2 + 1)
    shape = list(field.shape)
    shape[-2] = n_target // 2 + 1
    padded = torch.zeros(shape, dtype=field_ft.dtype, device=field.device)
    padded[..., :n_keep, :] = field_ft[..., :n_keep, :]
    return torch.fft.irfft(padded, n=n_target, dim=-2) * (n_target / n_src)


def build_finer_input(batch_x, phys_nx, phys_nt):
    """Rebuild the operator input on a finer (phys_nx, phys_nt) grid.

    Only the two space profiles are resampled spectrally; alpha and beta are
    constant. Nothing is upsampled from a coarse solution."""
    B = batch_x.shape[0]
    eta0 = _fourier_interp_space(batch_x[:, 0:1, :, 0:1], phys_nx)  # (B,1,phys_nx,1)
    u0 = _fourier_interp_space(batch_x[:, 1:2, :, 0:1], phys_nx)
    alpha = batch_x[:, 2:3, :1, :1]                                 # (B,1,1,1) constant
    beta = batch_x[:, 3:4, :1, :1]
    eta0 = eta0.expand(B, 1, phys_nx, phys_nt)
    u0 = u0.expand(B, 1, phys_nx, phys_nt)
    alpha = alpha.expand(B, 1, phys_nx, phys_nt)
    beta = beta.expand(B, 1, phys_nx, phys_nt)
    return torch.cat([eta0, u0, alpha, beta], dim=1).contiguous()


def pde_residual_boussinesq(eta, u, dx, dt, alpha, beta, dealias=False):
    """Continuity and momentum residuals:

        res_1 = eta_t + u_x + alpha (eta u)_x
        res_2 = u_t - (beta/3) u_xxt + eta_x + alpha (u^2)_x / 2

    The momentum nonlinearity is in conservative form, matching the flux used by
    the pseudospectral solver.
    """
    eta_x, _ = spectral_spatial_derivatives(eta, dx)
    u_x, u_xx = spectral_spatial_derivatives(u, dx)

    eta_t = finite_time_derivative(eta, dt)
    u_t = finite_time_derivative(u, dt)
    u_xxt = finite_time_derivative(u_xx, dt)

    eta_u = eta * u
    u2 = u * u
    if dealias:
        eta_u = _dealias_23_space(eta_u)
        u2 = _dealias_23_space(u2)
    eta_u_x, _ = spectral_spatial_derivatives(eta_u, dx)
    u2_x, _ = spectral_spatial_derivatives(u2, dx)

    res_eq_1 = eta_t + u_x + alpha * eta_u_x
    res_eq_2 = u_t - (beta / 3.0) * u_xxt + eta_x + alpha * 0.5 * u2_x

    return res_eq_1, res_eq_2


def _unnorm_channel(t, norm_stats, key_min, key_max, ch):
    return from_symmetric(
        t,
        norm_stats[key_min][:, ch:ch + 1, :, :],
        norm_stats[key_max][:, ch:ch + 1, :, :],
    )


def pino_loss(model, batch_x, batch_y, dx, dt, norm_stats,
              phys_weight=1.0, ic_weight=1.0, data_weight=0.01,
              x_limit=None, t_limit=None, phys_nx=None, phys_nt=None, dealias=False):
    """PINO objective; returns (loss, loss_pde, loss_ic, loss_data).

    The supervised term uses the training grid. The PDE and initial-condition terms
    may be evaluated zero-shot on a finer (phys_nx, phys_nt) grid, where the
    discrete residual of the true solution is smaller because of less aliasing.
    Pass phys_nx/phys_nt as None to keep the physics on the training grid.
    """
    nx, nt = batch_x.shape[2], batch_x.shape[3]
    use_finer = (phys_nx is not None and phys_nt is not None
                 and (phys_nx != nx or phys_nt != nt))
    need_data = (data_weight != 0.0)

    # ---- supervised term on the training grid ----
    pred_data = model(batch_x) if (need_data or not use_finer) else None
    if pred_data is not None and pred_data.shape[1] != 2:
        raise ValueError('PINO models must output two channels: [eta, u].')
    if need_data:
        # in normalized units, the same space the FNO's supervised loss uses, so
        # the pure-data and data-and-physics models are fit against the same
        # objective scale and the comparison between them is not carrying a
        # hidden factor of (f_max - f_min)^2
        diff = pred_data - batch_y
        loss_data = torch.mean(diff * diff)
    else:
        loss_data = torch.zeros((), device=batch_x.device)

    # ---- physics grid: finer zero-shot query, or the training grid ----
    if use_finer:
        if x_limit is None or t_limit is None:
            raise ValueError('finer-grid physics requires x_limit and t_limit')
        src_x = build_finer_input(batch_x, phys_nx, phys_nt)
        phys_pred = model(src_x)
        phys_dx = 2.0 * x_limit / phys_nx
        phys_dt = t_limit / (phys_nt - 1)
    else:
        src_x = batch_x
        phys_pred = pred_data if pred_data is not None else model(batch_x)
        phys_dx, phys_dt = dx, dt

    eta_p = _unnorm_channel(phys_pred[:, 0:1], norm_stats, 'output_min', 'output_max', 0)
    u_p = _unnorm_channel(phys_pred[:, 1:2], norm_stats, 'output_min', 'output_max', 1)
    alpha = _unnorm_channel(src_x[:, 2:3], norm_stats, 'input_min', 'input_max', 2)
    beta = _unnorm_channel(src_x[:, 3:4], norm_stats, 'input_min', 'input_max', 3)

    # residual in physical units, so the constraint is the true balance and not a
    # rescaled surrogate
    res_eq_1, res_eq_2 = pde_residual_boussinesq(eta_p, u_p, phys_dx, phys_dt, alpha, beta, dealias=dealias)
    loss_pde = torch.mean(res_eq_1 ** 2 + res_eq_2 ** 2)

    eta0_phys = _unnorm_channel(src_x[:, 0:1, :, 0:1], norm_stats, 'input_min', 'input_max', 0)
    u0_phys = _unnorm_channel(src_x[:, 1:2, :, 0:1], norm_stats, 'input_min', 'input_max', 1)
    loss_ic = torch.mean((u_p[:, :, :, 0:1] - u0_phys) ** 2 + (eta_p[:, :, :, 0:1] - eta0_phys) ** 2)

    loss = phys_weight * loss_pde + ic_weight * loss_ic + data_weight * loss_data
    return loss, loss_pde, loss_ic, loss_data