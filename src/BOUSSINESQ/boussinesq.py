import numpy as np
import torch


class Boussinesq:
    def __init__(self, x_min, x_max, t_min, t_max, a, b, A=1):
        self.domain = {
            'x_min': torch.tensor(x_min),
            'x_max': torch.tensor(x_max),
            't_min': torch.tensor(t_min),
            't_max': torch.tensor(t_max)
        }
        self.a = a
        self.b = b
        self.A = A

    def ic(self, x):
        """Initial condition: eta(x,0) = A*sech^2(x), u(x,0) = 0"""
        mid = (self.domain['x_max'] + self.domain['x_min']) / 2
        val = (x - mid)
        eta_0 = self.A / (torch.cosh(val)**2)
        u_0 = torch.zeros_like(x)
        return eta_0, u_0

    def residual(self, model, x, t):
        """Compute the PDE residual for the Boussinesq equations on points (x,t)."""
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        eta_pred, u_pred = model(x, t)

        u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        eta_t = torch.autograd.grad(eta_pred, t, grad_outputs=torch.ones_like(eta_pred), retain_graph=True, create_graph=True)[0]

        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        eta_x = torch.autograd.grad(eta_pred, x, grad_outputs=torch.ones_like(eta_pred), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_xxt = torch.autograd.grad(u_xx, t, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]

        eta_u = eta_pred * u_pred
        eta_u_x = torch.autograd.grad(eta_u, x, grad_outputs=torch.ones_like(eta_u), retain_graph=True, create_graph=True)[0]
        nonlinear = u_pred * u_x

        res_eq_1 = eta_t + u_x + self.a * eta_u_x
        res_eq_2 = u_t - (self.b / 3.0) * u_xxt + eta_x + self.a * nonlinear

        return res_eq_1, res_eq_2


class PseudoSpectralBoussinesq:
    def __init__(self, boussinesq, Nx=256, Nt=1000, device='cpu'):
        self.Nx = Nx
        self.Nt = Nt
        self.device = device
        self.a = boussinesq.a
        self.b = boussinesq.b

        self.t_min = boussinesq.domain['t_min'].item()
        self.t_max = boussinesq.domain['t_max'].item()
        self.dt = (self.t_max - self.t_min) / Nt

        self.x_min = boussinesq.domain['x_min']
        self.x_max = boussinesq.domain['x_max']
        self.x = torch.linspace(self.x_min.item(), self.x_max.item(), Nx + 1, device=device)[:-1]
        self.dx = self.x[1] - self.x[0]

        # FFT frequencies
        self.k = 2 * torch.pi * torch.fft.fftfreq(Nx, d=(self.x_max.item() - self.x_min.item()) / Nx).to(device)
        self.ik = 1j * self.k
        self.k2 = self.k ** 2

        eta0, u0 = boussinesq.ic(self.x)
        self.eta_hat = torch.fft.fft(eta0)
        self.u_hat = torch.fft.fft(u0)

        self.time_steps = np.linspace(self.t_min, self.t_max, Nt + 1)
        self.eta_history = [eta0.cpu().numpy()]
        self.u_history = [u0.cpu().numpy()]

    def field(self, eta_hat, u_hat):
        eta = torch.fft.ifft(eta_hat).real
        u = torch.fft.ifft(u_hat).real

        eta_x = torch.fft.ifft(self.ik * eta_hat).real
        u_x = torch.fft.ifft(self.ik * u_hat).real

        nl_term1_hat = self.ik * torch.fft.fft(eta * u)
        nl_term2_hat = self.ik * torch.fft.fft(0.5 * u ** 2)

        # eta_t = -u_x - a*(eta*u)_x
        eta_t_hat = -self.ik * u_hat - self.a * nl_term1_hat

        # u_t - (b/3)*u_xxt = -eta_x - a*u*u_x
        rhs_u = -self.ik * eta_hat - self.a * nl_term2_hat
        denom_u = 1.0 + (self.b / 3.0) * self.k2
        u_t_hat = rhs_u / denom_u

        return eta_t_hat, u_t_hat

    def RK4_step(self, eta_hat, u_hat):
        dt = self.dt
        k1_eta, k1_u = self.field(eta_hat, u_hat)
        k2_eta, k2_u = self.field(eta_hat + 0.5 * dt * k1_eta, u_hat + 0.5 * dt * k1_u)
        k3_eta, k3_u = self.field(eta_hat + 0.5 * dt * k2_eta, u_hat + 0.5 * dt * k2_u)
        k4_eta, k4_u = self.field(eta_hat + dt * k3_eta, u_hat + dt * k3_u)

        eta_hat_new = eta_hat + (dt / 6.0) * (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta)
        u_hat_new = u_hat + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        return eta_hat_new, u_hat_new

    def solve(self):
        eta_h, u_h = self.eta_hat, self.u_hat

        res_eta = np.zeros((self.Nt + 1, self.Nx), dtype=np.float32)
        res_u = np.zeros((self.Nt + 1, self.Nx), dtype=np.float32)

        res_eta[0] = torch.fft.ifft(eta_h).real.cpu().numpy()
        res_u[0] = torch.fft.ifft(u_h).real.cpu().numpy()

        for n in range(1, self.Nt + 1):
            eta_h, u_h = self.RK4_step(eta_h, u_h)
            res_eta[n] = torch.fft.ifft(eta_h).real.cpu().numpy()
            res_u[n] = torch.fft.ifft(u_h).real.cpu().numpy()

        return self.x.cpu().numpy(), self.time_steps, res_eta, res_u
