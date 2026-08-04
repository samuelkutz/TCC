import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

from methods.mlp import MLP
from tools import is_log_epoch, log_epoch


class PINN(nn.Module):
    """Physics-informed network for the Boussinesq system (Raissi et al., 2019).

    Maps (x, t) -> (eta, u) with an MLP backbone and trains on the PDE residual,
    the initial condition and an optional supervised data term.
    """

    def __init__(self, input_size, output_size, neurons, hidden_layers,
                 Boussinesq, domain_points=3000, ic_points=500,
                 optimizer_name='Adam', lr=1e-3, data=None, data_weight=1.0,
                 device='cpu'):
        super(PINN, self).__init__()
        self.device = torch.device(device)
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = neurons
        self.hidden_layers = hidden_layers
        self.Boussinesq = Boussinesq
        self.domain_points = domain_points
        self.ic_points = ic_points
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.data = data
        self.data_weight = data_weight

        # same Linear/Tanh sequence as the data-only MLP, so the two models differ
        # only in training signal
        self.net = MLP(input_size, output_size, neurons, hidden_layers,
                       activation='tanh', device=self.device)
        self.to(self.device)

        # pre-converted once to avoid a host-to-device copy every epoch
        self._x_flat = self._t_flat = self._eta_flat = self._u_flat = None
        if data is not None and data_weight > 0.0:
            x_data = torch.from_numpy(data['x']).float().to(self.device)
            t_data = torch.from_numpy(data['t']).float().to(self.device)
            eta_data = torch.from_numpy(data['eta']).float().to(self.device)
            u_data = torch.from_numpy(data['u']).float().to(self.device)
            T = t_data.unsqueeze(1).repeat(1, x_data.shape[0])
            X = x_data.unsqueeze(0).repeat(t_data.shape[0], 1)
            self._x_flat = X.reshape(-1, 1)
            self._t_flat = T.reshape(-1, 1)
            self._eta_flat = eta_data.reshape(-1, 1)
            self._u_flat = u_data.reshape(-1, 1)

    def _build_optimizer(self):
        if self.optimizer_name.lower() == 'sgd':
            return optim.SGD(self.parameters(), lr=self.lr)
        return optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, t):
        """(x, t) -> (eta, u), each (N, 1)."""
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        xt = torch.cat([x, t], dim=-1).to(self.device)
        output = self.net(xt)
        return output[..., 0:1], output[..., 1:2]

    def predict_eta_grid(self, x_np, t_np):
        """Evaluate eta on the grid x_np x t_np; returns (Nx, Nt)."""
        x_np = np.asarray(x_np, dtype=np.float32)
        t_np = np.asarray(t_np, dtype=np.float32)
        Nx = x_np.shape[0]
        Nt = t_np.shape[0]
        X, T = np.meshgrid(x_np, t_np, indexing='xy')
        x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(self.device)
        t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(self.device)
        self.eval()
        with torch.no_grad():
            eta_flat, _ = self(x_tensor, t_tensor)
        return eta_flat.cpu().numpy().reshape(Nt, Nx).T

    def _sample_domain(self, n_points):
        x_min = self.Boussinesq.domain['x_min']
        x_max = self.Boussinesq.domain['x_max']
        t_min = self.Boussinesq.domain['t_min']
        t_max = self.Boussinesq.domain['t_max']

        x = torch.rand(n_points, 1, device=self.device) * (x_max - x_min) + x_min
        t = torch.rand(n_points, 1, device=self.device) * (t_max - t_min) + t_min
        return x, t

    def _initial_condition_loss(self):
        x0 = torch.linspace(self.Boussinesq.domain['x_min'],
                            self.Boussinesq.domain['x_max'],
                            self.ic_points, device=self.device).unsqueeze(-1)
        t0 = torch.zeros_like(x0)
        eta_pred, u_pred = self(x0, t0)
        eta_true, u_true = self.Boussinesq.ic(x0)
        return torch.mean((eta_pred - eta_true) ** 2 + (u_pred - u_true) ** 2)

    def _data_loss(self):
        # random subsample of the flattened reference grid
        if self._x_flat is None:
            return torch.tensor(0.0, device=self.device)

        sample_size = min(self._x_flat.shape[0], self.domain_points)
        idx = torch.randperm(self._x_flat.shape[0], device=self.device)[:sample_size]
        eta_pred, u_pred = self(self._x_flat[idx], self._t_flat[idx])
        return torch.mean((eta_pred - self._eta_flat[idx]) ** 2 + (u_pred - self._u_flat[idx]) ** 2)

    def _pde_loss(self):
        x_f, t_f = self._sample_domain(self.domain_points)
        res_eq_1, res_eq_2 = self.Boussinesq.residual(self, x_f, t_f)
        return torch.mean(res_eq_1 ** 2 + res_eq_2 ** 2)

    def run_train_loop(self, epochs, print_interval, snapshot_callback=None):
        # PDE residual + initial condition + optional supervised data term
        self.optimizer = self._build_optimizer()
        history = []
        start_time = default_timer()

        for ep in range(epochs):
            self.train()
            self.optimizer.zero_grad()

            pde_loss = self._pde_loss()
            ic_loss = self._initial_condition_loss()
            data_loss = self._data_loss()
            loss = pde_loss + ic_loss + self.data_weight * data_loss

            loss.backward()
            self.optimizer.step()
            history.append(loss.item())

            if is_log_epoch(ep, epochs, print_interval):
                log_epoch(ep + 1, epochs, default_timer() - start_time,
                          total_loss=loss.item(), pde_loss=pde_loss.item(),
                          ic_loss=ic_loss.item(), data_loss=data_loss.item())
                if snapshot_callback is not None:
                    snapshot_callback(self, ep + 1)

        return history