"""Grid evaluation for the data-only MLP.

The MLP is a plain regressor on (x, t), so its coordinates are rescaled to
[-1, 1]^2 before the forward pass. Training and evaluation both go through here.
"""

import numpy as np
import torch

from tools import to_symmetric


def to_unit_square(x_np, t_np, x_limit, t_limit):
    """The shared map of `tools.to_symmetric`, with the domain as the envelope:
    x in [-x_limit, x_limit] -> [-1, 1];  t in [0, t_limit] -> [-1, 1]."""
    x_u = to_symmetric(np.asarray(x_np, dtype=np.float32), -x_limit, x_limit)
    t_u = to_symmetric(np.asarray(t_np, dtype=np.float32), 0.0, t_limit)
    return x_u, t_u


def mlp_grid_inputs(x_np, t_np, x_limit, t_limit, device):
    """Flattened (x, t) pairs over the tensor-product grid, in meshgrid 'xy' order."""
    x_u, t_u = to_unit_square(x_np, t_np, x_limit, t_limit)
    X, T = np.meshgrid(x_u, t_u, indexing='xy')        # both (Nt, Nx)
    stacked = np.stack([X.reshape(-1), T.reshape(-1)], axis=-1)
    return torch.from_numpy(stacked).float().to(device)


def mlp_eta_grid(model, x_np, t_np, x_limit, t_limit):
    """Evaluate the MLP on x_np x t_np; returns (Nx, Nt)."""
    device = next(model.parameters()).device
    inputs = mlp_grid_inputs(x_np, t_np, x_limit, t_limit, device)
    model.eval()
    with torch.no_grad():
        eta_flat = model(inputs)
    return eta_flat.cpu().numpy().reshape(len(t_np), len(x_np)).T
