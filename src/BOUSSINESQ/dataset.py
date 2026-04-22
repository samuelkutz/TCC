import numpy as np
import torch
from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq


def generate_dataset(param_values, nx, nt, x_limit=30.0, t_limit=15.0, device='cpu'):
    # generate training dataset by solving the boussinesq equation directly at the
    # target dataset resolution, rather than downsampling from a previously
    # generated high-resolution solution.
    """
    Generate Boussinesq dataset for FNO training.
    
    Returns: x_train (N, 4, nx, nt), y_train (N, 2, nx, nt)
    """
    n_cases = len(param_values)
    
    input_data = np.zeros((n_cases, nx, nt, 4), dtype=np.float32)
    output_data = np.zeros((n_cases, nx, nt, 2), dtype=np.float32)

    for i, val in enumerate(param_values):
        # solve boussinesq equation directly at the target dataset resolution
        bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, val, val)
        solver = PseudoSpectralBoussinesq(bsq, Nx=nx, Nt=nt - 1, device=device)
        x_sol, t_sol, eta_sol, u_sol = solver.solve()

        eta_sub = eta_sol.T
        u_sub = u_sol.T

        # input channels correspond to eta0, u0, alpha, beta
        ch0 = np.tile(eta_sub[:, 0:1], (1, nt))
        ch1 = np.tile(u_sub[:, 0:1], (1, nt))
        ch2 = np.ones((nx, nt)) * val
        ch3 = np.ones((nx, nt)) * val

        input_data[i, ..., 0] = ch0
        input_data[i, ..., 1] = ch1
        input_data[i, ..., 2] = ch2
        input_data[i, ..., 3] = ch3

        # output channels correspond to eta and u solution fields
        output_data[i, ..., 0] = eta_sub
        output_data[i, ..., 1] = u_sub

        if (i+1) % 2 == 0:
            print(f"Processed case {i+1}/{n_cases} (alpha=beta={val:.2f})")

    # convert to pytorch tensors: (batch, channels, height, width)
    x_train = torch.from_numpy(input_data).permute(0, 3, 1, 2)
    y_train = torch.from_numpy(output_data).permute(0, 3, 1, 2)

    return x_train, y_train
