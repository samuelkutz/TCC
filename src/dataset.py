import numpy as np
import torch
from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq


def generate_dataset(param_values, Nx_high=256, Nt_high=256, nx_fno=64, nt_fno=64,
                     x_limit=30.0, t_limit=15.0, device='cpu'):
    """
    Generate Boussinesq dataset for FNO training.
    
    Returns: x_train (N, 4, nx_fno, nt_fno), y_train (N, 2, nx_fno, nt_fno)
    """
    n_cases = len(param_values)
    
    input_data = np.zeros((n_cases, nx_fno, nt_fno, 4), dtype=np.float32)
    output_data = np.zeros((n_cases, nx_fno, nt_fno, 2), dtype=np.float32)

    for i, val in enumerate(param_values):
        # Solve Boussinesq equation
        bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, val, val)
        solver = PseudoSpectralBoussinesq(bsq, Nx=Nx_high, Nt=Nt_high, device=device)
        x_sol, t_sol, eta_sol, u_sol = solver.solve()

        # Downsample to FNO resolution using rounded indices to avoid duplicates
        idx_x = np.round(np.linspace(0, Nx_high - 1, nx_fno, endpoint=True)).astype(int)
        idx_t = np.round(np.linspace(0, Nt_high, nt_fno, endpoint=True)).astype(int)

        eta_sub = eta_sol[idx_t, :][:, idx_x].T
        u_sub = u_sol[idx_t, :][:, idx_x].T

        # Input channels: [eta0, u0, a, b]
        ch0 = np.tile(eta_sub[:, 0:1], (1, nt_fno))
        ch1 = np.tile(u_sub[:, 0:1], (1, nt_fno))
        ch2 = np.ones((nx_fno, nt_fno)) * val
        ch3 = np.ones((nx_fno, nt_fno)) * val

        input_data[i, ..., 0] = ch0
        input_data[i, ..., 1] = ch1
        input_data[i, ..., 2] = ch2
        input_data[i, ..., 3] = ch3

        # Output channels: [eta, u]
        output_data[i, ..., 0] = eta_sub
        output_data[i, ..., 1] = u_sub

        if (i+1) % 2 == 0:
            print(f"Processed case {i+1}/{n_cases} (alpha=beta={val:.2f})")

    # Convert to PyTorch tensors: (Batch, Channels, Height, Width)
    x_train = torch.from_numpy(input_data).permute(0, 3, 1, 2)
    y_train = torch.from_numpy(output_data).permute(0, 3, 1, 2)

    return x_train, y_train


def save_dataset(x_train, y_train, filepath='dataset.pth'):
    """Save dataset to PyTorch file"""
    torch.save({
        'x_train': x_train,
        'y_train': y_train
    }, filepath)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath='dataset.pth'):
    """Load dataset from PyTorch file"""
    data = torch.load(filepath)
    return data['x_train'], data['y_train']
