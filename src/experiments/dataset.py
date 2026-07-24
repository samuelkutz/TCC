import os
import numpy as np
import torch

from methods.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from tools import compute_norm_stats, save_dataset

# Standalone defaults for running this module directly. The authoritative pipeline
# (src/main.py) overrides every one of these from settings.json, so these values do
# NOT define the thesis dataset; they are aligned to settings.json only to avoid a
# standalone run silently producing a different dataset than the one described.
DEFAULT_DATASET_FILE = os.path.join('results', 'models', 'boussinesq_dataset.pth')
DEFAULT_DEVICE = 'cpu'
DEFAULT_PARAM_VALUES = list(np.linspace(0.1, 4.0, 20))
DEFAULT_X_LIMIT = 60.0
DEFAULT_T_LIMIT = 30.0
DEFAULT_DATASET_RES = 128


def generate_dataset(param_values, nx, nt, x_limit, t_limit, device):
    # solve boussinesq at target resolution for each param value; returns x_train (n,4,nx,nt), y_train (n,2,nx,nt)
    n_cases = len(param_values)

    input_data = np.zeros((n_cases, nx, nt, 4), dtype=np.float32)
    output_data = np.zeros((n_cases, nx, nt, 2), dtype=np.float32)

    for i, val in enumerate(param_values):
        # solve boussinesq equation directly at the target dataset resolution
        bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, val, val, 1)
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

        print(f"processed case {i+1}/{n_cases} (alpha=beta={val:.2f})")

    # convert to pytorch tensors: (batch, channels, height, width)
    x_train = torch.from_numpy(input_data).permute(0, 3, 1, 2)
    y_train = torch.from_numpy(output_data).permute(0, 3, 1, 2)

    return x_train, y_train


def run_dataset(dataset_file,
                device,
                param_values,
                x_limit,
                t_limit,
                dataset_res):
    if any(v is None for v in [dataset_file, device, param_values, x_limit, t_limit, dataset_res]):
        raise ValueError(
            'run_dataset requires explicit values for dataset_file, device, param_values, '
            'x_limit, t_limit, and dataset_res.'
        )
    os.makedirs(os.path.dirname(dataset_file) or '.', exist_ok=True)
    print('generating dataset with the following settings:')
    print(f'  dataset_file: {dataset_file}')
    print(f'  device: {device}')
    print(f'  param_values: {param_values}')
    print(f'  x_limit: {x_limit}, t_limit: {t_limit}')
    print(f'  dataset_res: {dataset_res}')

    x_train, y_train = generate_dataset(
        param_values,
        nx=dataset_res,
        nt=dataset_res,
        x_limit=x_limit,
        t_limit=t_limit,
        device=device,
    )
    norm_stats = compute_norm_stats(x_train, y_train)
    save_dataset(x_train, y_train, dataset_file, norm_stats=norm_stats)
    print(f'dataset written to {dataset_file}')


if __name__ == '__main__':
    run_dataset(
        dataset_file=DEFAULT_DATASET_FILE,
        device=DEFAULT_DEVICE,
        param_values=DEFAULT_PARAM_VALUES,
        x_limit=DEFAULT_X_LIMIT,
        t_limit=DEFAULT_T_LIMIT,
        dataset_res=DEFAULT_DATASET_RES,
    )
