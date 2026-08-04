"""The shared operator dataset: one pseudospectral solve per alpha = beta value.

Inputs are the four channels [eta(.,0), u(.,0), alpha, beta] tiled over the
space-time grid; outputs are the two solution fields [eta, u]. The FNO and the
PINO both train on this file, and the pointwise models are scored against solves
produced by the same solver, so every number in Chapter 4 traces to one ground
truth.
"""

import os

import numpy as np
import torch

from tools import compute_norm_stats, save_dataset
from experiments.common import operator_input_tensor, reference_solution, resolve_device


def generate_dataset(param_values, nx, nt, x_limit, t_limit, device, amplitude=1.0):
    # solve Boussinesq at the target resolution for each parameter value; returns
    # x_train (n, 4, nx, nt) and y_train (n, 2, nx, nt)
    n_cases = len(param_values)
    inputs = torch.empty((n_cases, 4, nx, nt), dtype=torch.float32)
    outputs = torch.empty((n_cases, 2, nx, nt), dtype=torch.float32)

    for i, value in enumerate(param_values):
        # Nt is the number of RK4 steps, so nt-1 steps give exactly nt time levels
        _, _, eta_sol, u_sol = reference_solution(
            value, x_limit, t_limit, nx=nx, nt=nt - 1, device=device, amplitude=amplitude)
        inputs[i] = operator_input_tensor(eta_sol, u_sol, value, nx=nx, nt=nt)[0]
        outputs[i, 0] = torch.from_numpy(np.ascontiguousarray(eta_sol.T))
        outputs[i, 1] = torch.from_numpy(np.ascontiguousarray(u_sol.T))
        print(f'  case {i + 1}/{n_cases} (alpha=beta={value:.2f})')

    return inputs, outputs


def run_dataset(dataset_file, param_values, x_limit, t_limit, dataset_res,
                device=None, amplitude=1.0):
    device = device or resolve_device()
    os.makedirs(os.path.dirname(dataset_file) or '.', exist_ok=True)
    print(f'  {len(param_values)} cases at resolution {dataset_res} on '
          f'[{-x_limit}, {x_limit}] x [0, {t_limit}], device {device}')
    print(f'  alpha=beta in [{min(param_values):.2f}, {max(param_values):.2f}]')

    x_train, y_train = generate_dataset(
        param_values, nx=dataset_res, nt=dataset_res,
        x_limit=x_limit, t_limit=t_limit, device=device, amplitude=amplitude)
    save_dataset(x_train, y_train, dataset_file,
                 norm_stats=compute_norm_stats(x_train, y_train, amplitude,
                                               min(param_values), max(param_values)))


if __name__ == '__main__':
    # standalone run, using the same settings.json the pipeline reads
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config

    run_dataset(**Config().dataset_kwargs)
