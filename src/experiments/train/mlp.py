"""Model 1 of the ladder: a data-only MLP fitting a single Boussinesq solution.

(x, t) -> eta by supervised regression against the pseudospectral reference at
alpha = beta = param_value. No PDE residual, no initial-condition term: the
equation never enters, which is what makes this the plainest testbed for
spectral bias among the six trained models.
"""

import os

import torch
import torch.optim as optim

from methods.mlp import MLP
from tools import is_log_epoch, log_epoch, save_model, set_seed
from experiments.common import (
    BandTracker, StageTimer, reference_solution, resolve_device, stage_paths,
    write_stage_metadata,
)
from experiments.predict import mlp_eta_grid, mlp_grid_inputs

LABEL = 'mlp'


def train_mlp(x_limit, t_limit, train_resolution, param_value, epochs, neurons,
              hidden_layers, activation, optimizer_name, lr, batch_size, print_interval,
              results_dir, amplitude=1.0, seed=None):
    # `seed` reseeds this stage in isolation. The pipeline seeds once globally in
    # main.py and leaves this None, so the stage inherits whatever RNG state the
    # earlier stages left; pass a seed to make a standalone run reproducible.
    if seed is not None:
        set_seed(seed)

    device = resolve_device()
    weights_dir, metadata_dir = stage_paths(results_dir, LABEL)

    # NOTE: PseudoSpectralBoussinesq takes Nt as the number of RK4 *steps* and returns
    # Nt+1 time levels. Passing Nt=train_resolution here yields train_resolution+1
    # levels, one more than the FNO/PINO dataset (which passes Nt=res-1). This is the
    # single-solve reference the pointwise models fit; the committed weights were
    # trained on it. To align the pointwise grid with the dataset's 127-step /
    # 128-level convention (Table 3.1), pass Nt=train_resolution-1 and retrain.
    x_sol, t_sol, eta_sol, _ = reference_solution(
        param_value, x_limit, t_limit, nx=train_resolution, nt=train_resolution,
        device=device, amplitude=amplitude)

    # supervised (x, t) -> eta pairs over the full grid, inputs mapped to [-1, 1]^2.
    # eta_sol is (Nt, Nx); flatten to match the (Nt, Nx) meshgrid ordering.
    inputs = mlp_grid_inputs(x_sol, t_sol, x_limit, t_limit, device)
    targets = torch.from_numpy(eta_sol.reshape(-1, 1)).float().to(device)
    bands = BandTracker(eta_sol.T)

    model = MLP(input_size=2, output_size=1, neurons=neurons, hidden_layers=hidden_layers,
                activation=activation, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  {LABEL}: {num_params:,} parameters on {device}')

    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    use_batches = bool(batch_size) and batch_size > 0
    if use_batches:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets),
            batch_size=batch_size, shuffle=True)

    history = []
    timer = StageTimer()
    for ep in range(epochs):
        model.train()
        if use_batches:
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = torch.mean((model(batch_x) - batch_y) ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
        else:
            optimizer.zero_grad()
            loss = torch.mean((model(inputs) - targets) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
        history.append(epoch_loss)

        if is_log_epoch(ep, epochs, print_interval):
            log_epoch(ep + 1, epochs, timer.elapsed, total_loss=epoch_loss)
            bands.record(ep + 1, mlp_eta_grid(model, x_sol, t_sol, x_limit, t_limit))
            model.train()

    training_duration = timer.elapsed
    model_file = save_model(model, os.path.join(weights_dir, f'{LABEL}_weights.pth'))

    write_stage_metadata({
        'mode': 'pure_data',
        'seed': seed,
        'train_history': history,
        'spectral_history': bands.as_dict(),
        'training_duration': training_duration,
        'final_loss': history[-1] if history else None,
        'num_params': num_params,
        'params': {
            'epochs': epochs,
            'neurons': neurons,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'optimizer_name': optimizer_name,
            'lr': lr,
            'batch_size': batch_size,
            'train_resolution': train_resolution,
            'param_value': param_value,
            'amplitude': amplitude,
            'x_limit': x_limit,
            't_limit': t_limit,
        },
        'model_file': model_file,
    }, metadata_dir, LABEL, 'MLP')
