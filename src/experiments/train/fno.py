"""Model 3 of the ladder: an FNO learning the parameter -> solution operator.

Unlike the pointwise models this one is trained once on the whole 20-point
alpha = beta family, so answering a new parameter costs a forward pass rather
than a fresh optimization.
"""

import os

import torch
from torch.optim import Adam

from methods.fno import FNO2d
from tools import (
    is_log_epoch, load_dataset, log_epoch, normalize_dataset, save_model, set_seed,
)
from experiments.common import (
    BandTracker, StageTimer, resolve_device, stage_paths, write_stage_metadata,
)

LABEL = 'fno'


def load_operator_dataset(dataset_file):
    # shared by the FNO and the PINO: the dataset written by experiments/dataset.py,
    # with its normalization statistics recomputed if the file predates them
    if not os.path.exists(dataset_file):
        raise RuntimeError(
            f'dataset not found at {dataset_file}; run the dataset stage first '
            '(python src/experiments/dataset.py, or the whole pipeline via src/main.py)')
    x_train, y_train, norm_stats = load_dataset(dataset_file)
    print(f'  loaded dataset from {dataset_file}')
    if norm_stats is None:
        raise RuntimeError(
            f'{dataset_file} carries no normalization envelopes; it predates the '
            'shared [-1, 1] convention. Regenerate it with the dataset stage.')
    return x_train, y_train, norm_stats


def train_fno(dataset_file, epochs, batch_size, lr, modes1, modes2, width,
              print_interval, results_dir, track_param=None, seed=None):
    # per-stage reseed for a reproducible standalone run; see train_mlp
    if seed is not None:
        set_seed(seed)

    device = resolve_device()
    weights_dir, metadata_dir = stage_paths(results_dir, LABEL)

    x_train, y_train, norm_stats = load_operator_dataset(dataset_file)

    # band snapshots are taken on the sample nearest the median parameter (the
    # alpha channel is index 2, constant per sample), so every model's band
    # tracking refers to the same alpha = beta as the pointwise MLP and PINN.
    # Done before the tensors are overwritten by their normalized versions.
    track_idx = (0 if track_param is None
                 else int((x_train[:, 2, 0, 0] - track_param).abs().argmin()))
    ref_x_raw = x_train[track_idx:track_idx + 1].clone().cpu()   # (1, C_in, Nx, Nt)
    bands = BandTracker(y_train[track_idx, 0].numpy())           # (Nx, Nt), eta channel
    x_train, y_train = normalize_dataset(x_train, y_train, norm_stats)

    model = FNO2d(modes1=modes1, modes2=modes2, width=width).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  {LABEL}: {num_params:,} parameters on {device}')
    optimizer = Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True)

    history = []
    timer = StageTimer()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        history.append(epoch_loss)

        if is_log_epoch(epoch, epochs, print_interval):
            log_epoch(epoch + 1, epochs, timer.elapsed, total_loss=epoch_loss)
            bands.record(epoch + 1, model.predict_physical(ref_x_raw, norm_stats)[0, 0].numpy())
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
            'batch_size': batch_size,
            'lr': lr,
            'modes1': modes1,
            'modes2': modes2,
            'width': width,
            'dataset_file': dataset_file,
        },
        'model_file': model_file,
        'dataset_file': dataset_file,
        'norm_stats': norm_stats,
    }, metadata_dir, LABEL, 'FNO')
