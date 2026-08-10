"""Model 4 of the ladder: the FNO backbone plus the Boussinesq residual.

`mode` toggles the two regimes exactly as it does for the PINN:
'data_and_physics' keeps the supervised term, 'pure_physics' zeroes its weight.
Every other hyperparameter matches the FNO, so a difference in behaviour traces
to the physics term alone.
"""

import os

import torch
from torch.optim import Adam

from methods.pino import PINO2d, pino_loss
from tools import (
    from_symmetric, is_log_epoch, log_epoch, normalize_dataset, save_model,
    set_seed, to_symmetric,
)
from experiments.common import (
    BandTracker, StageTimer, resolve_device, stage_paths, write_stage_metadata,
)
from experiments.train.fno import load_operator_dataset


def _label(mode):
    # regime lives in the directory (stage_paths); the file label is just the
    # model name in both regimes
    return 'pino'


def train_pino(mode, dataset_file, x_limit, t_limit, epochs, batch_size, lr, phys_weight,
               ic_weight, data_weight, modes1, modes2, width, print_interval, results_dir,
               phys_nx=None, phys_nt=None, dealias=False, track_param=None, seed=None):
    label = _label(mode)
    data_weight = data_weight if mode == 'data_and_physics' else 0.0

    # per-stage reseed for a reproducible standalone run; see train_mlp
    if seed is not None:
        set_seed(seed)

    device = resolve_device()
    weights_dir, metadata_dir = stage_paths(results_dir, 'pino', mode)

    x_train, y_train, norm_stats = load_operator_dataset(dataset_file)

    # band snapshots on the sample nearest the median parameter (alpha channel is
    # index 2), so all six models track the same alpha = beta as the MLP/PINN
    track_idx = (0 if track_param is None
                 else int((x_train[:, 2, 0, 0] - track_param).abs().argmin()))
    ref_x_raw = x_train[track_idx:track_idx + 1].clone().cpu()   # (1, C_in, Nx, Nt)
    bands = BandTracker(y_train[track_idx, 0].numpy())           # (Nx, Nt), eta channel
    x_train, y_train = normalize_dataset(x_train, y_train, norm_stats)
    norm_stats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in norm_stats.items()}

    nx, nt = x_train.shape[2], x_train.shape[3]
    dx = 2.0 * x_limit / nx
    dt = t_limit / (nt - 1)
    print(f'  physics spacing dx={dx:.6f}, dt={dt:.6f} on [{-x_limit}, {x_limit}] x [0, {t_limit}]')

    model = PINO2d(modes1=modes1, modes2=modes2, width=width,
                   out_channels=y_train.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  {label}: {num_params:,} parameters on {device}')
    optimizer = Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True)

    history = []
    timer = StageTimer()
    for epoch in range(epochs):
        model.train()
        totals = dict(total_loss=0.0, pde_loss=0.0, ic_loss=0.0, data_loss=0.0)
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_data = pino_loss(
                model=model, batch_x=batch_x, batch_y=batch_y, dx=dx, dt=dt,
                norm_stats=norm_stats, phys_weight=phys_weight, ic_weight=ic_weight,
                data_weight=data_weight, x_limit=x_limit, t_limit=t_limit,
                phys_nx=phys_nx, phys_nt=phys_nt, dealias=dealias,
            )
            loss.backward()
            optimizer.step()
            totals['total_loss'] += loss.item()
            totals['pde_loss'] += loss_pde.item()
            totals['ic_loss'] += loss_ic.item()
            totals['data_loss'] += loss_data.item()
        totals = {k: v / len(loader) for k, v in totals.items()}
        history.append(totals['total_loss'])

        if is_log_epoch(epoch, epochs, print_interval):
            log_epoch(epoch + 1, epochs, timer.elapsed, **totals)
            model.eval()
            with torch.no_grad():
                ref_norm = to_symmetric(ref_x_raw.to(device), norm_stats['input_min'],
                                        norm_stats['input_max'])
                pred = from_symmetric(model(ref_norm), norm_stats['output_min'],
                                      norm_stats['output_max'])
            model.train()
            bands.record(epoch + 1, pred[0, 0].cpu().numpy())

    training_duration = timer.elapsed
    model_file = save_model(model, os.path.join(weights_dir, f'{label}_weights.pth'))

    write_stage_metadata({
        'mode': mode,
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
            'x_limit': x_limit,
            't_limit': t_limit,
            'dx': dx,
            'dt': dt,
            'phys_weight': phys_weight,
            'ic_weight': ic_weight,
            'data_weight': data_weight,
            'phys_nx': phys_nx,
            'phys_nt': phys_nt,
            'dealias': dealias,
            'modes1': modes1,
            'modes2': modes2,
            'width': width,
            'out_channels': y_train.shape[1],
            'dataset_file': dataset_file,
        },
        'model_file': model_file,
        'dataset_file': dataset_file,
        'norm_stats': norm_stats,
    }, metadata_dir, label, 'PINO')
