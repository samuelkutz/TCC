import os
import torch
from timeit import default_timer
from torch.optim import Adam

from tools import (
    compute_norm_stats, load_dataset, normalize_dataset, save_model,
    normalize_tensor, unnormalize_tensor, compute_spectral_band_errors, save_metadata_json,
)
from methods.pino import PINO2d, pino_loss


def train_pino(mode, dataset_file, x_limit, t_limit, epochs, batch_size, lr, phys_weight, ic_weight, data_weight, modes1, modes2, width, print_interval, results_dir, phys_nx=None, phys_nt=None, dealias=False):
    data_weight = data_weight if mode == 'data' else 0.0
    label = 'pino' if mode == 'data' else 'pino_no_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, 'models', 'weights', 'pino', 'with_data' if mode == 'data' else 'no_data')
    metadata_dir = os.path.join(results_dir, 'models', 'metadata', 'pino', 'with_data' if mode == 'data' else 'no_data')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    if os.path.exists(dataset_file):
        x_train, y_train, norm_stats = load_dataset(dataset_file)
        print(f'loaded dataset from {dataset_file}')
    else:
        raise RuntimeError(
            f'Dataset not found at {dataset_file}. '
            'Generate it first using `python src/BOUSSINESQ/run_dataset.py`.'
        )

    if norm_stats is None:
        norm_stats = compute_norm_stats(x_train, y_train)

    # store raw reference sample before normalization for spectral snapshots
    ref_x_raw = x_train[0:1].clone().cpu()   # (1, C_in, Nx, Nt)
    eta_true_ref = y_train[0, 0].numpy().astype('float64')  # (Nx, Nt), eta channel

    x_train, y_train = normalize_dataset(x_train, y_train, norm_stats)
    spectral_history = {'epochs': [], 'low_band': [], 'mid_band': [], 'high_band': []}
    norm_stats = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in norm_stats.items()
    }

    nx = x_train.shape[2]
    nt = x_train.shape[3]
    dx = 2.0 * x_limit / nx
    dt = t_limit / (nt - 1)
    print(f'computed physics spacing dx={dx:.6f}, dt={dt:.6f} for x_limit={x_limit}, t_limit={t_limit}')

    model = PINO2d(modes1=modes1, modes2=modes2, width=width, out_channels=y_train.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'model parameter count: {num_params:,}')
    optimizer = Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_history = []
    start_time = default_timer()
    print(f'starting pino training ({mode})...')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pde = 0.0
        epoch_ic = 0.0
        epoch_data = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_data = pino_loss(
                model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                dx=dx,
                dt=dt,
                norm_stats=norm_stats,
                phys_weight=phys_weight,
                ic_weight=ic_weight,
                data_weight=data_weight,
                x_limit=x_limit,
                t_limit=t_limit,
                phys_nx=phys_nx,
                phys_nt=phys_nt,
                dealias=dealias,
            )
            loss.backward()
            optimizer.step()

            epoch_data += loss_data.item()
            epoch_loss += loss.item()
            epoch_pde += loss_pde.item()
            epoch_ic += loss_ic.item()

        epoch_data /= len(train_loader)
        epoch_loss /= len(train_loader)
        epoch_pde /= len(train_loader)
        epoch_ic /= len(train_loader)
        train_history.append(epoch_loss)

        if (epoch + 1) % print_interval == 0 or epoch == epochs - 1:
            elapsed = default_timer() - start_time
            print(
                f'epoch {epoch + 1}, elapsed {elapsed:.1f}s, '
                f'total_loss {epoch_loss:.4e}, '
                f'pde_loss {epoch_pde:.4e}, '
                f'ic_loss {epoch_ic:.4e}, '
                f'data_loss {epoch_data:.4e}'
            )
            model.eval()
            with torch.no_grad():
                ref_norm = normalize_tensor(ref_x_raw.to(device), norm_stats['input_min'], norm_stats['input_max'], norm_stats['eps'])
                pred_norm = model(ref_norm)
                pred = unnormalize_tensor(pred_norm, norm_stats['output_min'], norm_stats['output_max'], norm_stats['eps'])
            model.train()
            eta_pred = pred[0, 0].cpu().numpy().astype('float64')  # (Nx, Nt)
            low, mid, high = compute_spectral_band_errors(eta_pred, eta_true_ref)
            spectral_history['epochs'].append(epoch + 1)
            spectral_history['low_band'].append(low)
            spectral_history['mid_band'].append(mid)
            spectral_history['high_band'].append(high)

    training_duration = default_timer() - start_time
    final_loss = train_history[-1] if train_history else None

    model_file = os.path.join(weights_dir, f'{label}_weights.pth')
    save_model(model, filepath=model_file)

    model_metadata = {
        'train_history': train_history,
        'spectral_history': spectral_history,
        'training_duration': training_duration,
        'final_loss': final_loss,
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
        'mode': mode,
        'norm_stats': norm_stats,
    }
    model_metadata_file = os.path.join(metadata_dir, f'{label}_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'pino model metadata saved to {model_metadata_file}')

    json_payload = {k: v for k, v in model_metadata.items() if k != 'norm_stats'}
    json_payload['model'] = 'PINO'
    save_metadata_json(json_payload, os.path.join(metadata_dir, f'{label}_metadata.json'))
