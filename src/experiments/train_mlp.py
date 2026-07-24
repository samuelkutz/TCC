import os
import numpy as np
import torch
import torch.optim as optim
from timeit import default_timer

from methods.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from methods.mlp import MLP
from tools import save_model, compute_spectral_band_errors, save_metadata_json


def _to_unit(x_np, t_np, x_limit, t_limit):
    # map physical (x, t) onto [-1, 1]^2: x in [-x_limit, x_limit] -> [-1, 1],
    # t in [0, t_limit] -> [-1, 1].
    x_u = np.asarray(x_np, dtype=np.float32) / x_limit
    t_u = 2.0 * np.asarray(t_np, dtype=np.float32) / t_limit - 1.0
    return x_u, t_u


def predict_eta_grid_mlp(model, x_np, t_np, x_limit, t_limit):
    # evaluate the data-only MLP on the tensor-product grid (x_np x t_np); returns a
    # numpy array shaped (Nx, Nt). Inputs are mapped to [-1, 1]^2 before the forward
    # pass, mirroring the supervised training layout.
    x_u, t_u = _to_unit(x_np, t_np, x_limit, t_limit)
    Nx = x_u.shape[0]
    Nt = t_u.shape[0]
    X, T = np.meshgrid(x_u, t_u, indexing='xy')
    device = next(model.parameters()).device
    inp = torch.from_numpy(np.stack([X.reshape(-1), T.reshape(-1)], axis=-1)).float().to(device)
    model.eval()
    with torch.no_grad():
        eta_flat = model(inp)
    return eta_flat.cpu().numpy().reshape(Nt, Nx).T


def train_mlp(x_limit, t_limit, train_resolution, param_value, epochs, neurons,
              hidden_layers, activation, optimizer_name, lr, batch_size, print_interval,
              results_dir):
    label = 'mlp'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, 'models', 'weights', 'mlp')
    metadata_dir = os.path.join(results_dir, 'models', 'metadata', 'mlp')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # reference solution at alpha=beta=param_value on the train_resolution grid
    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, param_value, param_value, 1)
    solver = PseudoSpectralBoussinesq(bsq, Nx=train_resolution, Nt=train_resolution, device=device)
    x_sol, t_sol, eta_sol, u_sol = solver.solve()

    # supervised (x, t) -> eta pairs over the full grid, inputs mapped to [-1, 1]^2
    eta_true = eta_sol.T  # (Nx, Nt)
    x_u, t_u = _to_unit(x_sol, t_sol, x_limit, t_limit)
    X, T = np.meshgrid(x_u, t_u, indexing='xy')            # (Nt, Nx)
    inputs = torch.from_numpy(np.stack([X.reshape(-1), T.reshape(-1)], axis=-1)).float().to(device)
    # eta_sol is (Nt, Nx); flatten to match the (Nt, Nx) meshgrid ordering
    targets = torch.from_numpy(eta_sol.reshape(-1, 1)).float().to(device)

    # fixed reference for spectral snapshots (mirrors the PINN mechanism)
    eta_true_ref = eta_true.astype(np.float64)
    spectral_history = {'epochs': [], 'low_band': [], 'mid_band': [], 'high_band': []}

    model = MLP(input_size=2, output_size=1, neurons=neurons, hidden_layers=hidden_layers,
                activation=activation, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'model parameter count: {num_params:,}')

    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    use_batches = batch_size and batch_size > 0
    if use_batches:
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    start_time = default_timer()
    print('starting mlp training...')
    for ep in range(epochs):
        model.train()
        if use_batches:
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = torch.mean((pred - batch_y) ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
        else:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = torch.mean((pred - targets) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
        history.append(epoch_loss)

        if (ep + 1) % print_interval == 0 or ep == epochs - 1:
            elapsed = default_timer() - start_time
            print(f'epoch {ep + 1}, elapsed {elapsed:.1f}s, total_loss {epoch_loss:.4e}')
            eta_pred = predict_eta_grid_mlp(model, x_sol, t_sol, x_limit, t_limit).astype(np.float64)
            model.train()
            low, mid, high = compute_spectral_band_errors(eta_pred, eta_true_ref)
            spectral_history['epochs'].append(ep + 1)
            spectral_history['low_band'].append(low)
            spectral_history['mid_band'].append(mid)
            spectral_history['high_band'].append(high)

    training_duration = default_timer() - start_time
    final_loss = history[-1] if len(history) > 0 else None

    model_file = os.path.join(weights_dir, f'{label}_weights.pth')
    save_model(model, filepath=model_file)

    model_metadata = {
        'train_history': history,
        'spectral_history': spectral_history,
        'training_duration': training_duration,
        'final_loss': final_loss,
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
            'x_limit': x_limit,
            't_limit': t_limit,
        },
        'model_file': model_file,
        'mode': 'data',
    }
    model_metadata_file = os.path.join(metadata_dir, f'{label}_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'mlp model metadata saved to {model_metadata_file}')

    json_payload = {k: v for k, v in model_metadata.items() if k != 'norm_stats'}
    json_payload['model'] = 'MLP'
    save_metadata_json(json_payload, os.path.join(metadata_dir, f'{label}_metadata.json'))
