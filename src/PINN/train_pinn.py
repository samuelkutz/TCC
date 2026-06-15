import os
import numpy as np
import torch
from timeit import default_timer

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINN.PINN import PINN
from tools import save_model, compute_spectral_band_errors, save_metadata_json


def train_pinn(mode, x_limit, t_limit, train_resolution, param_value, epochs, neurons, hidden_layers, domain_points, ic_points, optimizer_name, lr, data_weight, print_interval, results_dir):
    data_weight = data_weight if mode == 'data' else 0.0
    label = 'pinn' if mode == 'data' else 'pinn_no_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, 'models', 'weights', 'pinn', 'with_data' if mode == 'data' else 'no_data')
    metadata_dir = os.path.join(results_dir, 'models', 'metadata', 'pinn', 'with_data' if mode == 'data' else 'no_data')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, param_value, param_value, 1)
    solver = PseudoSpectralBoussinesq(bsq, Nx=train_resolution, Nt=train_resolution, device=device)
    x_sol, t_sol, eta_sol, u_sol = solver.solve()
    data = {'x': x_sol, 't': t_sol, 'eta': eta_sol, 'u': u_sol} if data_weight > 0 else None

    # build fixed reference for spectral snapshots
    # eta_sol is (Nt+1, Nx); transpose to (Nx, Nt+1) for spatial-first FFT
    eta_true_ref = eta_sol.T.astype(np.float64)
    Nx_ref, Nt_ref = eta_true_ref.shape
    X_ref, T_ref = np.meshgrid(x_sol.astype(np.float32), t_sol.astype(np.float32), indexing='xy')
    x_tensor_ref = torch.from_numpy(X_ref.reshape(-1, 1)).to(device)
    t_tensor_ref = torch.from_numpy(T_ref.reshape(-1, 1)).to(device)
    spectral_history = {'epochs': [], 'low_band': [], 'mid_band': [], 'high_band': []}

    def spectral_snapshot(model, epoch):
        model.eval()
        with torch.no_grad():
            eta_flat, _ = model(x_tensor_ref, t_tensor_ref)
        model.train()
        eta_pred = eta_flat.cpu().numpy().reshape(Nt_ref, Nx_ref).T.astype(np.float64)
        low, mid, high = compute_spectral_band_errors(eta_pred, eta_true_ref)
        spectral_history['epochs'].append(epoch)
        spectral_history['low_band'].append(low)
        spectral_history['mid_band'].append(mid)
        spectral_history['high_band'].append(high)

    model = PINN(
        input_size=2,
        output_size=2,
        neurons=neurons,
        hidden_layers=hidden_layers,
        Boussinesq=bsq,
        domain_points=domain_points,
        ic_points=ic_points,
        optimizer_name=optimizer_name,
        lr=lr,
        data=data,
        data_weight=data_weight,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f'model parameter count: {num_params:,}')

    print(f'starting pinn training ({mode})...')
    start_time = default_timer()
    history = model.run_train_loop(bsq, epochs=epochs, seed=None, print_interval=print_interval, snapshot_callback=spectral_snapshot)
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
            'domain_points': domain_points,
            'ic_points': ic_points,
            'optimizer_name': optimizer_name,
            'lr': lr,
            'data_weight': data_weight,
            'train_resolution': train_resolution,
            'param_value': param_value,
            'x_limit': x_limit,
            't_limit': t_limit,
        },
        'model_file': model_file,
        'mode': mode,
    }
    model_metadata_file = os.path.join(metadata_dir, f'{label}_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'pinn model metadata saved to {model_metadata_file}')

    json_payload = {k: v for k, v in model_metadata.items() if k != 'norm_stats'}
    json_payload['model'] = 'PINN'
    save_metadata_json(json_payload, os.path.join(metadata_dir, f'{label}_metadata.json'))

