import os
import torch
from timeit import default_timer

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINN.PINN import PINN
from tools import save_model

RESULTS_DIR = 'RESULTS'


def train_pinn(mode='data',
               x_limit=60.0,
               t_limit=15.0,
               train_resolution=256,
               param_value=2.55,
               epochs=5000,
               neurons=64,
               hidden_layers=4,
               domain_points=3000,
               ic_points=500,
               optimizer_name='Adam',
               lr=1e-3,
               data_weight=1.0,
               print_interval=500):
    data_weight = data_weight if mode == 'data' else 0.0
    label = 'pinn' if mode == 'data' else 'pinn_no_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outdir = os.path.join(RESULTS_DIR, 'pinn', 'with_data' if mode == 'data' else 'no_data')
    model_dir = os.path.join(outdir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, param_value, param_value)
    solver = PseudoSpectralBoussinesq(bsq, Nx=train_resolution, Nt=train_resolution, device=device)
    x_sol, t_sol, eta_sol, u_sol = solver.solve()
    data = {'x': x_sol, 't': t_sol, 'eta': eta_sol, 'u': u_sol} if data_weight > 0 else None

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
    print(f'Model parameter count: {num_params:,}')

    print(f'starting pinn training ({mode})...')
    start_time = default_timer()
    history = model.run_train_loop(bsq, epochs=epochs, seed=None, print_interval=print_interval)
    training_duration = default_timer() - start_time
    final_loss = history[-1] if len(history) > 0 else None

    model_file = os.path.join(model_dir, f'{label}_weights.pth')
    save_model(model, filepath=model_file)

    model_metadata = {
        'train_history': history,
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
    model_metadata_file = os.path.join(model_dir, f'{label}_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'pinn model metadata saved to {model_metadata_file}')


def train_pinn_data():
    return train_pinn('data')


def train_pinn_no_data():
    return train_pinn('no_data')


if __name__ == '__main__':
    train_pinn()
