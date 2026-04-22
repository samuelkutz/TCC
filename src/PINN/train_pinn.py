import os
import numpy as np
import torch
from timeit import default_timer

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINN.PINN import PINN
from tools import save_model

RESULTS_DIR = 'RESULTS'

PINN_EPOCHS = 5000
PINN_NEURONS = 64
PINN_HIDDEN_LAYERS = 4
PINN_DOMAIN_POINTS = 3000
PINN_IC_POINTS = 500
PINN_OPTIMIZER = 'Adam'
PINN_LR = 1e-3
PINN_DATA_WEIGHT = 1.0
PINN_TRAIN_RESOLUTION = 256
PARAM_VALUE = 2.55


def train_pinn(mode='data'):
    data_weight = PINN_DATA_WEIGHT if mode == 'data' else 0.0
    label = 'pinn' if mode == 'data' else 'pinn_no_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outdir = os.path.join(RESULTS_DIR, 'pinn', 'with_data' if mode == 'data' else 'no_data')
    model_dir = os.path.join(outdir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    bsq = Boussinesq(-30, 30, 0, 15, PARAM_VALUE, PARAM_VALUE)
    solver = PseudoSpectralBoussinesq(bsq, Nx=PINN_TRAIN_RESOLUTION, Nt=PINN_TRAIN_RESOLUTION, device=device)
    x_sol, t_sol, eta_sol, u_sol = solver.solve()
    data = {'x': x_sol, 't': t_sol, 'eta': eta_sol, 'u': u_sol} if data_weight > 0 else None

    model = PINN(
        input_size=2,
        output_size=2,
        neurons=PINN_NEURONS,
        hidden_layers=PINN_HIDDEN_LAYERS,
        Boussinesq=bsq,
        domain_points=PINN_DOMAIN_POINTS,
        ic_points=PINN_IC_POINTS,
        optimizer_name=PINN_OPTIMIZER,
        lr=PINN_LR,
        data=data,
        data_weight=data_weight,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {num_params:,}')

    print(f'starting pinn training ({mode})...')
    start_time = default_timer()
    history = model.run_train_loop(bsq, epochs=PINN_EPOCHS, seed=None, print_interval=500)
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
            'epochs': PINN_EPOCHS,
            'neurons': PINN_NEURONS,
            'hidden_layers': PINN_HIDDEN_LAYERS,
            'domain_points': PINN_DOMAIN_POINTS,
            'ic_points': PINN_IC_POINTS,
            'optimizer_name': PINN_OPTIMIZER,
            'lr': PINN_LR,
            'data_weight': data_weight,
            'train_resolution': PINN_TRAIN_RESOLUTION,
            'param_value': PARAM_VALUE,
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
