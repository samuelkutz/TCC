import os
import numpy as np
import torch

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINN.PINN import PINN
from tools import load_model
from plots import plot_training_statistics, plot_relative_error_panel, save_solution_gif, compute_spectral_relative_error

RESULTS_DIR = 'RESULTS'
PINN_X_LIMIT = 60.0
PINN_T_LIMIT = 15.0


def eval_pinn(mode='data', model_metadata_file=None, x_limit=60.0, t_limit=15.0):
    label = 'pinn' if mode == 'data' else 'pinn_no_data'
    subdir = 'with_data' if mode == 'data' else 'no_data'
    if model_metadata_file is None:
        model_metadata_file = os.path.join(RESULTS_DIR, 'pinn', subdir, 'models', f'{label}_model_metadata.pth')
    model_metadata = torch.load(model_metadata_file, map_location='cpu')
    params = model_metadata['params']
    model_file = model_metadata['model_file']

    outdir = os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)

    plot_training_statistics(
        [model_metadata['train_history']],
        [label],
        outdir=outdir,
        filename=f'{label}_training_statistics.png',
        duration_seconds=model_metadata.get('training_duration'),
        final_loss=model_metadata.get('final_loss'),
        num_params=model_metadata.get('num_params'),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, params['param_value'], params['param_value'])
    data = None
    if params['data_weight'] > 0:
        solver = PseudoSpectralBoussinesq(bsq, Nx=params['train_resolution'], Nt=params['train_resolution'], device=device)
        x_sol, t_sol, eta_sol, u_sol = solver.solve()
        data = {'x': x_sol, 't': t_sol, 'eta': eta_sol, 'u': u_sol}

    model = PINN(
        input_size=2,
        output_size=2,
        neurons=params['neurons'],
        hidden_layers=params['hidden_layers'],
        Boussinesq=bsq,
        domain_points=params['domain_points'],
        ic_points=params['ic_points'],
        optimizer_name=params['optimizer_name'],
        lr=params['lr'],
        data=data,
        data_weight=params['data_weight'],
        device=device,
    )
    load_model(model_file, model, device=device)

    print(f'start pinn evaluation ({mode})')
    x_pred = np.linspace(-x_limit, x_limit, params['train_resolution'], dtype=np.float32)
    t_pred = np.linspace(0.0, t_limit, params['train_resolution'], dtype=np.float32)
    X, T = np.meshgrid(x_pred, t_pred, indexing='xy')
    x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
    t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

    model.eval()
    with torch.no_grad():
        eta_pred, _ = model(x_tensor, t_tensor)
    eta_pred = eta_pred.cpu().numpy().reshape(params['train_resolution'], params['train_resolution']).T

    bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, params['param_value'], params['param_value'])
    solver_eval = PseudoSpectralBoussinesq(bsq_eval, Nx=params['train_resolution'], Nt=params['train_resolution'] - 1, device=device)
    x, t, eta_true, u_true = solver_eval.solve()
    eta_true_t = eta_true.T

    rel_error = np.linalg.norm(eta_true_t - eta_pred) / (np.linalg.norm(eta_true_t) + 1e-8)
    spec_error = compute_spectral_relative_error(eta_true_t, eta_pred)
    print(f'val={params["param_value"]:.3f} res={params["train_resolution"]} rel_error={rel_error:.4e} spec_error={spec_error:.4e}')

    plot_relative_error_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        times=[0.0, t_limit],
        outdir=outdir,
        filename=f'{label}_summary_a{params["param_value"]:.3f}_res{params["train_resolution"]}.png',
        title=f'{label} relative error a={params["param_value"]:.3f} res={params["train_resolution"]}',
    )
    save_solution_gif(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        outdir=outdir,
        filename=f'{label}_animation_a{params["param_value"]:.3f}_res{params["train_resolution"]}.gif',
        title=f'{label} animation a={params["param_value"]:.3f} res={params["train_resolution"]}',
    )


if __name__ == '__main__':
    eval_pinn()
