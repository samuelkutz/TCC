import os
import numpy as np
import torch

from methods.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from methods.pinn import PINN
from tools import load_model
from experiments.plots_common import (
    plot_training_statistics,
    plot_model2_resolution_panel,
    plot_model2_spectral_panel,
    save_solution_gif,
)

def eval_pinn(mode, model_metadata_file, x_limit, t_limit, eval_params, resolutions, spectral_res, output_dir=None):
    label = 'pinn' if mode == 'data' else 'pinn_no_data'

    model_metadata = torch.load(model_metadata_file, map_location='cpu')
    params = model_metadata['params']
    model_file = model_metadata['model_file']

    outdir = output_dir or os.path.dirname(os.path.dirname(model_metadata_file))
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
    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, params['param_value'], params['param_value'], 1)
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
        data=None,
        data_weight=params['data_weight'],
        device=device,
    )
    load_model(model_file, model, device=device)

    median_param = eval_params[len(eval_params) // 2]
    median_res = resolutions[len(resolutions) // 2]

    # PINN uses only one alpha=beta evaluation value for these panels.
    print(f'start pinn evaluation with single alpha=beta={median_param:.3f}')

    res_x_list = []
    res_t_list = []
    res_true_list = []
    res_pred_list = []
    print('start pinn evaluation resolution panel')
    for res in resolutions:
        bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
        solver = PseudoSpectralBoussinesq(bsq_eval, Nx=res, Nt=res - 1, device=device)
        x, t, eta_true, u_true = solver.solve()
        eta_true_t = eta_true.T

        eta_pred = model.predict_eta_grid(
            np.linspace(-x_limit, x_limit, res, dtype=np.float32),
            np.linspace(0.0, t_limit, res, dtype=np.float32),
        )

        res_true_list.append(eta_true_t)
        res_pred_list.append(eta_pred)
        res_x_list.append(x)
        res_t_list.append(t)

    plot_model2_resolution_panel(
        res_x_list,
        res_t_list,
        res_true_list,
        res_pred_list,
        resolutions,
        outdir=outdir,
        filename=f'{label}_model2_resolution_panel.png',
        title=f'{"PINN" if label == "pinn" else "PINN No Data"} Resolution Panel (alpha=beta {median_param:.3f})',
        param_label=f'{median_param:.3f}',
    )

    print('start pinn evaluation spectral panel')
    bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
    spectral_res = int(resolutions[0])
    solver = PseudoSpectralBoussinesq(bsq_eval, Nx=spectral_res, Nt=spectral_res - 1, device=device)
    x, t, eta_true, u_true = solver.solve()
    eta_true_t = eta_true.T

    x_pred = np.linspace(-x_limit, x_limit, spectral_res, dtype=np.float32)
    t_pred = np.linspace(0.0, t_limit, spectral_res, dtype=np.float32)
    eta_pred = model.predict_eta_grid(x_pred, t_pred)

    plot_model2_spectral_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        outdir=outdir,
        filename=f'{label}_model2_spectral_panel.png',
        title=f'{"PINN" if label == "pinn" else "PINN No Data"} Spectral Panel (alpha=beta {median_param:.3f}, res {int(spectral_res)})',
        param_label=f'{median_param:.3f}',
        res_label=f'{int(spectral_res)}',
    )


def gif_pinn(mode, model_metadata_file, x_limit, t_limit, params, resolution, outdir):
    # one gif per parameter: eta(x,t) evolving in time, reference vs PINN prediction
    label = 'pinn' if mode == 'data' else 'pinn_no_data'
    title_tag = 'PINN' if mode == 'data' else 'PINN (sem dados)'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_metadata = torch.load(model_metadata_file, map_location='cpu')
    p = model_metadata['params']
    model_file = model_metadata['model_file']

    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, p['param_value'], p['param_value'], 1)
    model = PINN(
        input_size=2,
        output_size=2,
        neurons=p['neurons'],
        hidden_layers=p['hidden_layers'],
        Boussinesq=bsq,
        domain_points=p['domain_points'],
        ic_points=p['ic_points'],
        optimizer_name=p['optimizer_name'],
        lr=p['lr'],
        data=None,
        data_weight=p['data_weight'],
        device=device,
    )
    load_model(model_file, model, device=device)
    model.eval()

    res = int(resolution)
    for val in params:
        bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, val, val, 1)
        solver = PseudoSpectralBoussinesq(bsq_eval, Nx=res, Nt=res - 1, device=device)
        x, t, eta_true, u_true = solver.solve()
        eta_true_t = eta_true.T

        x_pred = np.linspace(-x_limit, x_limit, res, dtype=np.float32)
        t_pred = np.linspace(0.0, t_limit, res, dtype=np.float32)
        X, T = np.meshgrid(x_pred, t_pred, indexing='xy')
        x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
        t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

        with torch.no_grad():
            eta_pred, _ = model(x_tensor, t_tensor)
        eta_pred = eta_pred.cpu().numpy().reshape(res, res).T

        save_solution_gif(
            x, t, eta_true_t, eta_pred,
            outdir=outdir,
            filename=f'{label}_{val:.2f}.gif',
            title_prefix=rf'{title_tag}  $\alpha=\beta={val:.2f}$',
        )
