import os
import numpy as np
import torch

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINN.PINN import PINN
from tools import load_model
from _plots import (
    plot_training_statistics,
    plot_relative_error_panel,
    plot_stacked_solution_curves,
    plot_model2_resolution_panel,
    plot_model2_spectral_panel,
    plot_solution_surface_3d,
    plot_radially_revolved_gif,
    save_solution_gif,
    compute_spectral_relative_error,
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

        x_pred = np.linspace(-x_limit, x_limit, res, dtype=np.float32)
        t_pred = np.linspace(0.0, t_limit, res, dtype=np.float32)
        X, T = np.meshgrid(x_pred, t_pred, indexing='xy')
        x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
        t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

        model.eval()
        with torch.no_grad():
            eta_pred, _ = model(x_tensor, t_tensor)
        eta_pred = eta_pred.cpu().numpy().reshape(res, res).T

        # save gif for median resolution (use median_param supplied above)
        if res == median_res:
            save_solution_gif(
                x_pred,
                t_pred,
                eta_true_t,
                eta_pred,
                outdir=outdir,
                filename=f'{label}_animation_a{median_param:.3f}_res{res}.gif',
                title=f'{label} animation a={median_param:.3f} res={res}',
            )
            plot_solution_surface_3d(
                x_pred,
                t_pred,
                eta_true_t,
                eta_pred,
                outdir=outdir,
                filename=f'{label}_surface_a{median_param:.3f}_res{res}.png',
                title=f'{label} surface a={median_param:.3f} res={res}',
                true_label='Reference',
                pred_label='Prediction',
            )
            plot_radially_revolved_gif(
                x_pred,
                t_pred,
                eta_true_t,
                eta_pred,
                outdir=outdir,
                filename=f'{label}_radial_evolution_a{median_param:.3f}_res{res}.gif',
                title=f'{label.upper()} Radial Evolution a={median_param:.3f}',
                true_label='Reference',
                pred_label='Prediction',
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
        title=f'{"PINN" if label == "pinn" else "PINN No Data"} Resolution Panel (α=β {median_param:.3f})',
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
    X, T = np.meshgrid(x_pred, t_pred, indexing='xy')
    x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
    t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

    model.eval()
    with torch.no_grad():
        eta_pred, _ = model(x_tensor, t_tensor)
    eta_pred = eta_pred.cpu().numpy().reshape(spectral_res, spectral_res).T

    # save gif for median parameter at median resolution
    if median_param is not None:
        save_solution_gif(
            x_pred,
            t_pred,
            eta_true_t,
            eta_pred,
            outdir=outdir,
            filename=f'{label}_animation_a{median_param:.3f}_res{int(spectral_res)}.gif',
            title=f'{label} animation a={median_param:.3f} res={int(spectral_res)}',
        )

    plot_model2_spectral_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        outdir=outdir,
        filename=f'{label}_model2_spectral_panel.png',
        title=f'{"PINN" if label == "pinn" else "PINN No Data"} Spectral Panel (α=β {median_param:.3f}, res {int(spectral_res)})',
        param_label=f'{median_param:.3f}',
        res_label=f'{int(spectral_res)}',
    )