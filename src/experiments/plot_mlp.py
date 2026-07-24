import os
import numpy as np
import torch

from methods.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from methods.mlp import MLP
from tools import load_model
from experiments.train_mlp import predict_eta_grid_mlp
from experiments.plots_common import (
    plot_training_statistics,
    plot_model2_resolution_panel,
    plot_model2_spectral_panel,
)
from experiments.plot_spectral_bias import plot_spectral_bias_panel


def eval_mlp(model_metadata_file, x_limit, t_limit, eval_params, resolutions, spectral_res, output_dir=None):
    label = 'mlp'

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
    model = MLP(
        input_size=2,
        output_size=1,
        neurons=params['neurons'],
        hidden_layers=params['hidden_layers'],
        activation=params['activation'],
        device=device,
    )
    load_model(model_file, model, device=device)

    median_param = eval_params[len(eval_params) // 2]
    median_res = resolutions[len(resolutions) // 2]

    # MLP is trained at a single alpha=beta value, matching the PINN panels.
    print(f'start mlp evaluation with single alpha=beta={median_param:.3f}')

    res_x_list = []
    res_t_list = []
    res_true_list = []
    res_pred_list = []
    print('start mlp evaluation resolution panel')
    for res in resolutions:
        bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
        solver = PseudoSpectralBoussinesq(bsq_eval, Nx=res, Nt=res - 1, device=device)
        x, t, eta_true, u_true = solver.solve()
        eta_true_t = eta_true.T

        eta_pred = predict_eta_grid_mlp(
            model,
            np.linspace(-x_limit, x_limit, res, dtype=np.float32),
            np.linspace(0.0, t_limit, res, dtype=np.float32),
            x_limit, t_limit,
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
        title=f'MLP Resolution Panel (alpha=beta {median_param:.3f})',
        param_label=f'{median_param:.3f}',
    )

    print('start mlp evaluation spectral panel')
    bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
    spectral_res = int(resolutions[0])
    solver = PseudoSpectralBoussinesq(bsq_eval, Nx=spectral_res, Nt=spectral_res - 1, device=device)
    x, t, eta_true, u_true = solver.solve()
    eta_true_t = eta_true.T

    x_pred = np.linspace(-x_limit, x_limit, spectral_res, dtype=np.float32)
    t_pred = np.linspace(0.0, t_limit, spectral_res, dtype=np.float32)
    eta_pred = predict_eta_grid_mlp(model, x_pred, t_pred, x_limit, t_limit)

    plot_model2_spectral_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        outdir=outdir,
        filename=f'{label}_model2_spectral_panel.png',
        title=f'MLP Spectral Panel (alpha=beta {median_param:.3f}, res {int(spectral_res)})',
        param_label=f'{median_param:.3f}',
        res_label=f'{int(spectral_res)}',
    )

    print('start mlp band-error panel')
    spectral_history = model_metadata.get('spectral_history')
    if spectral_history is not None and spectral_history.get('epochs'):
        plot_spectral_bias_panel(
            [('MLP (data only)', spectral_history)],
            outdir=outdir,
            filename=f'{label}_spectral_bias_panel.png',
        )
    else:
        print('  no spectral_history in mlp metadata, skipping band-error panel')
