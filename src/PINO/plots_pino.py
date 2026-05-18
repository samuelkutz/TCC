import os
import numpy as np
import torch

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINO.PINO import PINO2d
from tools import load_model, normalize_tensor, unnormalize_tensor
from _plots import (
    plot_training_statistics,
    plot_relative_error_panel,
    plot_stacked_solution_curves,
    plot_model2_alpha_beta_panel,
    plot_model2_resolution_panel,
    plot_model2_spectral_panel,
    plot_solution_surface_3d,
    plot_radially_revolved_gif,
    save_solution_gif,
    compute_spectral_relative_error,
)

def eval_pino(mode, model_metadata_file, x_limit, t_limit, eval_params, resolutions, spectral_res, output_dir=None):
    label = 'pino' if mode == 'data' else 'pino_no_data'

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
    norm_stats = model_metadata.get('norm_stats', None)
    if norm_stats is None:
        raise RuntimeError('Model metadata must contain norm_stats for consistent PINO evaluation.')
    norm_stats = {
        'input_min': norm_stats['input_min'].to(device),
        'input_max': norm_stats['input_max'].to(device),
        'output_min': norm_stats['output_min'].to(device),
        'output_max': norm_stats['output_max'].to(device),
        'eps': norm_stats.get('eps', 1e-12),
    }

    model = PINO2d(
        modes1=params['modes1'],
        modes2=params['modes2'],
        width=params['width'],
        out_channels=params['out_channels'],
    ).to(device)
    load_model(model_file, model, device=device)

    median_param = eval_params[len(eval_params) // 2]
    median_res = resolutions[len(resolutions) // 2]

    alpha_true_list = []
    alpha_pred_list = []
    print('start pino evaluation alpha/beta panel')
    for val in eval_params:
        bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, val, val, 1)
        solver = PseudoSpectralBoussinesq(bsq, Nx=median_res, Nt=median_res - 1, device=device)
        x, t, eta_true, u_true = solver.solve()
        eta_true_t = eta_true.T

        eta0 = np.asarray(eta_true[0, :], dtype=np.float32)
        u0 = np.asarray(u_true[0, :], dtype=np.float32)
        ch0 = np.tile(eta0[:, None], (1, median_res))
        ch1 = np.tile(u0[:, None], (1, median_res))
        ch2 = np.ones((median_res, median_res), dtype=np.float32) * val
        ch3 = np.ones((median_res, median_res), dtype=np.float32) * val
        input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1)
        input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        input_tensor = normalize_tensor(
            input_tensor,
            norm_stats['input_min'],
            norm_stats['input_max'],
            norm_stats['eps'],
        )

        model.eval()
        with torch.no_grad():
            pred = model(input_tensor)
        pred = unnormalize_tensor(
            pred,
            norm_stats['output_min'],
            norm_stats['output_max'],
            norm_stats['eps'],
        )

        eta_pred = pred.squeeze().cpu().numpy()[0, :, :]
        # save gif for median parameter at median resolution
        if val == median_param:
            save_solution_gif(
                x,
                t,
                eta_true_t,
                eta_pred,
                outdir=outdir,
                filename=f'{label}_animation_a{val:.3f}_res{int(median_res)}.gif',
                title=f'{label} animation a={val:.3f} res={int(median_res)}',
            )
        alpha_true_list.append(eta_true_t)
        alpha_pred_list.append(eta_pred)

    plot_model2_alpha_beta_panel(
        x,
        t,
        alpha_true_list,
        alpha_pred_list,
        eval_params,
        outdir=outdir,
        filename=f'{label}_model2_alpha_beta_panel.png',
        title=f'{"PINO" if label == "pino" else "PINO No Data"} Alpha-Beta Panel (res {int(median_res)})',
        res_label=f'{int(median_res)}',
    )

    res_x_list = []
    res_t_list = []
    res_true_list = []
    res_pred_list = []
    print('start pino evaluation resolution panel')
    for res in resolutions:
        bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
        solver = PseudoSpectralBoussinesq(bsq, Nx=res, Nt=res - 1, device=device)
        x, t, eta_true, u_true = solver.solve()
        eta_true_t = eta_true.T

        eta0 = np.asarray(eta_true[0, :], dtype=np.float32)
        u0 = np.asarray(u_true[0, :], dtype=np.float32)
        ch0 = np.tile(eta0[:, None], (1, res))
        ch1 = np.tile(u0[:, None], (1, res))
        ch2 = np.ones((res, res), dtype=np.float32) * median_param
        ch3 = np.ones((res, res), dtype=np.float32) * median_param
        input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1)
        input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
        input_tensor = normalize_tensor(
            input_tensor,
            norm_stats['input_min'],
            norm_stats['input_max'],
            norm_stats['eps'],
        )

        model.eval()
        with torch.no_grad():
            pred = model(input_tensor)
        pred = unnormalize_tensor(
            pred,
            norm_stats['output_min'],
            norm_stats['output_max'],
            norm_stats['eps'],
        )

        eta_pred = pred.squeeze().cpu().numpy()[0, :, :]
        # save gif for median resolution (use median_param supplied above)
        if res == median_res:
            save_solution_gif(
                x,
                t,
                eta_true_t,
                eta_pred,
                outdir=outdir,
                filename=f'{label}_animation_a{median_param:.3f}_res{res}.gif',
                title=f'{label} animation a={median_param:.3f} res={res}',
            )
            plot_solution_surface_3d(
                x,
                t,
                eta_true_t,
                eta_pred,
                outdir=outdir,
                filename=f'{label}_surface_a{median_param:.3f}_res{res}.png',
                title=f'{label} surface a={median_param:.3f} res={res}',
                true_label='Reference',
                pred_label='Prediction',
            )
            plot_radially_revolved_gif(
                x,
                t,
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
        title=f'{"PINO" if label == "pino" else "PINO No Data"} Resolution Panel (α=β {median_param:.3f})',
        param_label=f'{median_param:.3f}',
    )

    print('start pino evaluation spectral panel')
    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
    # evaluate true solution on the spectral grid to match x_pred/t_pred used for spectral panel
    solver = PseudoSpectralBoussinesq(bsq, Nx=spectral_res, Nt=spectral_res - 1, device=device)
    x, t, eta_true, u_true = solver.solve()
    eta_true_t = eta_true.T

    eta0 = np.asarray(eta_true[0, :], dtype=np.float32)
    u0 = np.asarray(u_true[0, :], dtype=np.float32)
    ch0 = np.tile(eta0[:, None], (1, spectral_res))
    ch1 = np.tile(u0[:, None], (1, spectral_res))
    ch2 = np.ones((spectral_res, spectral_res), dtype=np.float32) * median_param
    ch3 = np.ones((spectral_res, spectral_res), dtype=np.float32) * median_param
    input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1)
    input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    input_tensor = normalize_tensor(
        input_tensor,
        norm_stats['input_min'],
        norm_stats['input_max'],
        norm_stats['eps'],
    )

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)
    pred = unnormalize_tensor(
        pred,
        norm_stats['output_min'],
        norm_stats['output_max'],
        norm_stats['eps'],
    )

    eta_pred = pred.squeeze().cpu().numpy()[0, :, :]
    x_pred = np.linspace(-x_limit, x_limit, spectral_res, dtype=np.float32)
    t_pred = np.linspace(0.0, t_limit, spectral_res, dtype=np.float32)
    plot_model2_spectral_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        outdir=outdir,
        filename=f'{label}_model2_spectral_panel.png',
        title=f'{"PINO" if label == "pino" else "PINO No Data"} Spectral Panel (α=β {median_param:.3f}, res {int(spectral_res)})',
        param_label=f'{median_param:.3f}',
        res_label=f'{int(spectral_res)}',
    )