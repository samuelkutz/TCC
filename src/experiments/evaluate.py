"""Evaluation drivers shared by the six trained models.

Pointwise models (MLP, PINN) fit one alpha = beta, so they get a resolution sweep
and a spectral panel. Operator models (FNO, PINO) learn the whole family, so they
get an alpha/beta sweep as well. The four `plot_*.py` modules rebuild their
network from saved metadata and hand a predictor to one of the two drivers.

A predictor has the signature
    predict(param_value, x_limit, t_limit, resolution) -> (x, t, eta_true, eta_pred)
with both fields shaped (Nx, Nt).
"""

import os

import torch

from experiments.common import (
    device_norm_stats, operator_input_tensor, reference_solution, resolve_device,
)
from experiments.plots.figures import (
    plot_alpha_beta_panel, plot_resolution_panel, plot_spectral_panel,
    plot_training_statistics, save_solution_gif,
)


def plot_stage_training_statistics(metadata, label, outdir):
    return plot_training_statistics(
        [metadata['train_history']], [label],
        outdir=outdir, filename=f'{label}_training_statistics.png',
        duration_seconds=metadata.get('training_duration'),
        final_loss=metadata.get('final_loss'),
        num_params=metadata.get('num_params'),
    )


def operator_predictor(model, norm_stats, device):
    """Predictor for an FNO/PINO, normalizing in and out via `norm_stats`."""
    stats = device_norm_stats(norm_stats, device)

    def predict(param_value, x_limit, t_limit, resolution):
        x, t, eta_true, u_true = reference_solution(
            param_value, x_limit, t_limit, nx=resolution, nt=resolution - 1, device=device)
        inputs = operator_input_tensor(eta_true, u_true, param_value,
                                       nx=resolution, nt=resolution, device=device)
        # predict_physical returns (1, C_out, Nx, Nt); channel 0 is eta
        eta_pred = model.predict_physical(inputs, stats).squeeze().cpu().numpy()[0, :, :]
        return x, t, eta_true.T, eta_pred

    return predict


def pointwise_predictor(predict_eta_grid, device=None):
    """Predictor for an MLP/PINN given its (x, t) grid evaluator."""
    device = device or resolve_device()

    def predict(param_value, x_limit, t_limit, resolution):
        x, t, eta_true, _ = reference_solution(
            param_value, x_limit, t_limit, nx=resolution, nt=resolution - 1, device=device)
        # query on the solver's own grid, not a rebuilt one, so prediction and
        # reference are compared at the same points
        return x, t, eta_true.T, predict_eta_grid(x, t)

    return predict


def evaluate_operator_model(predict, label, x_limit, t_limit, eval_params, resolutions,
                            spectral_panel_res, outdir):
    median_param = eval_params[len(eval_params) // 2]
    median_res = resolutions[len(resolutions) // 2]

    print(f'  {label}: alpha/beta panel at resolution {int(median_res)}')
    true_list, pred_list = [], []
    for value in eval_params:
        x, t, eta_true, eta_pred = predict(value, x_limit, t_limit, median_res)
        true_list.append(eta_true)
        pred_list.append(eta_pred)
    plot_alpha_beta_panel(x, t, true_list, pred_list, eval_params,
                          outdir=outdir, filename=f'{label}_model2_alpha_beta_panel.png')

    print(f'  {label}: resolution panel at alpha=beta={median_param:.3f}')
    x_list, t_list, true_list, pred_list = [], [], [], []
    for resolution in resolutions:
        x, t, eta_true, eta_pred = predict(median_param, x_limit, t_limit, resolution)
        x_list.append(x)
        t_list.append(t)
        true_list.append(eta_true)
        pred_list.append(eta_pred)
    plot_resolution_panel(x_list, t_list, true_list, pred_list, resolutions,
                          outdir=outdir, filename=f'{label}_model2_resolution_panel.png')

    print(f'  {label}: spectral panel at resolution {int(spectral_panel_res)}')
    x_pred, t_pred, eta_true, eta_pred = predict(
        median_param, x_limit, t_limit, int(spectral_panel_res))
    plot_spectral_panel(x_pred, t_pred, eta_true, eta_pred,
                        outdir=outdir, filename=f'{label}_model2_spectral_panel.png')


def evaluate_pointwise_model(predict, label, x_limit, t_limit, eval_params, resolutions,
                             spectral_panel_res, outdir):
    # trained at a single alpha = beta, so there is no parameter sweep
    median_param = eval_params[len(eval_params) // 2]
    print(f'  {label}: single alpha=beta={median_param:.3f}, resolution panel')

    x_list, t_list, true_list, pred_list = [], [], [], []
    for resolution in resolutions:
        x, t, eta_true, eta_pred = predict(median_param, x_limit, t_limit, resolution)
        x_list.append(x)
        t_list.append(t)
        true_list.append(eta_true)
        pred_list.append(eta_pred)
    plot_resolution_panel(x_list, t_list, true_list, pred_list, resolutions,
                          outdir=outdir, filename=f'{label}_model2_resolution_panel.png')

    print(f'  {label}: spectral panel at resolution {int(spectral_panel_res)}')
    x_pred, t_pred, eta_true, eta_pred = predict(
        median_param, x_limit, t_limit, int(spectral_panel_res))
    plot_spectral_panel(x_pred, t_pred, eta_true, eta_pred,
                        outdir=outdir, filename=f'{label}_model2_spectral_panel.png')


def render_solution_gifs(predict, label, title_tag, x_limit, t_limit, param_values,
                         resolution, outdir):
    for value in param_values:
        x, t, eta_true, eta_pred = predict(value, x_limit, t_limit, int(resolution))
        save_solution_gif(
            x, t, eta_true, eta_pred, outdir=outdir,
            filename=f'{label}_{value:.2f}.gif',
            title_prefix=rf'{title_tag}  $\alpha=\beta={value:.2f}$',
        )


def load_stage_metadata(model_metadata_file):
    """The saved .pth: hyperparameters, weights path and (operators) normalization.

    Fails immediately and legibly if the file or the weights it names are absent,
    rather than letting a skipped training stage surface as an obscure error
    several stages later.
    """
    if not os.path.exists(model_metadata_file):
        raise FileNotFoundError(
            f'metadata not found: {model_metadata_file}. The matching training '
            'stage has not run (or did not finish); run it before this evaluation.')
    metadata = torch.load(model_metadata_file, map_location='cpu', weights_only=False)
    for key in ('params', 'model_file', 'train_history'):
        if key not in metadata:
            raise KeyError(f"{model_metadata_file} is missing '{key}'; it looks "
                           'incomplete. Re-run the training stage.')
    if not os.path.exists(metadata['model_file']):
        raise FileNotFoundError(
            f"weights not found: {metadata['model_file']}, named by "
            f'{model_metadata_file}. Re-run the training stage.')
    return metadata
