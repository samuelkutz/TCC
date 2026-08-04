"""Evaluation figures for the PINN, in both regimes (data and physics / pure physics)."""

import os

from methods.pinn import PINN
from tools import load_model
from experiments.common import boussinesq_at, resolve_device
from experiments.evaluate import (
    evaluate_pointwise_model, load_stage_metadata, plot_stage_training_statistics,
    pointwise_predictor, render_solution_gifs,
)


def _label(mode):
    return 'pinn'


def _title_tag(mode):
    return 'PINN (data and physics)' if mode == 'data_and_physics' else 'PINN (pure physics)'


def _load(model_metadata_file, x_limit, t_limit, device):
    metadata = load_stage_metadata(model_metadata_file)
    params = metadata['params']
    model = PINN(
        input_size=2,
        output_size=2,
        neurons=params['neurons'],
        hidden_layers=params['hidden_layers'],
        Boussinesq=boussinesq_at(params['param_value'], x_limit, t_limit),
        domain_points=params['domain_points'],
        ic_points=params['ic_points'],
        optimizer_name=params['optimizer_name'],
        lr=params['lr'],
        data=None,
        data_weight=params['data_weight'],
        device=device,
    )
    load_model(metadata['model_file'], model, device=device)
    return metadata, model


def eval_pinn(mode, model_metadata_file, x_limit, t_limit, eval_params, resolutions,
              spectral_res, output_dir=None):
    label = _label(mode)
    device = resolve_device()
    metadata, model = _load(model_metadata_file, x_limit, t_limit, device)

    outdir = output_dir or os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)
    plot_stage_training_statistics(metadata, label, outdir)

    evaluate_pointwise_model(
        pointwise_predictor(model.predict_eta_grid, device),
        label, x_limit, t_limit, eval_params, resolutions,
        spectral_panel_res=int(spectral_res),
        outdir=outdir,
    )


def gif_pinn(mode, model_metadata_file, x_limit, t_limit, params, resolution, outdir):
    device = resolve_device()
    _, model = _load(model_metadata_file, x_limit, t_limit, device)
    render_solution_gifs(
        pointwise_predictor(model.predict_eta_grid, device),
        _label(mode), _title_tag(mode), x_limit, t_limit, params, resolution, outdir,
    )
