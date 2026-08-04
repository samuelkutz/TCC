"""Evaluation figures for the PINO, in both regimes (data and physics / pure physics)."""

import os

from methods.pino import PINO2d
from tools import load_model
from experiments.common import resolve_device
from experiments.evaluate import (
    evaluate_operator_model, load_stage_metadata, operator_predictor,
    plot_stage_training_statistics, render_solution_gifs,
)


def _label(mode):
    return 'pino'


def _title_tag(mode):
    return 'PINO (data and physics)' if mode == 'data_and_physics' else 'PINO (pure physics)'


def _load(model_metadata_file, device):
    metadata = load_stage_metadata(model_metadata_file)
    params = metadata['params']
    model = PINO2d(modes1=params['modes1'], modes2=params['modes2'],
                   width=params['width'], out_channels=params['out_channels']).to(device)
    load_model(metadata['model_file'], model, device=device)
    return metadata, model


def eval_pino(mode, model_metadata_file, x_limit, t_limit, eval_params, resolutions,
              spectral_res, output_dir=None):
    label = _label(mode)
    device = resolve_device()
    metadata, model = _load(model_metadata_file, device)

    outdir = output_dir or os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)
    plot_stage_training_statistics(metadata, label, outdir)

    evaluate_operator_model(
        operator_predictor(model, metadata.get('norm_stats'), device),
        label, x_limit, t_limit, eval_params, resolutions,
        spectral_panel_res=int(spectral_res),
        outdir=outdir,
    )


def gif_pino(mode, model_metadata_file, x_limit, t_limit, params, resolution, outdir):
    device = resolve_device()
    metadata, model = _load(model_metadata_file, device)
    render_solution_gifs(
        operator_predictor(model, metadata.get('norm_stats'), device),
        _label(mode), _title_tag(mode), x_limit, t_limit, params, resolution, outdir,
    )
