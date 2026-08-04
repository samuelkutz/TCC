"""Evaluation figures for the FNO: rebuild the operator, then run the shared driver."""

import os

from methods.fno import FNO2d
from tools import load_model
from experiments.common import resolve_device
from experiments.evaluate import (
    evaluate_operator_model, load_stage_metadata, operator_predictor,
    plot_stage_training_statistics, render_solution_gifs,
)

LABEL = 'fno'


def _load(model_metadata_file, device):
    metadata = load_stage_metadata(model_metadata_file)
    params = metadata['params']
    model = FNO2d(modes1=params['modes1'], modes2=params['modes2'],
                  width=params['width']).to(device)
    load_model(metadata['model_file'], model, device=device)
    return metadata, model


def eval_fno(model_metadata_file, x_limit, t_limit, eval_params, resolutions,
             spectral_res, output_dir=None):
    device = resolve_device()
    metadata, model = _load(model_metadata_file, device)

    outdir = output_dir or os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)
    plot_stage_training_statistics(metadata, LABEL, outdir)

    evaluate_operator_model(
        operator_predictor(model, metadata.get('norm_stats'), device),
        LABEL, x_limit, t_limit, eval_params, resolutions,
        spectral_panel_res=int(spectral_res),
        outdir=outdir,
    )


def gif_fno(model_metadata_file, x_limit, t_limit, params, resolution, outdir):
    device = resolve_device()
    metadata, model = _load(model_metadata_file, device)
    render_solution_gifs(
        operator_predictor(model, metadata.get('norm_stats'), device),
        LABEL, 'FNO', x_limit, t_limit, params, resolution, outdir,
    )
