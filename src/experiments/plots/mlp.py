"""Evaluation figures for the data-only MLP."""

import os

from methods.mlp import MLP
from tools import load_model
from experiments.common import resolve_device
from experiments.evaluate import (
    evaluate_pointwise_model, load_stage_metadata, plot_stage_training_statistics,
    pointwise_predictor,
)
from experiments.plots.figures import plot_spectral_bias_panel
from experiments.predict import mlp_eta_grid

LABEL = 'mlp'


def eval_mlp(model_metadata_file, x_limit, t_limit, eval_params, resolutions,
             spectral_res, output_dir=None):
    device = resolve_device()
    metadata = load_stage_metadata(model_metadata_file)
    params = metadata['params']

    model = MLP(input_size=2, output_size=1, neurons=params['neurons'],
                hidden_layers=params['hidden_layers'], activation=params['activation'],
                device=device)
    load_model(metadata['model_file'], model, device=device)

    outdir = output_dir or os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)
    plot_stage_training_statistics(metadata, LABEL, outdir)

    def predict_eta_grid(x_query, t_query):
        return mlp_eta_grid(model, x_query, t_query, x_limit, t_limit)

    evaluate_pointwise_model(
        pointwise_predictor(predict_eta_grid, device),
        LABEL, x_limit, t_limit, eval_params, resolutions,
        spectral_panel_res=int(spectral_res),
        outdir=outdir,
    )

    # the MLP is the only model whose own band-tracking curve is also emitted next
    # to its panels, because Section 4.3 discusses it before the six-model figure
    spectral_history = metadata.get('spectral_history')
    if spectral_history and spectral_history.get('epochs'):
        plot_spectral_bias_panel([('MLP (data only)', spectral_history)],
                                 outdir=outdir, filename=f'{LABEL}_spectral_bias_panel.png')
    else:
        print('  no spectral_history in mlp metadata, skipping band-error panel')
