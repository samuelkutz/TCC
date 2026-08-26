"""Pipeline driver: dataset, the six trained models, the NTK probe, every figure.

Runs top to bottom with no optional stages. Each stage takes its arguments from
`settings.json` through `config.Config`, and each fails immediately if an input
it needs is missing, so a broken stage does not surface as a confusing failure
several stages later.

Figures are written under `results/imgs` in two experiment sets:

    boussinesq_sciml/            the six trained models, grouped as
        nn/{pure_data,data_and_physics,pure_physics}   MLP, PINN(+data), PINN(phys)
        no/{pure_data,data_and_physics,pure_physics}   FNO,  PINO(+data), PINO(phys)
        comparison/              the six-model spectral-bias band tracking
    boussinesq_spectral_bias_mlp/  the R -> R spectral-bias probe (+ ntk/ drift)
    setup/                       the reference soliton profile

Every figure is drawn from saved weights/metadata, so `--plots-only` skips
training and regenerates the whole figure tree from the saved `.pth` files.
"""

import os
import shutil
import sys

from config import Config
from tools import set_seed
from experiments.common import BAND_METRIC, resolve_device
from experiments.dataset import run_dataset
from experiments.evaluate import load_stage_metadata
from experiments.plots.figures import plot_soliton_profile, plot_spectral_bias_panel
from experiments.plots.fno import eval_fno, gif_fno
from experiments.plots.mlp import eval_mlp, gif_mlp
from experiments.plots.ntk import plot_ntk
from experiments.plots.pinn import eval_pinn, gif_pinn
from experiments.plots.pino import eval_pino, gif_pino
from experiments.train.fno import train_fno
from experiments.train.mlp import train_mlp
from experiments.train.ntk import train_ntk
from experiments.train.pinn import train_pinn
from experiments.train.pino import train_pino

# Set 1: the six SciML models, keyed by architecture family and training regime.
SCIML = 'boussinesq_sciml'


def _stage(name):
    print(f'\n=== {name} ===')


def _metadata_files(cfg):
    """Every stage's metadata path, resolved once and shared by train and plot."""
    return {
        'mlp': cfg.metadata_file('mlp', 'mlp'),
        'fno': cfg.metadata_file('fno', 'fno'),
        'pino_data': cfg.metadata_file('pino', 'pino', 'data_and_physics'),
        'pino_no_data': cfg.metadata_file('pino', 'pino', 'pure_physics'),
        'pinn_data': cfg.metadata_file('pinn', 'pinn', 'data_and_physics'),
        'pinn_no_data': cfg.metadata_file('pinn', 'pinn', 'pure_physics'),
        'ntk': cfg.metadata_file(os.path.join('mlp', 'ntk'), 'mlp_ntk'),
    }


def plot_spectral_bias_evolution(ordered_metadata, outdir,
                                 filename='spectral_bias_evolution.png'):
    """The six-model band-tracking figure, read back from the saved histories."""
    models_data = []
    for label, metadata_file in ordered_metadata:
        history = load_stage_metadata(metadata_file).get('spectral_history')
        if not history or not history.get('epochs'):
            raise ValueError(f'no spectral_history in {metadata_file}; the training '
                             'stage must record band errors for this figure')
        if history.get('metric') != BAND_METRIC:
            print(f'  WARNING: {label} band history was recorded under '
                  f'{history.get("metric", "an earlier measure")!r}, not {BAND_METRIC!r}; '
                  'the panel is stale until this model is retrained')
        models_data.append((label, history))
    plot_spectral_bias_panel(models_data, outdir=outdir, filename=filename)


def train_all(cfg, device):
    """Dataset, the six models and the probe. Each writes its own `.pth`."""
    _stage('shared dataset')
    run_dataset(device=device, **cfg.dataset_kwargs)

    _stage('training mlp (data only)')
    train_mlp(**cfg.mlp_kwargs)

    _stage('training fno')
    train_fno(**cfg.fno_kwargs)

    _stage('training pino (data and physics)')
    train_pino('data_and_physics', **cfg.pino_kwargs)

    _stage('training pino (pure physics)')
    train_pino('pure_physics', **cfg.pino_kwargs)

    _stage('training pinn (data and physics)')
    train_pinn('data_and_physics', **cfg.pinn_kwargs)

    _stage('training pinn (pure physics)')
    train_pinn('pure_physics', **cfg.pinn_kwargs)

    _stage('training spectral-bias probe (NTK)')
    train_ntk(**cfg.ntk_kwargs)


def plot_all(cfg, meta):
    """Every figure and gif, drawn from the saved weights/metadata alone.

    Reads nothing a training stage leaves in memory, so it reproduces the whole
    figure tree from the committed `.pth` files (this is what `--plots-only`
    runs). Each model's figures land in its Set-1 slot; the probe is Set 2.
    """
    _stage('soliton profile')
    plot_soliton_profile(outdir=cfg.img_subdir('setup'), amplitude=cfg.amplitude)

    _stage('plotting mlp (data only)')
    eval_mlp(meta['mlp'], output_dir=cfg.img_subdir(SCIML, 'nn', 'pure_data'),
             **cfg.eval_kwargs)

    _stage('plotting fno')
    eval_fno(meta['fno'], output_dir=cfg.img_subdir(SCIML, 'no', 'pure_data'),
             **cfg.eval_kwargs)

    _stage('plotting pino (data and physics)')
    eval_pino('data_and_physics', meta['pino_data'],
              output_dir=cfg.img_subdir(SCIML, 'no', 'data_and_physics'), **cfg.eval_kwargs)

    _stage('plotting pino (pure physics)')
    eval_pino('pure_physics', meta['pino_no_data'],
              output_dir=cfg.img_subdir(SCIML, 'no', 'pure_physics'), **cfg.eval_kwargs)

    _stage('plotting pinn (data and physics)')
    eval_pinn('data_and_physics', meta['pinn_data'],
              output_dir=cfg.img_subdir(SCIML, 'nn', 'data_and_physics'), **cfg.eval_kwargs)

    _stage('plotting pinn (pure physics)')
    eval_pinn('pure_physics', meta['pinn_no_data'],
              output_dir=cfg.img_subdir(SCIML, 'nn', 'pure_physics'), **cfg.eval_kwargs)

    _stage('plotting spectral-bias probe (NTK)')
    plot_ntk(meta['ntk'], **cfg.ntk_plot_kwargs)

    _stage('spectral bias evolution')
    plot_spectral_bias_evolution(
        ordered_metadata=[
            ('MLP (pure data)', meta['mlp']),
            ('PINN (pure physics)', meta['pinn_no_data']),
            ('PINN (data and physics)', meta['pinn_data']),
            ('PINO (pure physics)', meta['pino_no_data']),
            ('PINO (data and physics)', meta['pino_data']),
            ('FNO (pure data)', meta['fno']),
        ],
        outdir=cfg.img_subdir(SCIML, 'comparison'),
    )

    # gifs: eta(x, t) evolving in time against the reference. Every model is shown
    # at the same two parameters, the median and the largest evaluation parameter.
    # The pointwise MLP and PINN are fixed to alpha = beta = median, so their
    # largest-parameter gif shows the fixed prediction against the shifted reference,
    # i.e. their lack of generalization across the parameter axis.
    gif_params = [cfg.median_param, sorted(cfg.eval_params)[-1]]

    _stage('mlp gifs')
    gif_mlp(meta['mlp'], x_limit=cfg.x_limit, t_limit=cfg.t_limit,
            params=[gif_params[0]], resolution=cfg.dataset_res, outdir=cfg.gif_dir)

    _stage('fno gifs')
    gif_fno(meta['fno'], x_limit=cfg.x_limit, t_limit=cfg.t_limit,
            params=gif_params, resolution=cfg.dataset_res, outdir=cfg.gif_dir)

    _stage('pino gifs (data and physics + pure physics)')
    gif_pino('data_and_physics', meta['pino_data'], x_limit=cfg.x_limit, t_limit=cfg.t_limit,
             params=gif_params, resolution=cfg.dataset_res, outdir=cfg.gif_dir)
    gif_pino('pure_physics', meta['pino_no_data'], x_limit=cfg.x_limit, t_limit=cfg.t_limit,
             params=gif_params, resolution=cfg.dataset_res, outdir=cfg.gif_dir)

    _stage('pinn gifs (data and physics + pure physics)')
    gif_pinn('data_and_physics', meta['pinn_data'], x_limit=cfg.x_limit, t_limit=cfg.t_limit,
             params=[gif_params[0]], resolution=cfg.dataset_res, outdir=cfg.gif_dir)
    gif_pinn('pure_physics', meta['pinn_no_data'], x_limit=cfg.x_limit, t_limit=cfg.t_limit,
             params=[gif_params[0]], resolution=cfg.dataset_res, outdir=cfg.gif_dir)


def main(plots_only=False):
    cfg = Config()
    set_seed(cfg.seed)
    device = resolve_device()
    print(f'device: {device}, seed: {cfg.seed}, '
          f'reseed_each_stage: {cfg.reseed_each_stage}, plots_only: {plots_only}')

    meta = _metadata_files(cfg)
    # plot_all regenerates the whole figure tree, so clear the previous one first
    # to avoid leaving orphaned files behind if the layout or naming changed
    if os.path.isdir(cfg.img_dir):
        shutil.rmtree(cfg.img_dir)
    os.makedirs(os.path.dirname(cfg.dataset_file), exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)

    if not plots_only:
        train_all(cfg, device)

    plot_all(cfg, meta)
    print('\nall experiments completed successfully.')


if __name__ == '__main__':
    main(plots_only=('--plots-only' in sys.argv))
