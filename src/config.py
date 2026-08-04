"""Turns `settings.json` into the arguments every stage takes.

The JSON is the only source of hyperparameters, domain constants and output
paths, and this is the only module that reads it, so the configuration table of
the methodology can be checked against the code by inspection.
"""

import json
import os

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
SETTINGS_FILE = os.path.join(SRC_DIR, 'settings.json')


def load_settings(path=SETTINGS_FILE):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


class Config:
    """Resolved configuration: derived constants, output paths, per-stage kwargs."""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else load_settings()
        s = self.settings

        self.seed = s['seed']
        # With reseed_each_stage False the pipeline seeds once and each stage
        # inherits the RNG state its predecessors left, which is how the committed
        # weights were produced. True makes every stage reproduce standalone.
        self.reseed_each_stage = bool(s.get('reseed_each_stage', False))
        self.stage_seed = self.seed if self.reseed_each_stage else None

        domain = s['domain']
        self.x_limit = domain['x_limit']
        self.t_limit = domain['t_limit']
        self.dataset_res = domain['dataset_res']
        self.amplitude = domain.get('amplitude', 1.0)
        spec = domain['param_values']
        self.param_values = list(np.linspace(spec['start'], spec['stop'], spec['n'],
                                             dtype=np.float32))

        self.eval_params = s['eval']['params']
        self.spectral_res = s['eval']['spectral_res']
        self.median_param = sorted(self.eval_params)[len(self.eval_params) // 2]
        self.resolutions = [self.dataset_res, self.dataset_res * 2, self.dataset_res * 4]

        # anchored to the repo root, not the working directory: the LaTeX
        # \graphicspath resolves figures at <repo>/results/imgs, so the pipeline
        # must write there whatever directory it is launched from
        self.results_dir = s['results_dir']
        if not os.path.isabs(self.results_dir):
            self.results_dir = os.path.join(REPO_ROOT, self.results_dir)
        self.img_dir = os.path.join(self.results_dir, 'imgs')
        self.gif_dir = os.path.join(self.img_dir, 'gifs')
        self.dataset_file = os.path.join(self.results_dir, 'models', 'boussinesq_dataset.pth')

    def img_subdir(self, *parts):
        return os.path.join(self.img_dir, *parts)

    def metadata_file(self, family, label, mode_dir=None):
        parts = [family] + ([mode_dir] if mode_dir else [])
        return os.path.join(self.results_dir, 'models', 'metadata', *parts,
                            f'{label}_model_metadata.pth')

    @property
    def dataset_kwargs(self):
        return dict(dataset_file=self.dataset_file, param_values=self.param_values,
                    x_limit=self.x_limit, t_limit=self.t_limit,
                    dataset_res=self.dataset_res, amplitude=self.amplitude)

    @property
    def mlp_kwargs(self):
        return dict(x_limit=self.x_limit, t_limit=self.t_limit,
                    train_resolution=self.dataset_res, param_value=self.median_param,
                    amplitude=self.amplitude, results_dir=self.results_dir,
                    seed=self.stage_seed, **self.settings['mlp'])

    @property
    def pinn_kwargs(self):
        return dict(x_limit=self.x_limit, t_limit=self.t_limit,
                    train_resolution=self.dataset_res, param_value=self.median_param,
                    amplitude=self.amplitude, results_dir=self.results_dir,
                    seed=self.stage_seed, **self.settings['pinn'])

    @property
    def fno_kwargs(self):
        return dict(dataset_file=self.dataset_file, results_dir=self.results_dir,
                    track_param=self.median_param, seed=self.stage_seed,
                    **self.settings['fno'])

    @property
    def pino_kwargs(self):
        return dict(dataset_file=self.dataset_file, x_limit=self.x_limit,
                    t_limit=self.t_limit, results_dir=self.results_dir,
                    track_param=self.median_param, seed=self.stage_seed,
                    **self.settings['pino'])

    @property
    def eval_kwargs(self):
        return dict(x_limit=self.x_limit, t_limit=self.t_limit,
                    eval_params=self.eval_params, resolutions=self.resolutions,
                    spectral_res=self.spectral_res)

    @property
    def ntk_kwargs(self):
        # the probe reseeds locally rather than inheriting the pipeline RNG state,
        # so it takes the global seed even when reseed_each_stage is off
        return dict(x_limit=self.x_limit, t_limit=self.t_limit,
                    param_value=self.median_param, amplitude=self.amplitude,
                    results_dir=self.results_dir, seed=self.seed,
                    **self.settings['spectral_bias_ntk'])

    @property
    def ntk_plot_kwargs(self):
        # the probe is the second experiment set: an R -> R network fitting the
        # single profile eta(x, 15), kept apart from the six SciML models
        return dict(outdir=self.img_subdir('boussinesq_spectral_bias_mlp'),
                    drift_outdir=self.img_subdir('boussinesq_spectral_bias_mlp', 'ntk'))
