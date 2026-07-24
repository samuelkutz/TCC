import os
import json
import random
import numpy as np
import torch

from experiments.dataset import run_dataset
from experiments.train_fno import train_fno
from experiments.plot_fno import eval_fno, gif_fno
from experiments.train_pino import train_pino
from experiments.plot_pino import eval_pino, gif_pino
from experiments.train_pinn import train_pinn
from experiments.plot_pinn import eval_pinn, gif_pinn
from experiments.train_mlp import train_mlp
from experiments.plot_mlp import eval_mlp
from experiments.ntk import run_ntk_experiment
from experiments.plots_common import plot_soliton_profile
from experiments.plot_spectral_bias import plot_spectral_bias_evolution
from experiments.spectral_bias_theory import run_spectral_bias_theory


with open(os.path.join(os.path.dirname(__file__), 'settings.json')) as settings_file:
    settings = json.load(settings_file)

RESULTS_DIR      = settings['results_dir']
SEED             = settings['seed']
X_LIMIT          = settings['domain']['x_limit']
T_LIMIT          = settings['domain']['t_limit']
DATASET_RES      = settings['domain']['dataset_res']
EVAL_PARAMS      = settings['eval']['params']

param_values_spec = settings['domain']['param_values']
PARAM_VALUES      = list(np.linspace(param_values_spec['start'], param_values_spec['stop'], param_values_spec['n'], dtype=np.float32))
MEDIAN_PDE_PARAM  = sorted(EVAL_PARAMS)[len(EVAL_PARAMS) // 2]

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_FILE = os.path.join(RESULTS_DIR, 'models', 'boussinesq_dataset.pth')

FNO_CONFIG = {
    'dataset_file': DATASET_FILE,
    'results_dir':  RESULTS_DIR,
    **settings['fno'],
}

PINO_CONFIG = {
    'dataset_file': DATASET_FILE,
    'x_limit':      X_LIMIT,
    't_limit':      T_LIMIT,
    'results_dir':  RESULTS_DIR,
    **settings['pino'],
}

PINN_CONFIG = {
    'x_limit':          X_LIMIT,
    't_limit':          T_LIMIT,
    'train_resolution': DATASET_RES,
    'param_value':      MEDIAN_PDE_PARAM,
    'results_dir':      RESULTS_DIR,
    **settings['pinn'],
}

MLP_CONFIG = {
    'x_limit':          X_LIMIT,
    't_limit':          T_LIMIT,
    'train_resolution': DATASET_RES,
    'param_value':      MEDIAN_PDE_PARAM,
    'results_dir':      RESULTS_DIR,
    **settings['mlp'],
}

EVAL_CONFIG = {
    'x_limit':      X_LIMIT,
    't_limit':      T_LIMIT,
    'eval_params':  EVAL_PARAMS,
    'resolutions':  [DATASET_RES, DATASET_RES * 2, DATASET_RES * 4],
    'spectral_res': settings['eval']['spectral_res'],
}

IMG_DIR = os.path.join(RESULTS_DIR, 'imgs')

PINN_NTK_CONFIG = {
    'x_limit': X_LIMIT,
    't_limit': T_LIMIT,
    'outdir':  os.path.join(IMG_DIR, 'pinn', 'ntk'),
    'seed':    SEED,
    **settings['pinn_ntk'],
}

# the spectral-bias theory (toy Fourier-feature model) is the theoretical companion
# to the MLP's empirical spectral bias, so its figures live alongside the MLP figures
SPECTRAL_BIAS_THEORY_CONFIG = {
    'outdir': os.path.join(IMG_DIR, 'mlp'),
    **settings['spectral_bias_theory'],
}

MLP_EVAL_DIR = os.path.join(IMG_DIR, 'mlp')
FNO_EVAL_DIR = os.path.join(IMG_DIR, 'fno')
PINO_WITH_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pino', 'with_data')
PINO_NO_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pino', 'no_data')
PINN_WITH_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pinn', 'with_data')
PINN_NO_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pinn', 'no_data')

MLP_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'mlp', 'mlp_model_metadata.pth')
FNO_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'fno', 'fno_model_metadata.pth')
PINO_WITH_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pino', 'with_data', 'pino_model_metadata.pth')
PINO_NO_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pino', 'no_data', 'pino_no_data_model_metadata.pth')
PINN_WITH_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pinn', 'with_data', 'pinn_model_metadata.pth')
PINN_NO_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pinn', 'no_data', 'pinn_no_data_model_metadata.pth')

# gifs: solution evolving in time (orange) vs reference, for the median and last eval params
GIF_DIR        = os.path.join(IMG_DIR, 'gifs')
GIF_LAST_PARAM = sorted(EVAL_PARAMS)[-1]
GIF_PARAMS     = [MEDIAN_PDE_PARAM, GIF_LAST_PARAM]
GIF_RES        = DATASET_RES


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)

    #  --- solution gifs (unchanged by the styling pass): disabled for a plots-only run ---
    #print('\n=== generating FNO gifs (median + last param) ===')
    #gif_fno(FNO_METADATA_FILE, x_limit=X_LIMIT, t_limit=T_LIMIT,
    #         params=GIF_PARAMS, resolution=GIF_RES, outdir=GIF_DIR)

    #print('\n=== generating PINO gifs (with data + no data, median + last param) ===')
    #gif_pino('data', PINO_WITH_DATA_METADATA_FILE, x_limit=X_LIMIT, t_limit=T_LIMIT,
    #          params=GIF_PARAMS, resolution=GIF_RES, outdir=GIF_DIR)
    #gif_pino('no_data', PINO_NO_DATA_METADATA_FILE, x_limit=X_LIMIT, t_limit=T_LIMIT,
    #          params=GIF_PARAMS, resolution=GIF_RES, outdir=GIF_DIR)

    #print('\n=== generating PINN gifs (with data + no data, median param) ===')
    #gif_pinn('data', PINN_WITH_DATA_METADATA_FILE, x_limit=X_LIMIT, t_limit=T_LIMIT,
    #          params=[MEDIAN_PDE_PARAM], resolution=GIF_RES, outdir=GIF_DIR)
    #gif_pinn('no_data', PINN_NO_DATA_METADATA_FILE, x_limit=X_LIMIT, t_limit=T_LIMIT,
    #          params=[MEDIAN_PDE_PARAM], resolution=GIF_RES, outdir=GIF_DIR)

    #print('\n=== plotting soliton profile ===')
    #plot_soliton_profile(outdir=IMG_DIR)

    print('\n=== generating shared dataset ===')
    run_dataset(
        dataset_file=DATASET_FILE,
        device=DEVICE,
        param_values=PARAM_VALUES,
        x_limit=X_LIMIT,
        t_limit=T_LIMIT,
        dataset_res=DATASET_RES,
    )

    print('\n=== training mlp (data only) ===')
    train_mlp(**MLP_CONFIG)

    print('\n=== plotting mlp (data only) ===')
    eval_mlp(
        model_metadata_file=MLP_METADATA_FILE,
        output_dir=MLP_EVAL_DIR,
        **EVAL_CONFIG,
    )

    print('\n=== training fno ===')
    train_fno(**FNO_CONFIG)

    print('\n=== plotting fno ===')
    eval_fno(
        model_metadata_file=FNO_METADATA_FILE,
        output_dir=FNO_EVAL_DIR,
        **EVAL_CONFIG,
    )

    print('\n=== training pino with data ===')
    train_pino('data', **PINO_CONFIG)

    print('\n=== plotting pino with data ===')
    eval_pino(
        'data',
        PINO_WITH_DATA_METADATA_FILE,
        output_dir=PINO_WITH_DATA_EVAL_DIR,
        **EVAL_CONFIG,
    )

    print('\n=== training pino without data ===')
    train_pino('no_data', **PINO_CONFIG)

    print('\n=== plotting pino without data ===')
    eval_pino(
        'no_data',
        PINO_NO_DATA_METADATA_FILE,
        output_dir=PINO_NO_DATA_EVAL_DIR,
        **EVAL_CONFIG,
    )

    print('\n=== training pinn with data ===')
    train_pinn('data', **PINN_CONFIG)

    print('\n=== plotting pinn with data ===')
    eval_pinn(
        'data',
        PINN_WITH_DATA_METADATA_FILE,
        output_dir=PINN_WITH_DATA_EVAL_DIR,
        **EVAL_CONFIG,
    )

    print('\n=== training pinn without data ===')
    train_pinn('no_data', **PINN_CONFIG)

    print('\n=== plotting pinn without data ===')
    eval_pinn(
        'no_data',
        PINN_NO_DATA_METADATA_FILE,
        output_dir=PINN_NO_DATA_EVAL_DIR,
        **EVAL_CONFIG,
    )

    print('\n=== running NTK experiment ===')
    run_ntk_experiment(**PINN_NTK_CONFIG)

    print('\n=== validating NTK spectral-bias theory (controlled toy) ===')
    run_spectral_bias_theory(**SPECTRAL_BIAS_THEORY_CONFIG)

    print('\n=== plotting spectral bias evolution ===')
    plot_spectral_bias_evolution(
        ordered_metadata=[
            ('MLP (data only)',  MLP_METADATA_FILE),
            ('PINN (no data)',   PINN_NO_DATA_METADATA_FILE),
            ('PINN (with data)', PINN_WITH_DATA_METADATA_FILE),
            ('PINO (no data)',   PINO_NO_DATA_METADATA_FILE),
            ('PINO (with data)', PINO_WITH_DATA_METADATA_FILE),
            ('FNO',              FNO_METADATA_FILE),
        ],
        outdir=IMG_DIR,
    )

    print('\nall experiments completed successfully.')


if __name__ == '__main__':
    main()
 
