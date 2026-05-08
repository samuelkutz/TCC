import os
import random
import numpy as np
import torch

from BOUSSINESQ.run_dataset import run_dataset
from FNO.train_fno import train_fno
from FNO.plots_fno import eval_fno
from PINO.train_pino import train_pino
from PINO.plots_pino import eval_pino
from PINN.train_pinn import train_pinn
from PINN.plots_pinn import eval_pinn


# experiment configuration for the full workflow.
RESULTS_DIR = 'results'
DATASET_FILE = os.path.join(RESULTS_DIR, 'models', 'boussinesq_dataset.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAM_VALUES = list(np.linspace(0.1, 4.0, 20, dtype=np.float32))
X_LIMIT = 60.0
T_LIMIT = 30.0
DATASET_RES = 128
EVAL_PARAMS = [0.1, 3.21, 4.2]
MEDIAN_PDE_PARAM = sorted(EVAL_PARAMS)[len(EVAL_PARAMS) // 2]
EPOCHS = 5000
SEED = 37

FNO_CONFIG = {
    'dataset_file': DATASET_FILE,
    'epochs': EPOCHS,
    'batch_size': 256,
    'lr': 1e-3,
    'modes1': 16,
    'modes2': 16,
    'width': 32,
    'print_interval': 500,
    'results_dir': RESULTS_DIR,
}

PINO_CONFIG = {
    'dataset_file': DATASET_FILE,
    'x_limit': X_LIMIT,
    't_limit': T_LIMIT,
    'epochs': EPOCHS,
    'batch_size': 256,
    'lr': 1e-3,
    'phys_weight': 1.0,
    'ic_weight': 1.0,
    'data_weight': 1.0,
    'modes1': 16,
    'modes2': 16,
    'width': 32,
    'print_interval': 500,
    'results_dir': RESULTS_DIR,
}


PINN_CONFIG = {
    'x_limit': X_LIMIT,
    't_limit': T_LIMIT,
    'train_resolution': DATASET_RES,
    'param_value': MEDIAN_PDE_PARAM,
    'epochs': EPOCHS,
    'neurons': 256,
    'hidden_layers': 4,
    'domain_points': 8000,
    'ic_points': 500,
    'optimizer_name': 'Adam',
    'lr': 1e-3,
    'data_weight': 1.0,
    'print_interval': 500,
    'results_dir': RESULTS_DIR,
}

EVAL_CONFIG = {
    'x_limit': X_LIMIT,
    't_limit': T_LIMIT,
    'eval_params': EVAL_PARAMS,
    'resolutions': [DATASET_RES, DATASET_RES * 2, DATASET_RES * 4],
    'spectral_res': 256,
}

IMG_DIR = os.path.join(RESULTS_DIR, 'imgs')

FNO_EVAL_DIR = os.path.join(IMG_DIR, 'fno')
PINO_WITH_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pino', 'with_data')
PINO_NO_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pino', 'no_data')
PINN_WITH_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pinn', 'with_data')
PINN_NO_DATA_EVAL_DIR = os.path.join(IMG_DIR, 'pinn', 'no_data')

FNO_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'fno', 'fno_model_metadata.pth')
PINO_WITH_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pino', 'with_data', 'pino_model_metadata.pth')
PINO_NO_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pino', 'no_data', 'pino_no_data_model_metadata.pth')
PINN_WITH_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pinn', 'with_data', 'pinn_model_metadata.pth')
PINN_NO_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'models', 'metadata', 'pinn', 'no_data', 'pinn_no_data_model_metadata.pth')


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)

    print('\n=== generating shared dataset ===')
    run_dataset(
        dataset_file=DATASET_FILE,
        device=DEVICE,
        param_values=PARAM_VALUES,
        x_limit=X_LIMIT,
        t_limit=T_LIMIT,
        dataset_res=DATASET_RES,
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

    print('\nall experiments completed successfully.')


if __name__ == '__main__':
    main()
