import os
import numpy as np
import torch

from BOUSSINESQ.run_dataset import run_dataset
from FNO.train_fno import train_fno
from FNO.plots_fno import eval_fno
from PINO.train_pino import train_pino
from PINO.plots_pino import eval_pino
from PINN.train_pinn import train_pinn
from PINN.plots_pinn import eval_pinn


# Experiment configuration for the full workflow.
RESULTS_DIR = 'RESULTS'
DATASET_FILE = os.path.join(RESULTS_DIR, 'boussinesq_dataset.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAM_VALUES = list(np.linspace(0.1, 3.0, 10, dtype=np.float32))
X_LIMIT = 60.0
T_LIMIT = 30.0
DATASET_RES = 128

FNO_CONFIG = {
    'dataset_file': DATASET_FILE,
    'epochs': 10000,
    'batch_size': 16,
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
    'epochs': 10000,
    'batch_size': 16,
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
    'train_resolution': 256,
    'param_value': 2.75,
    'epochs': 10000,
    'neurons': 64,
    'hidden_layers': 4,
    'domain_points': 3000,
    'ic_points': 500,
    'optimizer_name': 'Adam',
    'lr': 1e-3,
    'data_weight': 1.0,
    'results_dir': RESULTS_DIR,
}

EVAL_CONFIG = {
    'x_limit': X_LIMIT,
    't_limit': T_LIMIT,
    'eval_params': [0.1, 2.75, 5.75],
    'resolutions': [int(DATASET_RES / 2), DATASET_RES, DATASET_RES * 2],
}
FNO_METADATA_FILE = os.path.join(RESULTS_DIR, 'fno', 'models', 'fno_model_metadata.pth')
PINO_WITH_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'pino', 'with_data', 'models', 'pino_model_metadata.pth')
PINO_NO_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'pino', 'no_data', 'models', 'pino_no_data_model_metadata.pth')
PINN_WITH_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'pinn', 'with_data', 'models', 'pinn_model_metadata.pth')
PINN_NO_DATA_METADATA_FILE = os.path.join(RESULTS_DIR, 'pinn', 'no_data', 'models', 'pinn_no_data_model_metadata.pth')


def main():
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
        **EVAL_CONFIG,
    )

    print('\n=== training pino with data ===')
    train_pino('data', **PINO_CONFIG)
    print('\n=== plotting pino with data ===')
    eval_pino(
        'data',
        PINO_WITH_DATA_METADATA_FILE,
        **EVAL_CONFIG,
    )

    print('\n=== training pino without data ===')
    train_pino('no_data', **PINO_CONFIG)
    print('\n=== plotting pino without data ===')
    eval_pino(
        'no_data',
        PINO_NO_DATA_METADATA_FILE,
        **EVAL_CONFIG,
    )

    print('\n=== training pinn with data ===')
    train_pinn('data', **PINN_CONFIG)
    print('\n=== plotting pinn with data ===')
    eval_pinn(
        'data',
        PINN_WITH_DATA_METADATA_FILE,
        **EVAL_CONFIG,
    )

    print('\n=== training pinn without data ===')
    train_pinn('no_data', **PINN_CONFIG)
    print('\n=== plotting pinn without data ===')
    eval_pinn(
        'no_data',
        PINN_NO_DATA_METADATA_FILE,
        **EVAL_CONFIG,
    )

    print('\nall experiments completed successfully.')


if __name__ == '__main__':
    main()
