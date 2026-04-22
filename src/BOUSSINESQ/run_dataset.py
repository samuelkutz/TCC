import os
import numpy as np

from BOUSSINESQ.dataset import generate_dataset
from tools import save_dataset

# Hyperparameters and PDE setup for dataset generation.
# Change these values directly before running the dataset generator.
DEFAULT_DATASET_FILE = os.path.join('RESULTS', 'boussinesq_dataset.pth')
DEFAULT_DEVICE = 'cpu'
DEFAULT_PARAM_VALUES = list(np.arange(0.1, 3.01, 0.5))
DEFAULT_X_LIMIT = 60.0
DEFAULT_T_LIMIT = 15.0
DEFAULT_NX = 64
DEFAULT_NT = 64

# quanto de informação precisa para capturar?

def run_dataset(dataset_file,
                device,
                param_values,
                x_limit,
                t_limit,
                nx,
                nt):
    if any(v is None for v in [dataset_file, device, param_values, x_limit, t_limit, nx, nt]):
        raise ValueError(
            'run_dataset requires explicit values for dataset_file, device, param_values, '
            'x_limit, t_limit, nx, and nt.'
        )

    os.makedirs(os.path.dirname(dataset_file) or '.', exist_ok=True)
    print('Generating dataset with the following settings:')
    print(f'  dataset_file: {dataset_file}')
    print(f'  device: {device}')
    print(f'  param_values: {param_values}')
    print(f'  x_limit: {x_limit}, t_limit: {t_limit}')
    print(f'  nx: {nx}, nt: {nt}')

    x_train, y_train = generate_dataset(
        param_values,
        nx=nx,
        nt=nt,
        x_limit=x_limit,
        t_limit=t_limit,
        device=device,
    )
    save_dataset(x_train, y_train, dataset_file)
    print(f'dataset written to {dataset_file}')


if __name__ == '__main__':
    run_dataset(
        dataset_file=DEFAULT_DATASET_FILE,
        device=DEFAULT_DEVICE,
        param_values=DEFAULT_PARAM_VALUES,
        x_limit=DEFAULT_X_LIMIT,
        t_limit=DEFAULT_T_LIMIT,
        nx=DEFAULT_NX,
        nt=DEFAULT_NT,
    )
