import os
import numpy as np

from BOUSSINESQ.dataset import generate_dataset
from tools import compute_norm_stats, save_dataset

# hyperparameters and pde setup for dataset generation.
# change these values directly before running the dataset generator.
DEFAULT_DATASET_FILE = os.path.join('results', 'models', 'boussinesq_dataset.pth')
DEFAULT_DEVICE = 'cpu'
DEFAULT_PARAM_VALUES = list(np.arange(0.1, 3.01, 0.5))
DEFAULT_X_LIMIT = 60.0
DEFAULT_T_LIMIT = 15.0
DEFAULT_DATASET_RES = 64

# quanto de informação precisa para capturar?

def run_dataset(dataset_file,
                device,
                param_values,
                x_limit,
                t_limit,
                dataset_res):
    if any(v is None for v in [dataset_file, device, param_values, x_limit, t_limit, dataset_res]):
        raise ValueError(
            'run_dataset requires explicit values for dataset_file, device, param_values, '
            'x_limit, t_limit, and dataset_res.'
        )
    os.makedirs(os.path.dirname(dataset_file) or '.', exist_ok=True)
    print('Generating dataset with the following settings:')
    print(f'  dataset_file: {dataset_file}')
    print(f'  device: {device}')
    print(f'  param_values: {param_values}')
    print(f'  x_limit: {x_limit}, t_limit: {t_limit}')
    print(f'  dataset_res: {dataset_res}')

    x_train, y_train = generate_dataset(
        param_values,
        nx=dataset_res,
        nt=dataset_res,
        x_limit=x_limit,
        t_limit=t_limit,
        device=device,
    )
    norm_stats = compute_norm_stats(x_train, y_train)
    save_dataset(x_train, y_train, dataset_file, norm_stats=norm_stats)
    print(f'dataset written to {dataset_file}')


if __name__ == '__main__':
    run_dataset(
        dataset_file=DEFAULT_DATASET_FILE,
        device=DEFAULT_DEVICE,
        param_values=DEFAULT_PARAM_VALUES,
        x_limit=DEFAULT_X_LIMIT,
        t_limit=DEFAULT_T_LIMIT,
        dataset_res=DEFAULT_DATASET_RES,
    )