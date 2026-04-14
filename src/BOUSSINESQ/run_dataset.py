import os
import numpy as np

from BOUSSINESQ.dataset import generate_dataset
from tools import save_dataset

DEFAULT_DATA_FILE = os.path.join('RESULTS', 'boussinesq_dataset.pth')
DEFAULT_DEVICE = 'cpu'
DEFAULT_PARAM_VALUES = list(np.arange(0.1, 5.01, 0.5))


def run_dataset(dataset_file=DEFAULT_DATA_FILE, device=DEFAULT_DEVICE, param_values=DEFAULT_PARAM_VALUES):
    os.makedirs(os.path.dirname(dataset_file) or '.', exist_ok=True)
    x_train, y_train = generate_dataset(param_values, device=device)
    save_dataset(x_train, y_train, dataset_file)
    print(f'dataset written to {dataset_file}')


if __name__ == '__main__':
    run_dataset()
