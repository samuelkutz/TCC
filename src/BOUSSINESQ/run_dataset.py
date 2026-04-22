import os
import numpy as np

from BOUSSINESQ.dataset import generate_dataset
from tools import save_dataset

# Hyperparameters and PDE setup for dataset generation.
# Change these values directly before running the dataset generator.
DATASET_FILE = os.path.join('RESULTS', 'boussinesq_dataset.pth')
DEVICE = 'cpu'
PARAM_VALUES = list(np.arange(0.1, 3.01, 0.5))
X_LIMIT = 60.0
T_LIMIT = 15.0
NX_HIGH = 256 # aumentar em comparação a x limite
NT_HIGH = 256
NX_FNO = 64
NT_FNO = 64

# quanto de informação precisa para capturar? 

def run_dataset(dataset_file=DATASET_FILE,
                device=DEVICE,
                param_values=PARAM_VALUES,
                x_limit=X_LIMIT,
                t_limit=T_LIMIT,
                Nx_high=NX_HIGH,
                Nt_high=NT_HIGH,
                nx_fno=NX_FNO,
                nt_fno=NT_FNO):
    os.makedirs(os.path.dirname(dataset_file) or '.', exist_ok=True)
    print('Generating dataset with the following settings:')
    print(f'  dataset_file: {dataset_file}')
    print(f'  device: {device}')
    print(f'  param_values: {param_values}')
    print(f'  x_limit: {x_limit}, t_limit: {t_limit}')
    print(f'  Nx_high: {Nx_high}, Nt_high: {Nt_high}')
    print(f'  nx_fno: {nx_fno}, nt_fno: {nt_fno}')

    x_train, y_train = generate_dataset(
        param_values,
        Nx_high=Nx_high,
        Nt_high=Nt_high,
        nx_fno=nx_fno,
        nt_fno=nt_fno,
        x_limit=x_limit,
        t_limit=t_limit,
        device=device,
    )
    save_dataset(x_train, y_train, dataset_file)
    print(f'dataset written to {dataset_file}')


if __name__ == '__main__':
    run_dataset()
