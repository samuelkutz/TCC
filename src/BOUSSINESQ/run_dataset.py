import os

from BOUSSINESQ.dataset import generate_dataset, save_dataset

RESULTS_DIR = 'RESULTS'
DATA_FILE = os.path.join(RESULTS_DIR, 'boussinesq_dataset.pth')
PARAM_VALUES = [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.0]

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # generate reference dataset for operator training
    x_train, y_train = generate_dataset(PARAM_VALUES, device='cpu')
    save_dataset(x_train, y_train, DATA_FILE)
    print(f'dataset written to {DATA_FILE}')
