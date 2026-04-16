import os
import numpy as np
import torch
from timeit import default_timer
from torch.optim import Adam

from tools import RelativeL2_loss, save_model, load_dataset
from PINO.PINO import PINO2d, pino_loss

RESULTS_DIR = 'RESULTS'
DATA_FILE = os.path.join(RESULTS_DIR, 'boussinesq_dataset.pth')

PINO_EPOCHS = 5000
PINO_BATCH_SIZE = 16
PINO_LR = 1e-3
PINO_DX = 0.46875
PINO_DT = 0.1171875
PINO_PHYS_WEIGHT = 1.0
PINO_IC_WEIGHT = 1.0
PINO_DATA_WEIGHT = 1.0
MODES1 = 16
MODES2 = 16
WIDTH = 32
PRINT_INTERVAL = 500
PARAM_VALUES = np.arange(0.1, 5.01, 0.5)


def train_pino(mode='data', dataset_file=DATA_FILE):
    data_weight = PINO_DATA_WEIGHT if mode == 'data' else 0.0
    label = 'pino' if mode == 'data' else 'pino_no_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outdir = os.path.join(RESULTS_DIR, 'pino', 'with_data' if mode == 'data' else 'no_data')
    model_dir = os.path.join(outdir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(dataset_file):
        x_train, y_train = load_dataset(dataset_file)
        print(f'loaded dataset from {dataset_file}')
    else:
        raise RuntimeError(
            f'Dataset not found at {dataset_file}. '
            'Generate it first using `python src/BOUSSINESQ/run_dataset.py`.'
        )

    model = PINO2d(modes1=MODES1, modes2=MODES2, width=WIDTH, out_channels=y_train.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr=PINO_LR)
    loss_fn = RelativeL2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=PINO_BATCH_SIZE, shuffle=True)

    train_history = []
    start_time = default_timer()
    print(f'starting pino training ({mode})...')
    for epoch in range(PINO_EPOCHS):
        model.train()
        epoch_rel = 0.0

        epoch_data = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_data = pino_loss(
                model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                dx=PINO_DX,
                dt=PINO_DT,
                phys_weight=PINO_PHYS_WEIGHT,
                ic_weight=PINO_IC_WEIGHT,
                data_weight=data_weight,
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = model(batch_x)
                rel = loss_fn(pred, batch_y).item()
            epoch_rel += rel
            epoch_data += loss_data.item()

        epoch_rel /= len(train_loader)
        epoch_data /= len(train_loader)
        train_history.append(epoch_rel)

        if (epoch + 1) % PRINT_INTERVAL == 0 or epoch == PINO_EPOCHS - 1:
            elapsed = default_timer() - start_time
            print(
                f'epoch {epoch + 1}, elapsed {elapsed:.1f}s, '
                f'relative l2 loss {epoch_rel:.4e}, data_loss {epoch_data:.4e}'
            )

    training_duration = default_timer() - start_time
    final_loss = train_history[-1] if train_history else None

    model_file = os.path.join(model_dir, f'{label}_weights.pth')
    save_model(model, filepath=model_file)

    model_metadata = {
        'train_history': train_history,
        'training_duration': training_duration,
        'final_loss': final_loss,
        'params': {
            'epochs': PINO_EPOCHS,
            'batch_size': PINO_BATCH_SIZE,
            'lr': PINO_LR,
            'dx': PINO_DX,
            'dt': PINO_DT,
            'phys_weight': PINO_PHYS_WEIGHT,
            'ic_weight': PINO_IC_WEIGHT,
            'data_weight': data_weight,
            'modes1': MODES1,
            'modes2': MODES2,
            'width': WIDTH,
            'out_channels': y_train.shape[1],
            'dataset_file': dataset_file,
        },
        'model_file': model_file,
        'dataset_file': dataset_file,
        'mode': mode,
    }
    model_metadata_file = os.path.join(model_dir, f'{label}_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'pino model metadata saved to {model_metadata_file}')


def train_pino_data():
    return train_pino('data')


def train_pino_no_data():
    return train_pino('no_data')


if __name__ == '__main__':
    train_pino()
