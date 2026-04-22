import os
import torch
from timeit import default_timer
from torch.optim import Adam

from tools import RelativeL2_loss, save_model, load_dataset
from PINO.PINO import PINO2d, pino_loss

RESULTS_DIR = 'RESULTS'


def train_pino(mode='data',
               dataset_file='RESULTS/boussinesq_dataset.pth',
               x_limit=60.0,
               t_limit=15.0,
               epochs=5000,
               batch_size=16,
               lr=1e-3,
               phys_weight=1.0,
               ic_weight=1.0,
               data_weight=1.0,
               modes1=16,
               modes2=16,
               width=32,
               print_interval=500):
    data_weight = data_weight if mode == 'data' else 0.0
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

    nx = x_train.shape[2]
    nt = x_train.shape[3]
    dx = 2.0 * x_limit / nx
    dt = t_limit / (nt - 1)
    print(f'computed physics spacing dx={dx:.6f}, dt={dt:.6f} for x_limit={x_limit}, t_limit={t_limit}')

    model = PINO2d(modes1=modes1, modes2=modes2, width=width, out_channels=y_train.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {num_params:,}')
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = RelativeL2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_history = []
    start_time = default_timer()
    print(f'starting pino training ({mode})...')
    for epoch in range(epochs):
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
                dx=dx,
                dt=dt,
                phys_weight=phys_weight,
                ic_weight=ic_weight,
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

        if (epoch + 1) % print_interval == 0 or epoch == epochs - 1:
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
        'num_params': num_params,
        'params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'x_limit': x_limit,
            't_limit': t_limit,
            'dx': dx,
            'dt': dt,
            'phys_weight': phys_weight,
            'ic_weight': ic_weight,
            'data_weight': data_weight,
            'modes1': modes1,
            'modes2': modes2,
            'width': width,
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
