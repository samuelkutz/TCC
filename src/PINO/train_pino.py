import os
import torch
from timeit import default_timer
from torch.optim import Adam

from tools import compute_norm_stats, load_dataset, normalize_dataset, save_model
from PINO.PINO import PINO2d, pino_loss


def train_pino(mode, dataset_file, x_limit, t_limit, epochs, batch_size, lr, phys_weight, ic_weight, data_weight, modes1, modes2, width, print_interval, results_dir):
    data_weight = data_weight if mode == 'data' else 0.0
    label = 'pino' if mode == 'data' else 'pino_no_data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, 'models', 'weights', 'pino', 'with_data' if mode == 'data' else 'no_data')
    metadata_dir = os.path.join(results_dir, 'models', 'metadata', 'pino', 'with_data' if mode == 'data' else 'no_data')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    if os.path.exists(dataset_file):
        x_train, y_train, norm_stats = load_dataset(dataset_file)
        print(f'loaded dataset from {dataset_file}')
    else:
        raise RuntimeError(
            f'Dataset not found at {dataset_file}. '
            'Generate it first using `python src/BOUSSINESQ/run_dataset.py`.'
        )

    if norm_stats is None:
        norm_stats = compute_norm_stats(x_train, y_train)
    x_train, y_train = normalize_dataset(x_train, y_train, norm_stats)
    norm_stats = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in norm_stats.items()
    }

    nx = x_train.shape[2]
    nt = x_train.shape[3]
    dx = 2.0 * x_limit / nx
    dt = t_limit / (nt - 1)
    print(f'Computed physics spacing dx={dx:.6f}, dt={dt:.6f} for x_limit={x_limit}, t_limit={t_limit}')

    model = PINO2d(modes1=modes1, modes2=modes2, width=width, out_channels=y_train.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {num_params:,}')
    optimizer = Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_history = []
    start_time = default_timer()
    print(f'starting pino training ({mode})...')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pde = 0.0
        epoch_ic = 0.0
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
                norm_stats=norm_stats,
                phys_weight=phys_weight,
                ic_weight=ic_weight,
                data_weight=data_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_data += loss_data.item()
            epoch_loss += loss.item()
            epoch_pde += loss_pde.item()
            epoch_ic += loss_ic.item()

        epoch_data /= len(train_loader)
        epoch_loss /= len(train_loader)
        epoch_pde /= len(train_loader)
        epoch_ic /= len(train_loader)
        train_history.append(epoch_loss)

        if (epoch + 1) % print_interval == 0 or epoch == epochs - 1:
            elapsed = default_timer() - start_time
            print(
                f'epoch {epoch + 1}, elapsed {elapsed:.1f}s, '
                f'total_loss {epoch_loss:.4e}, '
                f'pde_loss {epoch_pde:.4e}, '
                f'ic_loss {epoch_ic:.4e}, '
                f'data_loss {epoch_data:.4e}'
            )

    training_duration = default_timer() - start_time
    final_loss = train_history[-1] if train_history else None

    model_file = os.path.join(weights_dir, f'{label}_weights.pth')
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
        'norm_stats': norm_stats,
    }
    model_metadata_file = os.path.join(metadata_dir, f'{label}_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'pino model metadata saved to {model_metadata_file}')


def train_pino_data(dataset_file, x_limit, t_limit, epochs, batch_size, lr, phys_weight, ic_weight, data_weight, modes1, modes2, width, print_interval, results_dir):
    return train_pino(
        'data',
        dataset_file,
        x_limit,
        t_limit,
        epochs,
        batch_size,
        lr,
        phys_weight,
        ic_weight,
        data_weight,
        modes1,
        modes2,
        width,
        print_interval,
        results_dir,
    )


def train_pino_no_data(dataset_file, x_limit, t_limit, epochs, batch_size, lr, phys_weight, ic_weight, data_weight, modes1, modes2, width, print_interval, results_dir):
    return train_pino(
        'no_data',
        dataset_file,
        x_limit,
        t_limit,
        epochs,
        batch_size,
        lr,
        phys_weight,
        ic_weight,
        data_weight,
        modes1,
        modes2,
        width,
        print_interval,
        results_dir,
    )


