import os
import torch
from timeit import default_timer
from torch.optim import Adam

from tools import (
    L2_loss, compute_norm_stats, load_dataset, normalize_dataset, save_model,
    normalize_tensor, unnormalize_tensor, compute_spectral_band_errors, save_metadata_json,
)
from FNO.FNO import FNO2d


def train_fno(dataset_file, epochs, batch_size, lr, modes1, modes2, width, print_interval, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, 'models', 'weights', 'fno')
    metadata_dir = os.path.join(results_dir, 'models', 'metadata', 'fno')
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

    # store raw reference sample before normalization for spectral snapshots
    ref_x_raw = x_train[0:1].clone().cpu()   # (1, C_in, Nx, Nt)
    eta_true_ref = y_train[0, 0].numpy().astype('float64')  # (Nx, Nt), eta channel

    x_train, y_train = normalize_dataset(x_train, y_train, norm_stats)
    spectral_history = {'epochs': [], 'low_band': [], 'mid_band': [], 'high_band': []}

    model = FNO2d(modes1=modes1, modes2=modes2, width=width).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'model parameter count: {num_params:,}')
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = L2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_history = []
    start_time = default_timer()
    print('starting fno training...')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_history.append(epoch_loss)

        if (epoch + 1) % print_interval == 0 or epoch == epochs - 1:
            elapsed = default_timer() - start_time
            print(f'epoch {epoch + 1}, elapsed {elapsed:.1f}s, total_loss {epoch_loss:.4e}')
            model.eval()
            with torch.no_grad():
                ref_norm = normalize_tensor(ref_x_raw, norm_stats['input_min'], norm_stats['input_max'], norm_stats['eps'])
                pred_norm = model(ref_norm.to(device))
                pred = unnormalize_tensor(pred_norm.cpu(), norm_stats['output_min'], norm_stats['output_max'], norm_stats['eps'])
            model.train()
            eta_pred = pred[0, 0].numpy().astype('float64')  # (Nx, Nt)
            low, mid, high = compute_spectral_band_errors(eta_pred, eta_true_ref)
            spectral_history['epochs'].append(epoch + 1)
            spectral_history['low_band'].append(low)
            spectral_history['mid_band'].append(mid)
            spectral_history['high_band'].append(high)

    training_duration = default_timer() - start_time
    final_loss = train_history[-1] if train_history else None

    model_file = os.path.join(weights_dir, 'fno_weights.pth')
    save_model(model, filepath=model_file)

    model_metadata = {
        'train_history': train_history,
        'spectral_history': spectral_history,
        'training_duration': training_duration,
        'final_loss': final_loss,
        'num_params': num_params,
        'params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'modes1': modes1,
            'modes2': modes2,
            'width': width,
            'dataset_file': dataset_file,
        },
        'model_file': model_file,
        'dataset_file': dataset_file,
        'norm_stats': norm_stats,
    }
    model_metadata_file = os.path.join(metadata_dir, 'fno_model_metadata.pth')
    torch.save(model_metadata, model_metadata_file)
    print(f'fno model metadata saved to {model_metadata_file}')

    json_payload = {k: v for k, v in model_metadata.items() if k != 'norm_stats'}
    json_payload['model'] = 'FNO'
    save_metadata_json(json_payload, os.path.join(metadata_dir, 'fno_metadata.json'))
