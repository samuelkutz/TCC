import os
import torch
from timeit import default_timer
from torch.optim import Adam

from tools import RelativeL2_loss, save_model, load_dataset
from FNO.FNO import FNO2d

RESULTS_DIR = 'RESULTS'


def train_fno(dataset_file='RESULTS/boussinesq_dataset.pth',
              epochs=5000,
              batch_size=16,
              lr=1e-3,
              modes1=16,
              modes2=16,
              width=32,
              print_interval=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outdir = os.path.join(RESULTS_DIR, 'fno')
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

    model = FNO2d(modes1=modes1, modes2=modes2, width=width).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {num_params:,}')
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = RelativeL2_loss()

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
            print(f'epoch {epoch + 1}, elapsed {elapsed:.1f}s, relative l2 loss {epoch_loss:.4e}')

    training_duration = default_timer() - start_time
    final_loss = train_history[-1] if train_history else None

    model_file = os.path.join(model_dir, 'fno_weights.pth')
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
            'modes1': modes1,
            'modes2': modes2,
            'width': width,
            'dataset_file': dataset_file,
        },
        'model_file': model_file,
        'dataset_file': dataset_file,
    }
    torch.save(model_metadata, os.path.join(model_dir, 'fno_model_metadata.pth'))
    print(f'fno model metadata saved to {os.path.join(model_dir, "fno_model_metadata.pth")}')


if __name__ == '__main__':
    train_fno()
