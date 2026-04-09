import argparse
import os
import numpy as np
import torch
from timeit import default_timer
from torch.optim import Adam

from BOUSSINESQ.dataset import load_dataset
from FNO.FNO import FNO2d
from tools import RelativeL2_loss, save_model

RESULTS_DIR = 'RESULTS'
DATA_FILE = os.path.join(RESULTS_DIR, 'boussinesq_dataset.pth')

FNO_EPOCHS = 5000
FNO_BATCH_SIZE = 16
FNO_LR = 1e-3
MODES1 = 16
MODES2 = 16
WIDTH = 32
PRINT_INTERVAL = 500
PARAM_VALUES = np.arange(0.1, 5.01, 0.5)

if __name__ == '__main__':
    # parse dataset file path for fno training
    parser = argparse.ArgumentParser(description='train fno model and save artifacts')
    parser.add_argument('--dataset-file', default=DATA_FILE)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outdir = os.path.join(RESULTS_DIR, 'fno')
    model_dir = os.path.join(outdir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    dataset_file = args.dataset_file
    if os.path.exists(dataset_file):
        x_train, y_train = load_dataset(dataset_file)
        print(f'loaded dataset from {dataset_file}')
    else:
        raise RuntimeError(
            f'Dataset not found at {dataset_file}. '
            'Generate it first using `python src/run_dataset.py`.'
        )

    # build fno model and use relative l2 loss for normalized predictions
    model = FNO2d(modes1=MODES1, modes2=MODES2, width=WIDTH).to(device)
    optimizer = Adam(model.parameters(), lr=FNO_LR)
    loss_fn = RelativeL2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=FNO_BATCH_SIZE, shuffle=True)

    train_history = []
    start_time = default_timer()
    print('starting fno training...')
    for epoch in range(FNO_EPOCHS):
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

        if (epoch + 1) % PRINT_INTERVAL == 0 or epoch == FNO_EPOCHS - 1:
            elapsed = default_timer() - start_time
            print(f'epoch {epoch + 1}, elapsed {elapsed:.1f}s, relative l2 loss {epoch_loss:.4e}')

    # save trained fno weights and training artifacts for later evaluation
    model_file = os.path.join(model_dir, 'fno_weights.pth')
    save_model(model, filepath=model_file)

    artifacts = {
        'train_history': train_history,
        'params': {
            'epochs': FNO_EPOCHS,
            'batch_size': FNO_BATCH_SIZE,
            'lr': FNO_LR,
            'modes1': MODES1,
            'modes2': MODES2,
            'width': WIDTH,
            'dataset_file': dataset_file,
        },
        'model_file': model_file,
        'dataset_file': dataset_file,
    }
    torch.save(artifacts, os.path.join(model_dir, 'fno_artifacts.pth'))
    print(f'fno artifacts saved to {os.path.join(model_dir, "fno_artifacts.pth")}')
