import os

import numpy as np
import torch

from dataset import generate_dataset, save_dataset, load_dataset
from tools import (
    train_model as train_fno_model,
    train_pino_model,
    train_pinn_model,
    save_model,
    load_model,
)
from plots import (
    plot_training_statistics,
    plot_relative_error_panel,
    save_solution_gif,
    compute_spectral_relative_error,
)
from FNO.fno_model import FNO2d, PINO2d
from PINN.PINN import PINN
from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq


RESULTS_DIR = 'RESULTS'
os.makedirs(RESULTS_DIR, exist_ok=True)

# PINN config
RUN_PINN = True
RUN_PINN_WITH_DATA = True
RUN_PINN_NO_DATA = True
PINN_EPOCHS = 5000
PINN_NEURONS = 64
PINN_HIDDEN_LAYERS = 4
PINN_DOMAIN_POINTS = 3000
PINN_IC_POINTS = 500
PINN_OPTIMIZER = 'Adam'
PINN_LR = 1e-3
PINN_DATA_WEIGHT = 1.0
PINN_DATA_WEIGHT_NONE = 0.0
PINN_TRAIN_RESOLUTION = 256
PINN_EVAL_RESOLUTION = 256


# dataset config
LOAD_DATASET = False
SAVE_DATASET = False
Nx_high = 256
Nt_high = 256
nx_fno = 128
nt_fno = 128
param_values = np.arange(0.1, 5.01, 0.5)

# FNO config
RUN_FNO = False
FNO_EPOCHS = 5000
FNO_BATCH_SIZE = 16
FNO_LR = 1e-3
modes1 = 16
modes2 = 16
width = 32

# PINO config
RUN_PINO = False
RUN_PINO_WITH_DATA = False
RUN_PINO_NO_DATA = False
PINO_EPOCHS = 5000
PINO_BATCH_SIZE = 16
PINO_LR = 1e-3
PINO_DX = 0.46875 # dx = 60 / 128 = 0.46875
PINO_DT = 0.1171875 # dt = 15 / 128 = 0.1171875
PINO_PHYS_WEIGHT = 1.0
PINO_IC_WEIGHT = 1.0
PINO_DATA_WEIGHT = 1.0
PINO_DATA_WEIGHT_NONE = 0.0

# experiment config
RESOLUTIONS = [256, 128, 64]
EVAL_PARAMS = [float(param_values[0]), float(param_values[len(param_values) // 2]), float(param_values[-1] + 0.75)]
SNAPSHOT_TIMES = [0.0, 15.0]

DATASET_FILE = os.path.join(
    RESULTS_DIR,
    f"dataset_a{param_values.min():.3f}-{param_values.max():.3f}_Nx{Nx_high}Nt{Nt_high}_nx{nx_fno}nt{nt_fno}_ncases{len(param_values)}.pth",
)


def build_model_input(eta0, u0, a, b, res):
    eta0 = np.asarray(eta0, dtype=np.float32)
    u0 = np.asarray(u0, dtype=np.float32)
    ch0 = np.tile(eta0[:, None], (1, res))
    ch1 = np.tile(u0[:, None], (1, res))
    ch2 = np.ones((res, res), dtype=np.float32) * a
    ch3 = np.ones((res, res), dtype=np.float32) * b
    input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1)
    return torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0)


def evaluate_param_case(model, val, res, device, outdir, label):
    bsq = Boussinesq(-30, 30, 0, 15, val, val)
    solver = PseudoSpectralBoussinesq(bsq, Nx=res, Nt=res - 1, device=device)
    x, t, eta_true, u_true = solver.solve()

    eta_true_t = eta_true.T
    u_true_t = u_true.T
    input_tensor = build_model_input(eta_true[0, :], u_true[0, :], val, val, res).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)
        eta_pred = pred.squeeze().cpu().numpy()[0, :, :]

    rel_error = np.linalg.norm(eta_true_t - eta_pred) / (np.linalg.norm(eta_true_t) + 1e-8)
    spec_error = compute_spectral_relative_error(eta_true_t, eta_pred)

    plot_relative_error_panel(
        x,
        t,
        eta_true_t,
        eta_pred,
        times=[0.0, 15.0],
        outdir=outdir,
        filename=f'{label}_summary_a{val:.3f}_res{res}.png',
        title=f'{label.upper()} Relative Error Summary (a={val:.3f}, res={res})',
    )
    if res == 256:
        save_solution_gif(
            x,
            t,
            eta_true_t,
            eta_pred,
            outdir=outdir,
            filename=f'{label}_animation_a{val:.3f}_res{res}.gif',
            title=f'{label.upper()} Animation (a={val:.3f}, res={res})',
        )

    return rel_error, spec_error


def evaluate_pinn_case(model, val, res, device, outdir, label):
    bsq = Boussinesq(-30, 30, 0, 15, val, val)
    solver = PseudoSpectralBoussinesq(bsq, Nx=res, Nt=res - 1, device=device)
    x, t, eta_true, u_true = solver.solve()

    eta_true_t = eta_true.T
    x_pred, t_pred, eta_pred = predict_pinn_grid(model, res, device)

    rel_error = np.linalg.norm(eta_true_t - eta_pred) / (np.linalg.norm(eta_true_t) + 1e-8)
    spec_error = compute_spectral_relative_error(eta_true_t, eta_pred)
    label_tag = label

    plot_relative_error_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        times=[0.0, 15.0],
        outdir=outdir,
        filename=f'{label_tag}_summary_a{val:.3f}_res{res}.png',
        title=f'{label.upper()} Relative Error Summary (a={val:.3f}, res={res})',
    )
    if res == 256:
        save_solution_gif(
            x_pred,
            t_pred,
            eta_true_t,
            eta_pred,
            outdir=outdir,
            filename=f'{label}_animation_a{val:.3f}_res{res}.gif',
            title=f'{label.upper()} Animation (a={val:.3f}, res={res})',
        )

    return rel_error, spec_error


def predict_pinn_grid(model, res, device):
    x = np.linspace(-30.0, 30.0, res, dtype=np.float32)
    t = np.linspace(0.0, 15.0, res, dtype=np.float32)
    X, T = np.meshgrid(x, t, indexing='xy')
    x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
    t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

    model.eval()
    with torch.no_grad():
        eta_pred, _ = model(x_tensor, t_tensor)

    eta_pred = eta_pred.cpu().numpy().reshape(res, res)
    # convert to x,t orientation for plotting
    return x, t, eta_pred.T


def run_fno_experiments(x_train, y_train, device):
    outdir = os.path.join(RESULTS_DIR, 'fno')
    os.makedirs(outdir, exist_ok=True)

    x_train_cpu = x_train.cpu() if isinstance(x_train, torch.Tensor) else x_train
    y_train_cpu = y_train.cpu() if isinstance(y_train, torch.Tensor) else y_train

    model, history = train_fno_model(
        x_train_cpu,
        y_train_cpu,
        epochs=FNO_EPOCHS,
        batch_size=FNO_BATCH_SIZE,
        modes1=modes1,
        modes2=modes2,
        width=width,
        lr=FNO_LR,
        device=device,
    )
    filename = save_model(
        model,
        model_name='fno',
        epochs=FNO_EPOCHS,
        n_samples=len(param_values),
        modes=(modes1, modes2),
        width=width,
        extra='fno',
    )

    plot_training_statistics(
        [history],
        ['FNO'],
        outdir=outdir,
        filename='fno_training_statistics.png',
    )

    print('\nEvaluating FNO on selected parameters...')
    fno_metrics = []
    middle_val = EVAL_PARAMS[len(EVAL_PARAMS) // 2]
    for val in EVAL_PARAMS:
        resolutions = RESOLUTIONS if val == middle_val else [256]
        for res in resolutions:
            rel_error, spec_error = evaluate_param_case(
                model,
                val,
                res,
                device,
                outdir,
                label='fno',
            )
            fno_metrics.append((val, res, rel_error, spec_error))
    return history, fno_metrics


def run_pino_experiments(x_train, y_train, device, data_weight, label_suffix):
    outdir = os.path.join(RESULTS_DIR, 'pino', label_suffix)
    os.makedirs(outdir, exist_ok=True)

    x_train_cpu = x_train.cpu() if isinstance(x_train, torch.Tensor) else x_train
    y_train_cpu = y_train.cpu() if isinstance(y_train, torch.Tensor) else y_train

    model, history = train_pino_model(
        x_train_cpu,
        y_train_cpu,
        epochs=PINO_EPOCHS,
        batch_size=PINO_BATCH_SIZE,
        modes1=modes1,
        modes2=modes2,
        width=width,
        dx=PINO_DX,
        dt=PINO_DT,
        phys_weight=PINO_PHYS_WEIGHT,
        ic_weight=PINO_IC_WEIGHT,
        data_weight=data_weight,
        lr=PINO_LR,
        device=device,
    )
    filename = save_model(
        model,
        model_name=f'pino_{label_suffix}',
        epochs=PINO_EPOCHS,
        n_samples=len(param_values),
        modes=(modes1, modes2),
        width=width,
        extra=label_suffix,
    )

    plot_training_statistics(
        [history],
        [label_suffix],
        outdir=outdir,
        filename=f'pino_{label_suffix}_training_statistics.png',
    )

    print(f'\nEvaluating PINO ({label_suffix}) on selected parameters...')
    pino_metrics = []
    middle_val = EVAL_PARAMS[len(EVAL_PARAMS) // 2]
    for val in EVAL_PARAMS:
        resolutions = RESOLUTIONS if val == middle_val else [256]
        for res in resolutions:
            rel_error, spec_error = evaluate_param_case(
                model,
                val,
                res,
                device,
                outdir,
                label=f'pino_{label_suffix}',
            )
            pino_metrics.append((val, res, rel_error, spec_error))
    return history, pino_metrics


def _load_pinn_model(model_path, param_value, device):
    bsq = Boussinesq(-30, 30, 0, 15, param_value, param_value)
    model = PINN(
        input_size=2,
        output_size=2,
        neurons=PINN_NEURONS,
        hidden_layers=PINN_HIDDEN_LAYERS,
        Boussinesq=bsq,
        domain_points=PINN_DOMAIN_POINTS,
        ic_points=PINN_IC_POINTS,
        optimizer_name=PINN_OPTIMIZER,
        lr=PINN_LR,
        data=None,
        data_weight=0.0,
        device=device,
    )
    load_model(model_path, model, device=device)
    return model


def run_pinn_experiments(device, data_weight, label_suffix):
    outdir = os.path.join(RESULTS_DIR, 'pinn', label_suffix)
    os.makedirs(outdir, exist_ok=True)

    param_value = float(param_values[len(param_values) // 2])
    model, history = train_pinn_model(
        param_value,
        epochs=PINN_EPOCHS,
        neurons=PINN_NEURONS,
        hidden_layers=PINN_HIDDEN_LAYERS,
        domain_points=PINN_DOMAIN_POINTS,
        ic_points=PINN_IC_POINTS,
        optimizer_name=PINN_OPTIMIZER,
        lr=PINN_LR,
        data_weight=data_weight,
        train_resolution=PINN_TRAIN_RESOLUTION,
        device=device,
    )
    filename = save_model(
        model,
        model_name=f'pinn_{label_suffix}',
        epochs=PINN_EPOCHS,
        n_samples=1,
        modes=(modes1, modes2),
        width=width,
        extra=f'pinn_{label_suffix}_a{param_value:.3f}',
    )

    plot_training_statistics(
        [history],
        [label_suffix],
        outdir=outdir,
        filename=f'pinn_{label_suffix}_training_statistics.png',
    )

    print(f'\nEvaluating PINN ({label_suffix}) on selected parameters and resolutions...')
    pinn_metrics = []
    rel_error, spec_error = evaluate_pinn_case(
        model,
        param_value,
        PINN_EVAL_RESOLUTION,
        device,
        outdir,
        label=f'pinn_{label_suffix}',
    )
    pinn_metrics.append((param_value, PINN_EVAL_RESOLUTION, rel_error, spec_error))

    return history, pinn_metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    if LOAD_DATASET and os.path.exists(DATASET_FILE):
        x_train, y_train = load_dataset(DATASET_FILE)
        print(f'Loaded dataset from {DATASET_FILE}')
    else:
        print('Generating dataset...')
        x_train, y_train = generate_dataset(
            param_values,
            Nx_high=Nx_high,
            Nt_high=Nt_high,
            nx_fno=nx_fno,
            nt_fno=nt_fno,
            device=device,
        )
        save_dataset(x_train, y_train, DATASET_FILE)
        print(f'Dataset saved to {DATASET_FILE}')

    print(f'Dataset ready: x_train={x_train.shape}, y_train={y_train.shape}\n')

    fno_metrics = []
    if RUN_FNO:
        print('Running FNO experiments...')
        fno_history, fno_metrics = run_fno_experiments(x_train, y_train, device)
    else:
        print('Skipping FNO experiments.')

    pino_metrics = []
    if RUN_PINO:
        if RUN_PINO_WITH_DATA:
            print('\nRunning PINO experiments with data...')
            pino_history_with, pino_metrics_with = run_pino_experiments(
                x_train,
                y_train,
                device,
                data_weight=PINO_DATA_WEIGHT,
                label_suffix='with_data',
            )
            pino_metrics.append(('with_data', pino_metrics_with))

        if RUN_PINO_NO_DATA:
            print('\nRunning PINO experiments without data...')
            pino_history_no, pino_metrics_no = run_pino_experiments(
                x_train,
                y_train,
                device,
                data_weight=PINO_DATA_WEIGHT_NONE,
                label_suffix='no_data',
            )
            pino_metrics.append(('no_data', pino_metrics_no))

    pinn_metrics = []
    if RUN_PINN:
        if RUN_PINN_WITH_DATA:
            print('\nRunning PINN experiments with data...')
            pinn_history_with, pinn_metrics_with = run_pinn_experiments(
                device,
                data_weight=PINN_DATA_WEIGHT,
                label_suffix='with_data',
            )
            pinn_metrics.append(('with_data', pinn_metrics_with))

        if RUN_PINN_NO_DATA:
            print('\nRunning PINN experiments without data...')
            pinn_history_no, pinn_metrics_no = run_pinn_experiments(
                device,
                data_weight=PINN_DATA_WEIGHT_NONE,
                label_suffix='no_data',
            )
            pinn_metrics.append(('no_data', pinn_metrics_no))
    else:
        print('Skipping PINN experiments.')

    print('\nExperiments completed. Results saved under:', RESULTS_DIR)
    if RUN_FNO:
        print('\nFNO metrics summary:')
        for item in fno_metrics[:6]:
            print(item)

    if RUN_PINO:
        for label, metrics in pino_metrics:
            print(f'\nPINO metrics summary ({label}):')
            for item in metrics[:6]:
                print(item)

    if RUN_PINN:
        print('\nPINN metrics summary:')
        for item in pinn_metrics[:6]:
            print(item)


if __name__ == '__main__':
    main()
