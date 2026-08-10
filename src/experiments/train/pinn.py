"""Model 2 of the ladder: the same backbone as the MLP plus the Boussinesq residual.

`mode` is the single toggle between the two regimes reported in Chapter 4:
'data_and_physics' keeps the supervised term alongside the physics,
'pure_physics' zeroes its weight so the network is driven by the equation and
the initial condition alone.
"""

import os

from methods.pinn import PINN
from tools import save_model, set_seed
from experiments.common import (
    BandTracker, StageTimer, boussinesq_at, reference_solution, resolve_device,
    stage_paths, write_stage_metadata,
)


def _label(mode):
    # the regime is carried by the directory (stage_paths), so the file label is
    # just the model name in both regimes
    return 'pinn'


def train_pinn(mode, x_limit, t_limit, train_resolution, param_value, epochs, neurons,
               hidden_layers, domain_points, ic_points, optimizer_name, lr, data_weight,
               print_interval, results_dir, amplitude=1.0, seed=None):
    label = _label(mode)
    data_weight = data_weight if mode == 'data_and_physics' else 0.0

    # per-stage reseed for a reproducible standalone run; see train_mlp
    if seed is not None:
        set_seed(seed)

    device = resolve_device()
    weights_dir, metadata_dir = stage_paths(results_dir, 'pinn', mode)

    bsq = boussinesq_at(param_value, x_limit, t_limit, amplitude)
    # nt=res-1 steps give exactly res time levels, the convention of Table 3.1
    x_sol, t_sol, eta_sol, u_sol = reference_solution(
        param_value, x_limit, t_limit, nx=train_resolution, nt=train_resolution - 1,
        device=device, amplitude=amplitude)
    data = {'x': x_sol, 't': t_sol, 'eta': eta_sol, 'u': u_sol} if data_weight > 0 else None

    bands = BandTracker(eta_sol.T)

    def snapshot(model, epoch):
        bands.record(epoch, model.predict_eta_grid(x_sol, t_sol))
        model.train()

    model = PINN(
        input_size=2,
        output_size=2,
        neurons=neurons,
        hidden_layers=hidden_layers,
        Boussinesq=bsq,
        domain_points=domain_points,
        ic_points=ic_points,
        optimizer_name=optimizer_name,
        lr=lr,
        data=data,
        data_weight=data_weight,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  {label}: {num_params:,} parameters on {device}')

    timer = StageTimer()
    history = model.run_train_loop(epochs=epochs, print_interval=print_interval,
                                   snapshot_callback=snapshot)
    training_duration = timer.elapsed

    model_file = save_model(model, os.path.join(weights_dir, f'{label}_weights.pth'))

    write_stage_metadata({
        'mode': mode,
        'seed': seed,
        'train_history': history,
        'spectral_history': bands.as_dict(),
        'training_duration': training_duration,
        'final_loss': history[-1] if history else None,
        'num_params': num_params,
        'params': {
            'epochs': epochs,
            'neurons': neurons,
            'hidden_layers': hidden_layers,
            'domain_points': domain_points,
            'ic_points': ic_points,
            'optimizer_name': optimizer_name,
            'lr': lr,
            'data_weight': data_weight,
            'train_resolution': train_resolution,
            'param_value': param_value,
            'amplitude': amplitude,
            'x_limit': x_limit,
            't_limit': t_limit,
        },
        'model_file': model_file,
    }, metadata_dir, label, 'PINN')
