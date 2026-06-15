import os
import numpy as np
import torch

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from PINN.PINN import PINN
from tools import load_model
from _plots import (
    plot_training_statistics,
    plot_model2_resolution_panel,
    plot_model2_spectral_panel,
    plot_pinn_ntk_panel,
)

def eval_pinn(mode, model_metadata_file, x_limit, t_limit, eval_params, resolutions, spectral_res, output_dir=None):
    label = 'pinn' if mode == 'data' else 'pinn_no_data'

    model_metadata = torch.load(model_metadata_file, map_location='cpu')
    params = model_metadata['params']
    model_file = model_metadata['model_file']

    outdir = output_dir or os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)

    plot_training_statistics(
        [model_metadata['train_history']],
        [label],
        outdir=outdir,
        filename=f'{label}_training_statistics.png',
        duration_seconds=model_metadata.get('training_duration'),
        final_loss=model_metadata.get('final_loss'),
        num_params=model_metadata.get('num_params'),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, params['param_value'], params['param_value'], 1)
    model = PINN(
        input_size=2,
        output_size=2,
        neurons=params['neurons'],
        hidden_layers=params['hidden_layers'],
        Boussinesq=bsq,
        domain_points=params['domain_points'],
        ic_points=params['ic_points'],
        optimizer_name=params['optimizer_name'],
        lr=params['lr'],
        data=None,
        data_weight=params['data_weight'],
        device=device,
    )
    load_model(model_file, model, device=device)

    median_param = eval_params[len(eval_params) // 2]
    median_res = resolutions[len(resolutions) // 2]

    # PINN uses only one alpha=beta evaluation value for these panels.
    print(f'start pinn evaluation with single alpha=beta={median_param:.3f}')

    res_x_list = []
    res_t_list = []
    res_true_list = []
    res_pred_list = []
    print('start pinn evaluation resolution panel')
    for res in resolutions:
        bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
        solver = PseudoSpectralBoussinesq(bsq_eval, Nx=res, Nt=res - 1, device=device)
        x, t, eta_true, u_true = solver.solve()
        eta_true_t = eta_true.T

        x_pred = np.linspace(-x_limit, x_limit, res, dtype=np.float32)
        t_pred = np.linspace(0.0, t_limit, res, dtype=np.float32)
        X, T = np.meshgrid(x_pred, t_pred, indexing='xy')
        x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
        t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

        model.eval()
        with torch.no_grad():
            eta_pred, _ = model(x_tensor, t_tensor)
        eta_pred = eta_pred.cpu().numpy().reshape(res, res).T

        res_true_list.append(eta_true_t)
        res_pred_list.append(eta_pred)
        res_x_list.append(x)
        res_t_list.append(t)

    plot_model2_resolution_panel(
        res_x_list,
        res_t_list,
        res_true_list,
        res_pred_list,
        resolutions,
        outdir=outdir,
        filename=f'{label}_model2_resolution_panel.png',
        title=f'{"PINN" if label == "pinn" else "PINN No Data"} Resolution Panel (alpha=beta {median_param:.3f})',
        param_label=f'{median_param:.3f}',
    )

    print('start pinn evaluation spectral panel')
    bsq_eval = Boussinesq(-x_limit, x_limit, 0, t_limit, median_param, median_param, 1)
    spectral_res = int(resolutions[0])
    solver = PseudoSpectralBoussinesq(bsq_eval, Nx=spectral_res, Nt=spectral_res - 1, device=device)
    x, t, eta_true, u_true = solver.solve()
    eta_true_t = eta_true.T

    x_pred = np.linspace(-x_limit, x_limit, spectral_res, dtype=np.float32)
    t_pred = np.linspace(0.0, t_limit, spectral_res, dtype=np.float32)
    X, T = np.meshgrid(x_pred, t_pred, indexing='xy')
    x_tensor = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
    t_tensor = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

    model.eval()
    with torch.no_grad():
        eta_pred, _ = model(x_tensor, t_tensor)
    eta_pred = eta_pred.cpu().numpy().reshape(spectral_res, spectral_res).T

    plot_model2_spectral_panel(
        x_pred,
        t_pred,
        eta_true_t,
        eta_pred,
        outdir=outdir,
        filename=f'{label}_model2_spectral_panel.png',
        title=f'{"PINN" if label == "pinn" else "PINN No Data"} Spectral Panel (alpha=beta {median_param:.3f}, res {int(spectral_res)})',
        param_label=f'{median_param:.3f}',
        res_label=f'{int(spectral_res)}',
    )


def run_ntk_experiment(param_value, x_limit, t_limit, widths=None, hidden_layers=4,
                        epochs=15000, lr=1e-5, optimizer_name='sgd', n_ntk=40,
                        log_interval=500, outdir=None, filename='pinn_ntk_panel.png'):
    # train pinns of varying width, track relative parameter/ntk change and eigenvalue spectra
    if widths is None:
        widths = [8, 128, 512]

    outdir = outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'imgs', 'pinn', 'ntk',
    )
    os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsq = Boussinesq(-x_limit, x_limit, 0, t_limit, param_value, param_value, 1)

    # fixed probe points - same across all ntk evaluations so relative changes are meaningful
    x_probe = torch.linspace(-x_limit, x_limit, n_ntk, device=device).unsqueeze(-1)
    t_probe = torch.linspace(0.0, t_limit, n_ntk, device=device).unsqueeze(-1)
    t_zero = torch.zeros_like(x_probe)

    def _param_vec(model):
        return torch.cat([p.detach().view(-1) for p in model.parameters()])

    def _compute_ntk(model):
        # compute K = J J^T via per-residual backward passes; J in R^{N_r x N_theta}
        model.eval()
        params = list(model.parameters())
        x_in = x_probe.detach().clone().requires_grad_(True)
        t_in = t_probe.detach().clone().requires_grad_(True)
        res1, res2 = bsq.residual(model, x_in, t_in)
        all_res = torch.cat([res1.reshape(-1), res2.reshape(-1)])
        N_r = all_res.shape[0]
        rows = []
        for i in range(N_r):
            grads = torch.autograd.grad(
                all_res[i], params,
                retain_graph=(i < N_r - 1),
                create_graph=False,
                allow_unused=True,
            )
            flat = torch.cat([
                g.detach().reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                for g, p in zip(grads, params)
            ])
            rows.append(flat)
        J = torch.stack(rows, dim=0)
        K = (J @ J.T).detach()
        model.train()
        return K

    all_theta_rel = []
    all_k_rel = []
    all_ev_init = []
    all_ev_final = []
    log_epochs = None

    for w in widths:
        print(f'starting ntk experiment (width={w}, layers={hidden_layers})...')
        model = PINN(
            input_size=2,
            output_size=2,
            neurons=w,
            hidden_layers=hidden_layers,
            Boussinesq=bsq,
            domain_points=n_ntk,
            ic_points=n_ntk,
            optimizer_name=optimizer_name,
            lr=lr,
            data=None,
            data_weight=0.0,
            device=device,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        theta0 = _param_vec(model)
        K0 = _compute_ntk(model)
        theta0_norm = theta0.norm().item() + 1e-30
        K0_norm = K0.norm().item() + 1e-30

        ev0 = torch.linalg.eigvalsh(K0).flip(0).clamp(min=1e-30).cpu().numpy()
        all_ev_init.append(ev0)

        theta_hist = []
        k_hist = []
        ep_log = []

        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()

            x_f = torch.rand(n_ntk, 1, device=device) * (2 * x_limit) - x_limit
            t_f = torch.rand(n_ntk, 1, device=device) * t_limit
            res1, res2 = bsq.residual(model, x_f, t_f)
            pde_loss = torch.mean(res1 ** 2 + res2 ** 2)

            eta_pred, u_pred = model(x_probe, t_zero)
            eta_ic, u_ic = bsq.ic(x_probe)
            ic_loss = torch.mean((eta_pred - eta_ic) ** 2 + (u_pred - u_ic) ** 2)

            loss = pde_loss + ic_loss
            loss.backward()
            optimizer.step()

            if ep % log_interval == 0 or ep == epochs - 1:
                theta_n = _param_vec(model)
                theta_rel = (theta_n - theta0).norm().item() / theta0_norm
                K_n = _compute_ntk(model)
                k_rel = (K_n - K0).norm().item() / K0_norm
                theta_hist.append(theta_rel)
                k_hist.append(k_rel)
                ep_log.append(ep)
                print(
                    f'epoch {ep}, loss {loss.item():.4e}, '
                    f'theta_rel {theta_rel:.4f}, k_rel {k_rel:.4f}'
                )

        K_final = _compute_ntk(model)
        ev_final = torch.linalg.eigvalsh(K_final).flip(0).clamp(min=1e-30).cpu().numpy()
        all_ev_final.append(ev_final)
        all_theta_rel.append(theta_hist)
        all_k_rel.append(k_hist)
        if log_epochs is None:
            log_epochs = ep_log

    plot_pinn_ntk_panel(
        all_theta_rel, all_k_rel, log_epochs,
        all_ev_init, all_ev_final, widths,
        outdir=outdir, filename=filename,
    )