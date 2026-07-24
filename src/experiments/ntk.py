import os

import numpy as np
import torch

from methods.boussinesq import Boussinesq
from methods.pinn import PINN
from tools import set_seed
from experiments.plots_common import _style_thesis, _ensure_outdir, THESIS_PALETTE


def plot_pinn_ntk_panel(theta_rel_histories, k_rel_histories, log_epochs,
                         eigenvalues_init, eigenvalues_final, widths, outdir, filename):
    # NTK diagnostic rendered as THREE standalone figures (parameter drift, kernel
    # drift, eigenvalue spectrum) instead of one wide three-column panel, so each
    # can be \includegraphics'd on its own with legible, print-sized fonts.
    import plotly.graph_objects as go

    _colors = THESIS_PALETTE
    _ensure_outdir(outdir)
    base, _, ext = filename.rpartition('.')
    base = base or filename
    ext = ext or 'png'

    # frac raised well above the 0.48 subfigure width so the in-figure text prints
    # about half its former size (ticks ~6.5 pt, axis ~7.4 pt, legend ~6.9 pt) and
    # no longer dominates the panel; the legend is moved out below the axes so it
    # never overlaps the curves.
    W, H, FRAC = 860, 560, 1.1

    def _finish(fig, title, xlab, ylab, suffix, logx=False, logy=False):
        fig.update_layout(
            xaxis_title=xlab, yaxis_title=ylab,
            width=W, height=H, showlegend=True,
            margin=dict(t=12, b=82, l=64, r=16),
            legend=dict(orientation='h', xanchor='center', x=0.5,
                        yanchor='top', y=-0.20),
        )
        if logx:
            fig.update_xaxes(type='log')
        if logy:
            fig.update_yaxes(type='log')
        _style_thesis(fig, W, frac=FRAC)
        outpath = os.path.join(outdir, f'{base}_{suffix}.{ext}')
        try:
            fig.write_image(outpath, scale=2.0)
            print(f'ntk {suffix} figure saved to {outpath}')
        except Exception as e:
            print('Erro ao salvar figura NTK como PNG. Instale kaleido: pip install kaleido')
            raise e

    # (a) relative parameter change
    fig_a = go.Figure()
    for i, (w, theta_hist) in enumerate(zip(widths, theta_rel_histories)):
        fig_a.add_trace(go.Scatter(
            x=log_epochs, y=theta_hist, mode='lines', name=f'width={w}',
            line=dict(color=_colors[i % len(_colors)], width=2.6),
        ))
    _finish(fig_a, 'Relative parameter change', 'Iteration',
            'Rel. parameter change', 'param')

    # (b) relative NTK change
    fig_b = go.Figure()
    for i, (w, k_hist) in enumerate(zip(widths, k_rel_histories)):
        fig_b.add_trace(go.Scatter(
            x=log_epochs, y=k_hist, mode='lines', name=f'width={w}',
            line=dict(color=_colors[i % len(_colors)], width=2.6),
        ))
    _finish(fig_b, 'Relative NTK change', 'Iteration',
            'Rel. NTK change', 'kernel')

    # (c) eigenvalue spectrum: solid = initialization, dashed = end of training
    fig_c = go.Figure()
    for i, (w, ev_init, ev_final) in enumerate(zip(widths, eigenvalues_init, eigenvalues_final)):
        color = _colors[i % len(_colors)]
        idx = np.arange(1, len(ev_init) + 1).tolist()
        fig_c.add_trace(go.Scatter(
            x=idx, y=list(ev_init), mode='lines+markers', name=f'width={w} (init)',
            line=dict(color=color, width=2.4, dash='solid'),
            marker=dict(size=5, symbol='circle'),
        ))
        idx_f = np.arange(1, len(ev_final) + 1).tolist()
        fig_c.add_trace(go.Scatter(
            x=idx_f, y=list(ev_final), mode='lines+markers', name=f'width={w} (final)',
            line=dict(color=color, width=2.4, dash='dash'),
            marker=dict(size=5, symbol='diamond'),
        ))
    _finish(fig_c, 'NTK eigenvalue spectrum', 'Eigenvalue index', 'Eigenvalue',
            'spectrum', logx=True, logy=True)


def run_ntk_experiment(param_value, x_limit, t_limit, widths=None, hidden_layers=4,
                        epochs=15000, lr=1e-5, optimizer_name='sgd', n_ntk=40,
                        log_interval=500, outdir=None, filename='pinn_ntk_panel.png',
                        seed=37):
    # train pinns of varying width, track relative parameter/ntk change and eigenvalue spectra
    # reseed locally so the diagnostic reproduces regardless of pipeline order: this
    # experiment retrains its probe networks and would otherwise inherit whatever RNG
    # state the earlier training stages left behind.
    set_seed(seed)
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
