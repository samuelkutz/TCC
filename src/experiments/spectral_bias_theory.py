"""Empirical validation of the NTK spectral-bias theory of Chapter 2.

The chapter derives, for a network trained by full-batch gradient descent on data
sampled on a regular periodic grid, that the fitting error obeys the linearized
(frozen-Jacobian) dynamics

    e^{(n)} = (I - mu K_0)^n e_0 = sum_k (v_k^T e_0)(1 - mu lambda_k)^n v_k,   (eq:error_discrete)

with the empirical Neural Tangent Kernel K_0 = J_0 J_0^T = V Lambda V^T and step
size (learning rate) mu. Transformed to Fourier space the same evolution reads

    ehat(tau) = sum_k [(F v_k)^* ehat_0] e^{-lambda_k tau} (F v_k),           (eq:fourier_evolution)

so the low frequencies -- carried by the smooth, large-eigenvalue eigenvectors --
are resolved first. The number of iterations for the k-th eigendirection to fall
below a tolerance eps is

    n_k = ln(|c_k(0)| / eps) / (mu lambda_k),   with c_k(0) = v_k^T e_0.       (eq. 523)

Toy problem.
-----------
To test these predictions in the exact regime in which they are derived -- the
lazy / frozen-Jacobian limit on a periodic grid -- we use a small network on a
128-point periodic grid whose first layer is a *fixed* Fourier-feature map,
    phi_k(x) = g_k cos(2 pi k x),  g_k sin(2 pi k x),   k = 1, ..., M,
and whose only trainable parameters are a linear read-out head, trained by
full-batch gradient descent. Because the network is linear in the trainable
parameters, its Jacobian is exactly constant, so the frozen-Jacobian assumption
behind eq:error_discrete holds identically and the measured error can be compared
against the linearized prediction without any lazy-regime approximation error.
The per-frequency gains g_k are chosen to decay with the frequency k, so that the
NTK eigenvalues lambda_k order by frequency and its eigenvectors v_k are exactly
the grid Fourier modes -- precisely the "regular grid + periodic target" setting
of Section 2 in which the eigenbasis is the Fourier basis. This reproduces the
low-frequency-first bias in a fully controlled way. Nothing here touches the main
FNO/PINO/PINN/MLP models; it is a self-contained CPU experiment. Quantities are
named after the thesis: e (error), ehat (its spectrum), tau = n mu, lambda_k (NTK
eigenvalues), mu (learning rate), n_k (iteration count).
"""

import os

import numpy as np
import torch

from experiments.plots_common import _ensure_outdir, _save_thesis_fig


# house palette, shared with the rest of the figures
_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# figure geometry: these panels are \includegraphics'd at width=\linewidth inside
# 0.48\textwidth subfigures, but the assumed frac is raised the same way as the
# corrected spectral-bias / NTK panels so the printed fonts stay small (ticks
# ~6.5 pt, axis ~7.4 pt, legend ~6.9 pt) instead of dominating the plot.
_W, _H, _FRAC = 820, 520, 1.1


class _FourierFeatureNet(torch.nn.Module):
    # small network with a FIXED Fourier-feature first layer and a trainable linear
    # read-out head. Being linear in the trainable parameters, its Jacobian is
    # exactly the (constant) feature matrix, realizing the frozen-Jacobian regime.
    def __init__(self, x, n_frequencies, gain_exponent):
        super().__init__()
        cols = []
        for k in range(1, n_frequencies + 1):
            gk = np.sqrt(2.0 * float(k) ** (-gain_exponent) / len(x))
            cols.append(gk * np.cos(2.0 * np.pi * k * x))
            cols.append(gk * np.sin(2.0 * np.pi * k * x))
        phi = torch.tensor(np.stack(cols, axis=1), dtype=torch.float64)
        self.register_buffer('phi', phi)                       # (N, 2M) fixed features
        self.head = torch.nn.Linear(phi.shape[1], 1, bias=False).double()
        torch.nn.init.zeros_(self.head.weight)                 # u_theta0 == 0  =>  e_0 = -target

    def forward(self):
        return self.head(self.phi).reshape(-1)


def _empirical_ntk(model):
    # J_0 in R^{N x N_theta}: row i is the parameter gradient of the network output
    # at grid point i, assembled by one backward pass per output. K_0 = J_0 J_0^T.
    params = [p for p in model.parameters() if p.requires_grad]
    out = model()
    N = out.shape[0]
    rows = []
    for i in range(N):
        grads = torch.autograd.grad(out[i], params, retain_graph=(i < N - 1))
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    J0 = torch.stack(rows, dim=0)
    return (J0 @ J0.T).detach()


def _train_and_record(model, target, mu, n_iterations):
    # full-batch gradient descent on the SUM-of-squares loss 1/2 ||e||^2 so a single
    # step is exactly theta <- theta - mu J^T e, i.e. eq:error_discrete_step. Records
    # the measured error e^{(n)} at every iteration, including n = 0.
    optimizer = torch.optim.SGD(model.parameters(), lr=mu)
    e_hist = []
    for _ in range(n_iterations + 1):
        e = model() - target                     # measured error e^{(n)}
        e_hist.append(e.detach().cpu().numpy().copy())
        loss = 0.5 * torch.sum(e ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.asarray(e_hist)                     # (n_iterations+1, N)


def run_spectral_bias_theory(outdir, n_grid=128, n_frequencies=18, gain_exponent=1.5,
                             n_iterations=2500, tolerance=0.08, seed=37):
    import plotly.graph_objects as go

    _ensure_outdir(outdir)
    device = torch.device('cpu')                  # 128-point toy: CPU is plenty
    torch.set_default_dtype(torch.float64)         # clean eigendecomposition
    rng = np.random.default_rng(seed)

    # ---- periodic grid and multi-frequency target (random phases excite every mode)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    target_np = np.zeros_like(x)
    for k in range(1, n_frequencies + 1):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        target_np += np.sin(2.0 * np.pi * k * x + phase)
    target = torch.tensor(target_np, dtype=torch.float64, device=device)

    # ---- network, empirical NTK and its spectrum at initialization -------------
    torch.manual_seed(seed)
    model = _FourierFeatureNet(x, n_frequencies, gain_exponent).to(device)
    K0 = _empirical_ntk(model)
    evals, evecs = torch.linalg.eigh(K0)          # ascending
    order = torch.argsort(evals, descending=True)
    lam = evals[order].clamp(min=0.0).cpu().numpy()          # lambda_1 >= ... >= lambda_N
    V = evecs[:, order].cpu().numpy()                        # columns v_k
    lam_max = float(lam[0])
    mu = 1.0 / lam_max                                        # mu = 1 / lambda_max
    n_modes = int(np.sum(lam > 1e-9))                        # rank = 2 * n_frequencies

    # ---- measured GD error trajectory and its linearized prediction ------------
    e_meas = _train_and_record(model, target, mu, n_iterations)   # (T, N)
    e0 = e_meas[0].copy()                                          # e^{(0)} = e_0 = -target
    n_axis = np.arange(e_meas.shape[0])

    # projection onto the NTK eigenbasis, c_k^{(n)} = v_k^T e^{(n)}
    c0 = V.T @ e0                                                  # c_k(0)
    contraction = (1.0 - mu * lam)                               # (1 - mu lambda_k)
    C_pred = c0[None, :] * (contraction[None, :] ** n_axis[:, None])   # predicted c_k^{(n)}
    C_meas = e_meas @ V                                          # measured c_k^{(n)}
    e_pred = C_pred @ V.T                                        # predicted error field

    def _disp_window(curves, floor=1e-12):
        # last iteration at which any displayed curve is still above `floor`, plus a
        # small margin: past this point every mode sits at the float64 round-off
        # floor and the curves are dead-flat, which is not what the figure is about.
        w = 0
        for y in curves:
            above = np.where(np.asarray(y) > floor)[0]
            if above.size:
                w = max(w, int(above[-1]))
        return min(int(e_meas.shape[0] - 1), int(1.15 * w) + 20)

    # ============================================================================
    # Figure 1 -- e(tau): error decay per NTK eigendirection
    # ============================================================================
    # a spread of eigenindices (skip index 0: with mu = 1/lambda_max the top mode
    # has 1 - mu lambda_1 = 0 and is annihilated in a single step).
    eig_idx = [i for i in (2, 8, 18, 30) if i < n_modes]
    fig = go.Figure()
    for j, k in enumerate(eig_idx):
        color = _PALETTE[j % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=n_axis.tolist(), y=np.abs(C_meas[:, k]).tolist(),
            mode='lines', name=f'k={k+1} measured',
            line=dict(color=color, width=2.4),
        ))
        fig.add_trace(go.Scatter(
            x=n_axis[::40].tolist(), y=np.abs(C_pred[::40, k]).tolist(),
            mode='markers', name=f'k={k+1} theory',
            marker=dict(color=color, size=5, symbol='circle-open', line=dict(width=1.4)),
        ))
    w1 = _disp_window([np.abs(C_meas[:, k]) for k in eig_idx])
    fig.update_xaxes(title_text='Iteration n  (&#964; = n&#956;)', range=[0, w1])
    # clip x to the decay window: past it every displayed mode sits at the float64
    # round-off floor (~1e-14) and the curves are dead-flat, which reads strangely;
    # the geometric decay, where measured and theory overlap, is the point.
    fig.update_yaxes(title_text='|v<sub>k</sub><sup>T</sup> e<sup>(n)</sup>|', type='log',
                     range=[-15, 0.6])
    _save_thesis_fig(
        fig, os.path.join(outdir, 'ntk_error_decay.png'),
        _W, _H, _FRAC,
        extra_layout=dict(
            margin=dict(t=12, b=80, l=66, r=16),
            showlegend=True,
            legend=dict(orientation='h', xanchor='center', x=0.5, yanchor='top', y=-0.22),
        ),
    )

    # ============================================================================
    # Figure 2 -- ehat(tau): spectral error decay in Fourier space
    # ============================================================================
    ehat_meas = np.abs(np.fft.rfft(e_meas, axis=1))               # |ehat_k^{(n)}| measured
    ehat_pred = np.abs(np.fft.rfft(e_pred, axis=1))               # linearized prediction
    n_rfft = ehat_meas.shape[1]
    fourier_modes = [k for k in (1, 4, 9, 15) if k < n_rfft]
    fig = go.Figure()
    for j, k in enumerate(fourier_modes):
        color = _PALETTE[j % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=n_axis.tolist(), y=ehat_meas[:, k].tolist(),
            mode='lines', name=f'k={k} measured',
            line=dict(color=color, width=2.4),
        ))
        fig.add_trace(go.Scatter(
            x=n_axis[::40].tolist(), y=ehat_pred[::40, k].tolist(),
            mode='markers', name=f'k={k} theory',
            marker=dict(color=color, size=5, symbol='circle-open', line=dict(width=1.4)),
        ))
    w2 = _disp_window([ehat_meas[:, k] for k in fourier_modes])
    fig.update_xaxes(title_text='Iteration n  (&#964; = n&#956;)', range=[0, w2])
    fig.update_yaxes(title_text='|&#234;<sub>k</sub><sup>(n)</sup>|', type='log',
                     range=[-15, 0.6])
    _save_thesis_fig(
        fig, os.path.join(outdir, 'ntk_spectral_decay.png'),
        _W, _H, _FRAC,
        extra_layout=dict(
            margin=dict(t=12, b=80, l=66, r=16),
            showlegend=True,
            legend=dict(orientation='h', xanchor='center', x=0.5, yanchor='top', y=-0.22),
        ),
    )

    # ============================================================================
    # Figure 3 -- iteration count: measured crossings vs theoretical n_k
    # ============================================================================
    eps = float(tolerance)
    meas_n, theo_n, color_lam = [], [], []
    abs_C_meas = np.abs(C_meas)
    for k in range(n_modes):
        ck0 = abs(c0[k])
        lk = lam[k]
        if lk <= 0.0 or ck0 <= eps:
            continue                                # mode already resolved / never decays
        nk = np.log(ck0 / eps) / (mu * lk)          # theoretical n_k
        below = np.where(abs_C_meas[:, k] < eps)[0]
        if below.size == 0:
            continue                                # did not cross within n_iterations
        meas_n.append(int(below[0]))
        theo_n.append(float(nk))
        color_lam.append(float(np.log10(lk)))

    fig = go.Figure()
    if theo_n:
        hi = max(max(theo_n), max(meas_n))
        pad = 0.05 * hi
        diag = [-pad, hi + pad]
        fig.add_trace(go.Scatter(
            x=diag, y=diag, mode='lines', name='y = x',
            line=dict(color='#888888', width=1.6, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=theo_n, y=meas_n, mode='markers', name='eigendirections',
            marker=dict(size=8, color=color_lam, colorscale='Viridis',
                        showscale=True, line=dict(width=0.6, color='#333333'),
                        colorbar=dict(title='log&#8321;&#8320; &#955;<sub>k</sub>',
                                      thickness=12, outlinewidth=0)),
        ))
    fig.update_xaxes(title_text='Theoretical n<sub>k</sub> = ln(|c<sub>k</sub>(0)|/&#949;) / (&#956; &#955;<sub>k</sub>)')
    fig.update_yaxes(title_text='Measured iterations to |c<sub>k</sub>| &lt; &#949;')
    _save_thesis_fig(
        fig, os.path.join(outdir, 'ntk_iteration_count.png'),
        _W, _H, _FRAC,
        extra_layout=dict(
            margin=dict(t=12, b=58, l=72, r=16),
            showlegend=True,
            legend=dict(orientation='h', xanchor='center', x=0.5, yanchor='top', y=-0.22),
        ),
    )

    torch.set_default_dtype(torch.float32)
    print(f'spectral-bias theory figures saved to {outdir} '
          f'(n_modes={n_modes}, lambda_max={lam_max:.4g}, mu={mu:.4g}, eps={eps:g})')
