import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# Thesis figure styling
# ---------------------------------------------------------------------------
# The figures are \includegraphics'd into a 12 pt report whose text block is
# ~160 mm ~ 6.30 in wide. A plotly figure of logical width W px, included at
# frac*\textwidth, maps W px -> frac*6.30 in, so a font of f px prints at
#     f * frac * 6.30 * 72 / W   points.
# We invert that so in-figure text lands at a chosen *printed* point size, kept
# just around the 12 pt body text for a consistent visual ratio across every
# figure regardless of its pixel dimensions.
THESIS_TEXTWIDTH_IN = 6.30

# target printed sizes, in points (body text is 12 pt); figure text sits a
# little below the body, the usual ratio for a 12 pt document. Kept a touch
# smaller than the body so the panels read as figures, not as running text.
_THESIS_PT = dict(tick=7.5, axis=8.5, subplot=9.0, title=10.0, legend=8.0, annot=7.0)


def _thesis_px(pt, fig_width_px, frac):
    # plotly font size (logical px) that prints at `pt` when a figure of logical
    # width `fig_width_px` is \includegraphics'd at `frac`*\textwidth
    return pt * fig_width_px / (frac * THESIS_TEXTWIDTH_IN * 72.0)


def _style_thesis(fig, fig_width_px, frac=1.0):
    # apply consistent, print-sized fonts to a plotly figure and return the
    # resolved px sizes so callers can size their own annotations to match.
    px = {k: _thesis_px(v, fig_width_px, frac) for k, v in _THESIS_PT.items()}
    fig.update_layout(
        template='plotly_white',
        font=dict(size=px['tick']),
        title_font_size=px['title'],
        legend_font_size=px['legend'],
    )
    fig.update_xaxes(title_font_size=px['axis'], tickfont_size=px['tick'])
    fig.update_yaxes(title_font_size=px['axis'], tickfont_size=px['tick'])
    # subplot titles (and any annotations already present) come from make_subplots
    # as layout annotations; size them to the subplot-title target
    for ann in fig.layout.annotations:
        ann.font.size = px['subplot']
    return px


def _ensure_outdir(outdir):
    # make sure output directory exists before saving files
    os.makedirs(outdir, exist_ok=True)
    return outdir


# plot training loss
def plot_training_loss(train_loss_history, outdir, filename, duration_seconds=None, final_loss=None, num_params=None):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    epochs = np.arange(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss_history, color='#1f77b4', lw=2.2)
    ax.set_title('Training $L^2$ Loss History', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('$L^2$ Loss', fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis='both', which='major', labelsize=10)

    annotation_lines = []
    if num_params is not None:
        annotation_lines.append(f'Params: {num_params:,}')
    if duration_seconds is not None:
        annotation_lines.append(f'Training time: {duration_seconds:.1f}s')
    if final_loss is not None:
        annotation_lines.append(f'Final loss: {final_loss:.2e}')
    if annotation_lines:
        ax.text(
            0.02,
            0.98,
            '\n'.join(annotation_lines),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.4'),
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f'training loss plot saved to {outpath}')
    plt.close(fig)


def plot_soliton_profile(outdir, filename='soliton_profile.png', A=1.0):
    import plotly.graph_objects as go
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.linspace(-6, 6, 600)
    y = A / np.cosh(x) ** 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x.tolist(), y=y.tolist(),
        mode='lines',
        line=dict(color='#ff7f0e', width=2.2),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='η(x,0)',
        template='plotly_white',
        width=1000, height=400,
        margin=dict(t=15, b=45, l=55, r=25),
    )
    _style_thesis(fig, 1000, frac=1.0)
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'soliton profile saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar soliton profile como PNG. Instale kaleido: pip install kaleido')
        raise e


def _build_tanh_mlp_on_grid(n_grid, width, hidden_layers, periodic, seed):
    # build a tanh MLP u_theta(x): R -> R on a uniform periodic grid on [-1, 1).
    # returns the grid x, the model, and the (embedded) network input on the grid.
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')

    # uniform periodic grid on [-1, 1), matching eq:grid
    x = (-1.0 + 2.0 * np.arange(n_grid) / n_grid).astype(np.float32)
    x_t = torch.tensor(x, device=device).unsqueeze(-1)
    if periodic:
        # periodic embedding: map the coordinate onto the circle
        inp = torch.cat([torch.cos(np.pi * x_t), torch.sin(np.pi * x_t)], dim=1)
        din = 2
    else:
        inp = x_t
        din = 1

    layers = [nn.Linear(din, width), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    model = nn.Sequential(*layers).to(device)
    return x, model, inp


def _empirical_ntk(model, inp):
    # empirical NTK K0 = J J^T at the current parameters, J_i = d u_theta(x_i) / d theta
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    out = model(inp).reshape(-1)
    n = out.shape[0]
    rows = []
    for i in range(n):
        grads = torch.autograd.grad(out[i], params, retain_graph=(i < n - 1), create_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    J = torch.stack(rows, dim=0)
    return (J @ J.T).detach().cpu().numpy()


def _empirical_ntk_on_grid(n_grid, width, hidden_layers, periodic, seed):
    # build a tanh MLP on a uniform grid and return (x, K0) at initialization
    x, model, inp = _build_tanh_mlp_on_grid(n_grid, width, hidden_layers, periodic, seed)
    model.eval()
    K0 = _empirical_ntk(model, inp)
    return x, K0


def plot_ntk_eigenvector_spectrogram(outdir, filename='ntk_eigenvector_spectrogram.png',
                                     n_grid=128, width=128, hidden_layers=4, seed=37,
                                     periodic=True, n_show=64):
    # companion experiment for the simple argument: the Fourier content |F v_k| of each
    # NTK eigenvector v_k, with eigenvectors ordered by descending eigenvalue lambda_k.
    # if the bright band drifts from low frequency (top, large lambda) to high frequency
    # (bottom, small lambda), the single empirical claim holds -- straight from the
    # eigenvectors, with no S(xi_k) and no circulance assumption.
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    _, K0 = _empirical_ntk_on_grid(n_grid, width, hidden_layers, periodic, seed)

    w, Q = np.linalg.eigh(K0)            # ascending eigenvalues, columns are eigenvectors
    order = np.argsort(w)[::-1]          # sort by descending lambda_k
    Q = Q[:, order]

    # |F v_k| for each eigenvector: rfft along the grid axis, rows = eigenvectors
    spec = np.abs(np.fft.rfft(Q, axis=0)).T   # (n_eig, n_pos)
    spec = spec / (spec.max(axis=1, keepdims=True) + 1e-12)  # row-normalize for visibility
    if n_show is not None:
        spec = spec[:n_show]
    n_eig, n_pos = spec.shape

    import plotly.graph_objects as go

    modes = np.arange(n_pos)
    ranks = np.arange(1, n_eig + 1)
    fig = go.Figure(data=go.Heatmap(
        z=spec, x=modes.tolist(), y=ranks.tolist(),
        colorscale='Viridis',
        colorbar=dict(title='|F v_k| (row max = 1)'),
    ))
    fig.update_layout(
        title_text='Fourier content |F v_k| of NTK eigenvectors (row-normalized)',
        title_x=0.5,
        xaxis_title='Fourier mode',
        yaxis_title='Eigenvector rank (by descending lambda_k)',
        template='plotly_white',
        width=1000, height=600,
    )
    fig.update_yaxes(autorange='reversed')  # rank 1 (largest lambda_k) on top
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'ntk eigenvector spectrogram saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar ntk eigenvector spectrogram como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_ntk_spectral_prediction(outdir, filename='ntk_spectral_prediction.png',
                                 n_grid=128, width=128, hidden_layers=4, seed=37, periodic=True,
                                 target_modes=(1, 2, 3, 4), probe_modes=(1, 2, 3, 4),
                                 n_iters=30000, stability_factor=1.0, n_checkpoints=70):
    # decisive experiment: predict the per-frequency error decay from the NTK at init and
    # compare against actual gradient descent. linearized (lazy) prediction is
    #   e(n) = (I - mu K0)^n e0,  the exact forward-Euler/GD integration of de/dtau = -K0 e.
    # success = the trained curves overlay the NTK curves, and low modes decay before high.
    import torch

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x, model, inp = _build_tanh_mlp_on_grid(n_grid, width, hidden_layers, periodic, seed)
    device = inp.device
    params = [p for p in model.parameters() if p.requires_grad]

    # target: a real periodic signal carrying unit amplitude in the chosen Fourier modes
    n_pos = n_grid // 2 + 1
    u_hat_target = np.zeros(n_pos, dtype=complex)
    for m in target_modes:
        u_hat_target[m] = 1.0
    u = np.fft.irfft(u_hat_target, n=n_grid).astype(np.float32)
    u_t = torch.tensor(u, device=device)

    # NTK and initial error at initialization
    model.eval()
    K0 = _empirical_ntk(model, inp)
    e0 = (model(inp).reshape(-1).detach().cpu().numpy() - u)

    # linearized prediction  e(n) = Q (1 - mu w)^n Q^T e0
    w, Q = np.linalg.eigh(K0)
    w = np.clip(w, 0.0, None)
    lam_max = float(w[-1])
    mu = stability_factor / lam_max
    coeff0 = Q.T @ e0
    factor = 1.0 - mu * w

    def e_pred(n):
        return Q @ ((factor ** n) * coeff0)

    # actual full-batch gradient descent on L = 1/2 sum (u_theta - u)^2
    opt = torch.optim.SGD(params, lr=mu)

    checkpoints = np.unique(np.round(np.logspace(0, np.log10(n_iters), n_checkpoints)).astype(int))
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= n_iters)]
    cp_set = set(int(c) for c in checkpoints)

    probe = list(probe_modes)
    measured = {m: [] for m in probe}
    predicted = {m: [] for m in probe}
    rec_iters = []

    for n in range(1, n_iters + 1):
        opt.zero_grad()
        out = model(inp).reshape(-1)
        loss = 0.5 * ((out - u_t) ** 2).sum()
        loss.backward()
        opt.step()
        if n in cp_set:
            e_meas_hat = np.abs(np.fft.rfft(out.detach().cpu().numpy() - u))
            e_pred_hat = np.abs(np.fft.rfft(e_pred(n)))
            for m in probe:
                measured[m].append(float(e_meas_hat[m]))
                predicted[m].append(float(e_pred_hat[m]))
            rec_iters.append(int(n))

    import plotly.graph_objects as go

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig = go.Figure()
    for i, m in enumerate(probe):
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=rec_iters, y=measured[m], mode='lines', name=f'mode {m} (gradient descent)',
            line=dict(color=c, width=2.6),
        ))
        fig.add_trace(go.Scatter(
            x=rec_iters, y=predicted[m], mode='lines', name=f'mode {m} (NTK prediction)',
            line=dict(color=c, width=1.6, dash='dash'),
        ))
    fig.update_layout(
        title_text='Per-frequency error: gradient descent vs NTK prediction',
        title_x=0.5,
        xaxis_title='Iteration n',
        yaxis_title='|error spectrum at mode k|',
        xaxis_type='log', yaxis_type='log',
        template='plotly_white', width=1000, height=560,
        legend=dict(x=0.01, y=0.01, xanchor='left', yanchor='bottom'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'ntk spectral prediction saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar ntk spectral prediction como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_ntk_iteration_estimator(outdir, filename='ntk_iteration_estimator.png',
                                 n_grid=128, width=128, hidden_layers=4, seed=37, periodic=True,
                                 target_modes=(1, 2, 3, 4, 5, 6, 8, 10), n_iters=40000,
                                 stability_factor=1.0, eps_frac=1e-2, n_checkpoints=200):
    # test of the iteration estimator n_k = ceil( ln(|c_k(0)|/eps) / (mu lambda_k) ): the
    # predicted number of gradient-descent steps to bring eigen-mode k below eps. we run
    # actual GD, project the measured error onto the init eigenvectors v_k, read off the
    # iteration where |c_k(n)| first crosses eps, and scatter measured vs predicted. points
    # on the diagonal mean the estimator is right.
    import torch

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x, model, inp = _build_tanh_mlp_on_grid(n_grid, width, hidden_layers, periodic, seed)
    device = inp.device
    params = [p for p in model.parameters() if p.requires_grad]

    n_pos = n_grid // 2 + 1
    u_hat_target = np.zeros(n_pos, dtype=complex)
    for m in target_modes:
        u_hat_target[m] = 1.0
    u = np.fft.irfft(u_hat_target, n=n_grid).astype(np.float32)
    u_t = torch.tensor(u, device=device)

    model.eval()
    K0 = _empirical_ntk(model, inp)
    e0 = (model(inp).reshape(-1).detach().cpu().numpy() - u)

    w, Q = np.linalg.eigh(K0)
    w = np.clip(w, 0.0, None)
    lam_max = float(w[-1])
    mu = stability_factor / lam_max

    c0 = Q.T @ e0                                  # modal amplitudes c_k(0) = v_k^T e0
    eps = eps_frac * float(np.max(np.abs(c0)))

    # estimator: n_k = ln(|c_k(0)|/eps) / (mu lambda_k)   (only where it is positive)
    with np.errstate(divide='ignore', invalid='ignore'):
        n_pred = np.log(np.abs(c0) / eps) / (mu * w)

    # actual GD, recording modal amplitudes c_k(n) = v_k^T e(n) at log-spaced checkpoints
    opt = torch.optim.SGD(params, lr=mu)
    checkpoints = np.unique(np.round(np.logspace(0, np.log10(n_iters), n_checkpoints)).astype(int))
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= n_iters)]
    cp_set = set(int(c) for c in checkpoints)

    rec_iters = []
    c_hist = []                                    # |c_k(n)| at each checkpoint
    for n in range(1, n_iters + 1):
        opt.zero_grad()
        out = model(inp).reshape(-1)
        loss = 0.5 * ((out - u_t) ** 2).sum()
        loss.backward()
        opt.step()
        if n in cp_set:
            e_meas = out.detach().cpu().numpy() - u
            c_hist.append(np.abs(Q.T @ e_meas))
            rec_iters.append(int(n))
    rec_iters = np.asarray(rec_iters)
    c_hist = np.asarray(c_hist)                     # (n_cp, N)

    # measured crossing iteration per mode: first checkpoint with |c_k(n)| < eps,
    # refined by log-log interpolation between the bracketing checkpoints
    n_meas = np.full(w.shape[0], np.nan)
    for k in range(w.shape[0]):
        below = np.where(c_hist[:, k] < eps)[0]
        if below.size == 0 or c_hist[0, k] < eps:
            continue
        j = below[0]
        if j == 0:
            n_meas[k] = rec_iters[0]
            continue
        n1, n2 = rec_iters[j - 1], rec_iters[j]
        a1, a2 = c_hist[j - 1, k], c_hist[j, k]
        # interpolate in (log n, log amplitude)
        t = (np.log(eps) - np.log(a1)) / (np.log(a2) - np.log(a1) + 1e-30)
        n_meas[k] = np.exp(np.log(n1) + t * (np.log(n2) - np.log(n1)))

    # keep eigen-modes whose predicted crossing falls inside the training budget
    valid = np.isfinite(n_meas) & np.isfinite(n_pred) & (n_pred >= 1) & (n_pred <= n_iters)
    xp = n_pred[valid]
    yp = n_meas[valid]
    lam = w[valid]

    import plotly.graph_objects as go

    lim_lo = max(1.0, float(min(xp.min(), yp.min())) * 0.6)
    lim_hi = min(float(n_iters), float(max(xp.max(), yp.max())) * 1.5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode='lines',
        line=dict(color='black', width=1.3, dash='dash'),
        name='n_measured = n_k',
    ))
    fig.add_trace(go.Scatter(
        x=xp.tolist(), y=yp.tolist(), mode='markers', showlegend=False,
        marker=dict(
            size=11, color=np.log10(lam).tolist(), colorscale='Viridis',
            showscale=True, colorbar=dict(title='log10 lambda_k'),
            line=dict(color='black', width=0.5),
        ),
    ))
    fig.update_layout(
        title_text='Iteration estimator n_k vs measured crossing',
        title_x=0.5,
        xaxis_title='Predicted n_k = ln(|c_k(0)|/eps) / (mu lambda_k)',
        yaxis_title='Measured iteration where |c_k(n)| < eps',
        xaxis_type='log', yaxis_type='log',
        xaxis_range=[np.log10(lim_lo), np.log10(lim_hi)],
        yaxis_range=[np.log10(lim_lo), np.log10(lim_hi)],
        template='plotly_white', width=820, height=760,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'ntk iteration estimator saved to {outpath} ({int(valid.sum())} modes, eps={eps:.2e}, mu={mu:.2e})')
    except Exception as e:
        print('Erro ao salvar ntk iteration estimator como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_ntk_spectral_estimate(outdir, filename='ntk_spectral_estimate.png',
                               n_grid=128, width=128, hidden_layers=4, seed=37, periodic=True,
                               target_n_modes=16, snapshots=(0, 30, 300, 3000, 30000),
                               n_iters=30000, stability_factor=1.0):
    # test of the full spectral estimate: the linearized NTK predicts the entire error
    # spectrum at any time,  e_hat(n) = F (I - mu K0)^n e0.  using a broadband target, we
    # snapshot the measured spectrum |e_hat_k(n)| over all modes at several iterations and
    # overlay the prediction. agreement (and the low-to-high erosion front) tests the
    # estimate of the whole spectrum, not just a few probe modes over time.
    import torch

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x, model, inp = _build_tanh_mlp_on_grid(n_grid, width, hidden_layers, periodic, seed)
    device = inp.device
    params = [p for p in model.parameters() if p.requires_grad]

    # broadband target: unit amplitude across the first target_n_modes Fourier modes
    n_pos = n_grid // 2 + 1
    u_hat_target = np.zeros(n_pos, dtype=complex)
    u_hat_target[1:target_n_modes + 1] = 1.0
    u = np.fft.irfft(u_hat_target, n=n_grid).astype(np.float32)
    u_t = torch.tensor(u, device=device)

    model.eval()
    K0 = _empirical_ntk(model, inp)
    e0 = (model(inp).reshape(-1).detach().cpu().numpy() - u)

    w, Q = np.linalg.eigh(K0)
    w = np.clip(w, 0.0, None)
    lam_max = float(w[-1])
    mu = stability_factor / lam_max
    coeff0 = Q.T @ e0
    factor = 1.0 - mu * w

    def e_pred(n):
        return Q @ ((factor ** n) * coeff0)

    snaps = sorted(s for s in set(snapshots) if 0 <= s <= n_iters)
    snap_set = set(snaps)
    measured = {}
    if 0 in snap_set:
        measured[0] = np.abs(np.fft.rfft(e0))

    opt = torch.optim.SGD(params, lr=mu)
    for n in range(1, n_iters + 1):
        opt.zero_grad()
        out = model(inp).reshape(-1)
        loss = 0.5 * ((out - u_t) ** 2).sum()
        loss.backward()
        opt.step()
        if n in snap_set:
            measured[n] = np.abs(np.fft.rfft(out.detach().cpu().numpy() - u))

    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale

    modes = np.arange(n_pos)
    cs = sample_colorscale('Viridis', [i / max(len(snaps) - 1, 1) for i in range(len(snaps))])
    fig = go.Figure()
    for i, n in enumerate(snaps):
        c = cs[i]
        meas = np.clip(measured[n], 1e-12, None)
        pred = np.clip(np.abs(np.fft.rfft(e_pred(n))), 1e-12, None)
        fig.add_trace(go.Scatter(
            x=modes.tolist(), y=meas.tolist(), mode='lines',
            name=f'n={n} (gradient descent)', line=dict(color=c, width=2.6),
        ))
        fig.add_trace(go.Scatter(
            x=modes.tolist(), y=pred.tolist(), mode='lines',
            line=dict(color=c, width=1.5, dash='dash'), showlegend=False,
        ))
    fig.update_layout(
        title_text='Error spectrum along training: gradient descent (solid) vs NTK prediction (dashed)',
        title_x=0.5,
        xaxis_title='Fourier mode',
        yaxis_title='|e_hat_k|',
        yaxis_type='log',
        xaxis_range=[0, target_n_modes + 4],
        template='plotly_white', width=1000, height=560,
        legend=dict(x=0.99, y=0.01, xanchor='right', yanchor='bottom', title='dashed = NTK estimate'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'ntk spectral estimate saved to {outpath} (mu={mu:.2e})')
    except Exception as e:
        print('Erro ao salvar ntk spectral estimate como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_ntk_drift(outdir, filename='ntk_drift.png',
                   n_grid=128, width=128, hidden_layers=4, seed=37, periodic=True,
                   target_modes=(1, 2, 3, 4), n_iters=40000, stability_factor=1.0,
                   n_checkpoints=60):
    # Wang et al. (2020) style NTK stability diagnostic for the grid experiment: track the
    # relative parameter change ||theta_n - theta_0|| / ||theta_0|| and the relative NTK
    # change ||K_n - K_0||_F / ||K_0||_F along training, plus the eigenvalue spectrum at
    # init vs final. a kink in the relative NTK change flags where the lazy (constant-kernel)
    # regime ends -- i.e. where the linearized predictions start to depart from real GD.
    import torch

    _ensure_outdir(outdir)

    x, model, inp = _build_tanh_mlp_on_grid(n_grid, width, hidden_layers, periodic, seed)
    params = [p for p in model.parameters() if p.requires_grad]

    def flat_params():
        return torch.cat([p.detach().reshape(-1) for p in params]).cpu().numpy()

    n_pos = n_grid // 2 + 1
    u_hat_target = np.zeros(n_pos, dtype=complex)
    for m in target_modes:
        u_hat_target[m] = 1.0
    u = np.fft.irfft(u_hat_target, n=n_grid).astype(np.float32)
    u_t = torch.tensor(u, device=inp.device)

    model.eval()
    theta0 = flat_params()
    theta0_norm = float(np.linalg.norm(theta0))
    K0 = _empirical_ntk(model, inp)
    K0_norm = float(np.linalg.norm(K0))
    eig_init = np.sort(np.linalg.eigvalsh(K0))[::-1]
    mu = stability_factor / float(eig_init[0])

    opt = torch.optim.SGD(params, lr=mu)
    # log-spaced checkpoints, plus a linear grid so the late iterations (around the suspected
    # transition) are densely covered enough to resolve a sudden change
    log_cp = np.round(np.logspace(0, np.log10(n_iters), n_checkpoints)).astype(int)
    lin_cp = np.arange(2000, n_iters + 1, 2000)
    checkpoints = np.unique(np.concatenate([log_cp, lin_cp]))
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= n_iters)]
    cp_set = set(int(c) for c in checkpoints)

    log_iters, theta_rel, k_rel = [], [], []
    for n in range(1, n_iters + 1):
        opt.zero_grad()
        out = model(inp).reshape(-1)
        loss = 0.5 * ((out - u_t) ** 2).sum()
        loss.backward()
        opt.step()
        if n in cp_set:
            th = flat_params()
            theta_rel.append(float(np.linalg.norm(th - theta0) / theta0_norm))
            Kn = _empirical_ntk(model, inp)
            k_rel.append(float(np.linalg.norm(Kn - K0) / K0_norm))
            log_iters.append(int(n))
    eig_final = np.sort(np.linalg.eigvalsh(_empirical_ntk(model, inp)))[::-1]

    # diagnostic: largest single-checkpoint jump in the relative NTK change (sudden change?)
    kr = np.asarray(k_rel)
    li = np.asarray(log_iters)
    if kr.size > 1:
        dk = np.diff(kr)
        j = int(np.argmax(dk))
        print(f'ntk drift (mu={mu:.2e}): final ||dtheta||/||theta0|| = {theta_rel[-1]:.3e}, '
              f'final ||dK||_F/||K0||_F = {k_rel[-1]:.3e}')
        print(f'  largest jump in rel NTK change: +{dk[j]:.3e} between iter {li[j]} and {li[j + 1]} '
              f'({kr[j]:.3e} -> {kr[j + 1]:.3e})')

    plot_pinn_ntk_panel(
        theta_rel_histories=[theta_rel],
        k_rel_histories=[k_rel],
        log_epochs=log_iters,
        eigenvalues_init=[eig_init],
        eigenvalues_final=[eig_final],
        widths=[width],
        outdir=outdir,
        filename=filename,
    )


def plot_training_statistics(histories, labels, outdir, filename, log_scale=True, duration_seconds=None, final_loss=None, num_params=None):
    import plotly.graph_objects as go
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig = go.Figure()
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = list(range(1, len(history) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=list(history),
            mode='lines', name=label,
            line=dict(color=_colors[i % len(_colors)], width=2.2),
            opacity=0.9,
        ))

    annotation_lines = []
    if num_params is not None:
        annotation_lines.append(f'Params: {num_params:,}')
    if duration_seconds is not None:
        annotation_lines.append(f'Training time: {duration_seconds:.1f}s')
    if final_loss is not None:
        annotation_lines.append(f'Final loss: {final_loss:.2e}')

    fig.update_layout(
        title_text='Training L2 Loss', title_x=0.5,
        xaxis_title='Epoch',
        yaxis_title='L2 Loss',
        yaxis_type='log' if log_scale else 'linear',
        template='plotly_white',
        width=1000, height=500,
        showlegend=len(labels) > 1,
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
    )
    if annotation_lines:
        fig.add_annotation(
            text='<br>'.join(annotation_lines),
            xref='paper', yref='paper',
            x=0.02, y=0.98, xanchor='left', yanchor='top',
            showarrow=False,
            bordercolor='#aaaaaa', borderwidth=1, borderpad=8,
            bgcolor='white', opacity=0.92,
            font=dict(size=11),
        )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'training statistics plot saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar training statistics como PNG. Instale kaleido: pip install kaleido')
        raise e


# spectral relative error metrics
def compute_spectral_relative_error(eta_true, eta_pred):
    # compute relative error in fourier spectrum, using rfft2 over space-time
    true_fft = np.fft.rfft2(eta_true)
    pred_fft = np.fft.rfft2(eta_pred)
    diff_fft = true_fft - pred_fft
    rel_error = np.linalg.norm(diff_fft) / (np.linalg.norm(true_fft) + 1e-12)
    return float(rel_error)


def compute_relative_error(eta_true, eta_pred, floor_ratio=1e-2, floor_min=1e-3):
    # |eta_true - eta_pred| / max(|eta_true|, floor) - floor avoids spikes near zero
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    abs_true = np.abs(eta_true)
    floor = np.maximum(np.max(abs_true) * floor_ratio, floor_min)
    denom = np.maximum(abs_true, floor)
    rel_error = np.abs(eta_true - eta_pred) / denom
    return np.nan_to_num(rel_error, posinf=1e3, neginf=0.0)


def spatial_wavenumbers(x):
    # spatial wavenumbers kx = n*pi/L for domain [-L, L]
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)

    domain_length = x[-1] - x[0]
    L = domain_length / 2.0
    n_freq = len(x) // 2 + 1
    return np.arange(n_freq, dtype=float) * np.pi / L


def spectral_mode_index(x):
    # discrete mode index: kx * (2L/pi) so axis shows integer mode numbers
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)

    domain_length = x[-1] - x[0]
    L = domain_length / 2.0
    return spatial_wavenumbers(x) * (2.0 * L / np.pi)


def plot_spectral_summary(eta_true, eta_pred, x, t, outdir, filename, title):
    # plot true spectrum and its relative difference over spatial frequencies kx
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x)
    t = np.asarray(t)
    kx = spectral_mode_index(x)

    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    rel_spec = compute_relative_error(true_spec, pred_spec)
    rel_spec = np.clip(rel_spec, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im0 = axes[0].imshow(true_spec.T,
                          extent=[kx[0], kx[-1], t[0], t[-1]],
                          origin='lower',
                          aspect='auto',
                          cmap='viridis')
    axes[0].set_title('True Spatial Spectrum Over Time')
    axes[0].set_xlabel('Spectral index n')
    axes[0].set_ylabel('Time')
    plt.colorbar(im0, ax=axes[0], label='Amplitude')

    im1 = axes[1].imshow(rel_spec.T,
                          extent=[kx[0], kx[-1], t[0], t[-1]],
                          origin='lower',
                          aspect='auto',
                          cmap='inferno',
                          vmin=0.0,
                          vmax=1.0)
    axes[1].set_title('Relative Spatial Spectrum Error')
    axes[1].set_xlabel('Spectral index n')
    plt.colorbar(im1, ax=axes[1], label='Relative Error')

    fig.suptitle(title)
    plt.savefig(outpath, dpi=150)
    print(f'spectral summary saved to {outpath}')
    plt.close()


def plot_relative_error_panel(x, t, eta_true, eta_pred, times, outdir, filename, title):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x)
    t = np.asarray(t)
    times = np.array(times if times is not None else [t[0], t[-1]], dtype=float)
    indices = [int(np.argmin(np.abs(t - ti))) for ti in times]

    rel_error = compute_relative_error(eta_true, eta_pred)
    rel_error = np.clip(rel_error, 0.0, 1.0)
    time_relative_norm = np.clip(
        np.linalg.norm(np.abs(eta_true - eta_pred), axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-8),
        0.0,
        1.0,
    )

    kx = spectral_mode_index(x)
    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    rel_spec = np.abs(pred_spec - true_spec) / (true_spec + 1e-12)
    rel_spec = np.clip(rel_spec, 0.0, 1.0)

    fig = plt.figure(figsize=(14, 14), constrained_layout=False)
    gs = fig.add_gridspec(3, 2)

    for idx_pos, time_index in enumerate(indices):
        if idx_pos >= 2:
            break
        ax = fig.add_subplot(gs[0, idx_pos])
        ax.plot(x, eta_true[:, time_index], color='black', lw=1.8, label='True')
        ax.plot(x, eta_pred[:, time_index], color='#ff7f0e', lw=1.8, linestyle='--', label='predicted')
        ax.set_title(f'Prediction at $t = {t[time_index]:.2f}$')
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('eta(x,t)')
        ax.grid(True, alpha=0.3)
        if idx_pos == 0:
            ax.legend(fontsize='small', loc='best')

    ax10 = fig.add_subplot(gs[1, 0])
    im = ax10.imshow(
        rel_error.T, extent=[x[0], x[-1], t[0], t[-1]],
        origin='lower', aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
    ax10.set_title('Relative Error Heatmap')
    ax10.set_xlabel('Space (x)')
    ax10.set_ylabel('Time (t)')
    divider = make_axes_locatable(ax10)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Relative Error')

    ax11 = fig.add_subplot(gs[1, 1])
    time_plot = t + 5.0
    ax11.plot(time_plot, time_relative_norm, color='crimson', lw=2)
    ax11.axvline(x=time_plot[-1], color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='evaluation stop')
    ax11.set_title('Time-Resolved Relative Error Norm')
    ax11.set_xlabel('Time (t + 5)')
    ax11.set_ylabel('Relative error')
    ax11.grid(True, alpha=0.3)
    mean_err = np.mean(time_relative_norm)
    ax11.text(0.5, 0.9, f'Mean: {mean_err:.2e}', transform=ax11.transAxes,
              ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

    ax20 = fig.add_subplot(gs[2, 0])
    im2 = ax20.imshow(
        true_spec.T, extent=[kx[0], kx[-1], t[0], t[-1]],
        origin='lower', aspect='auto', cmap='viridis')
    ax20.set_title('True Spatial Spectrum Over Time')
    ax20.set_xlabel('Spectral index n')
    ax20.set_ylabel('Time')
    divider2 = make_axes_locatable(ax20)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, label='Amplitude')

    ax21 = fig.add_subplot(gs[2, 1])
    im3 = ax21.imshow(
        rel_spec.T, extent=[kx[0], kx[-1], t[0], t[-1]],
        origin='lower', aspect='auto', cmap='inferno', vmin=0.0, vmax=1.0)
    ax21.set_title('Relative Spatial Spectrum Error')
    ax21.set_xlabel('Spectral index n')
    divider3 = make_axes_locatable(ax21)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, label='Relative Error')

    fig.suptitle(title)
    fig.subplots_adjust(top=0.95, bottom=0.03, hspace=0.35, wspace=0.28)
    fig.savefig(outpath, dpi=150)
    print(f'relative error summary saved to {outpath}')
    plt.close(fig)


# plot solution snapshots for several instants
def plot_solution_snapshots(x, t, eta_true, eta_pred, times, outdir, filename, title):
    # create snapshots of eta(x,t) and relative error at chosen time instants
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    times = np.array(times, dtype=float)
    t = np.asarray(t)
    indices = [int(np.argmin(np.abs(t - ti))) for ti in times]

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, idx in enumerate(indices):
        axes[row, 0].plot(x, eta_true[:, idx], label='True', color='black')
        axes[row, 0].plot(x, eta_pred[:, idx], label='Predicted', color='#ff7f0e', alpha=0.85)
        axes[row, 0].set_title(f'Time = {t[idx]:.2f}')
        axes[row, 0].set_ylabel('eta(x,t)')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        rel_error_line = compute_relative_error(eta_true[:, idx], eta_pred[:, idx])
        axes[row, 1].plot(x, rel_error_line, color='crimson')
        axes[row, 1].set_title(f'Relative Error at $t = {t[idx]:.2f}$')
        axes[row, 1].set_ylabel('Relative error')
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Space (x)')
    axes[-1, 1].set_xlabel('Space (x)')
    fig.suptitle(title)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98)
    plt.savefig(outpath, dpi=150)
    print(f'solution snapshots saved to {outpath}')
    plt.close()


# ---------------------------------------------------------------------------
# Shared geometry for the stacked 3D-surface panels (alpha/beta and resolution)
# ---------------------------------------------------------------------------
# The 3D scene is widened (aspectratio x, plus a scene domain spanning the whole
# left cell) so the surface print fills its column like the 2D plot beside it,
# instead of floating in the middle with white margins. Filling the cell pushes
# plotly's own y/z axis titles out of the domain, so we drop them from the scene
# and redraw 'η' and 'Space (x)' as paper-coord annotations; 'Time (t)' stays as
# the scene x-axis title. Height is unchanged, so the figure still fits the page.
_SURFACE_ASPECTRATIO = dict(x=3.6, y=1.85, z=0.62)
_SURFACE_SCENE_DOMAIN_X = [0.015, 0.60]
_SURFACE_AXIS_LABEL_PX = 22


def _surface_scene_annotations(n_scene_rows, vertical_spacing=0.08, font_size=_SURFACE_AXIS_LABEL_PX):
    # 'η' (height) and 'Space (x)' (depth) labels for each widened scene, placed
    # in paper coordinates at the left of its row. Rows are the n_scene_rows
    # surface rows plus one box/distribution row at the bottom, all equal height.
    n_rows = n_scene_rows + 1
    h = (1.0 - vertical_spacing * (n_rows - 1)) / n_rows
    # faint white pill so the labels stay legible whether they land on the white
    # margin ('η') or over the dark flat part of the widened surface ('Space (x)')
    bg = dict(bgcolor='rgba(255,255,255,0.65)', borderpad=1)
    anns = []
    for i in range(n_scene_rows):
        bottom = 1.0 - i * (h + vertical_spacing) - h
        anns.append(dict(text='η', x=0.005, y=bottom + h * 0.78,
                         xref='paper', yref='paper', showarrow=False,
                         xanchor='left', font=dict(size=font_size), **bg))
        anns.append(dict(text='Space (x)', x=0.02, y=bottom + h * 0.12,
                         xref='paper', yref='paper', showarrow=False,
                         xanchor='left', font=dict(size=font_size), **bg))
    return anns


def plot_model2_alpha_beta_panel(x, t, eta_true_list, eta_pred_list, param_values, outdir, filename, title,
                                 eval_resolution=None, res_label=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    param_values = np.asarray(param_values, dtype=float)
    n_params = len(param_values)
    if n_params == 0:
        raise ValueError('param_values must contain at least one value')

    time_rel_norms = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)

    z_global_min = min(float(np.asarray(e).min()) for e in eta_pred_list)
    z_global_max = max(float(np.asarray(e).max()) for e in eta_pred_list)
    z_margin = (z_global_max - z_global_min) * 0.05
    z_lim = [z_global_min - z_margin, z_global_max + z_margin]

    camera = dict(eye=dict(x=-1.5, y=-1.8, z=0.8), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
    scene_kw = dict(
        xaxis_title='Time (t)', yaxis_title='', zaxis_title='',
        zaxis=dict(range=z_lim), aspectmode='manual', aspectratio=_SURFACE_ASPECTRATIO,
        domain=dict(x=_SURFACE_SCENE_DOMAIN_X),
    )

    # ---------- main panel ----------
    specs = [[{'type': 'scene'}, {'type': 'xy'}]] * n_params + [[{'type': 'xy', 'colspan': 2}, None]]
    subplot_titles = []
    for alpha in param_values:
        subplot_titles += [f'Predicted η(x,t),  α=β={alpha:.2f}', f'Relative error,  α=β={alpha:.2f}']
    subplot_titles.append('Relative error distribution vs. α=β')

    fig = make_subplots(
        rows=n_params + 1, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[0.6, 0.4],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    S = _style_thesis(fig, 1900, frac=1.0)

    scene_layout = {}
    for i, (alpha, eta_pred) in enumerate(zip(param_values, eta_pred_list)):
        eta_pred = np.asarray(eta_pred, dtype=float)
        fig.add_trace(go.Surface(
            x=t, y=x, z=eta_pred,
            colorscale='Inferno', cmin=z_global_min, cmax=z_global_max,
            showscale=False,
        ), row=i + 1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=time_rel_norms[i],
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ), row=i + 1, col=2)
        fig.update_xaxes(title_text='Time (t)', row=i + 1, col=2)
        fig.update_yaxes(title_text='Relative error', row=i + 1, col=2)
        scene_key = 'scene' if i == 0 else f'scene{i + 1}'
        scene_layout[scene_key] = dict(**scene_kw, camera=camera)

    for alpha, rel_norm in zip(param_values, time_rel_norms):
        fig.add_trace(go.Box(
            y=rel_norm, name=f'α=β={alpha:.2f}',
            marker_color='#e6550d', fillcolor='#fdd0a2',
            line_color='#e6550d', showlegend=False,
            width=0.2,
        ), row=n_params + 1, col=1)
    fig.update_xaxes(title_text='α=β value', row=n_params + 1, col=1)
    fig.update_yaxes(title_text='Relative error', row=n_params + 1, col=1)

    fig.update_layout(
        width=1900, height=680 * n_params + 360,
        margin=dict(t=70, b=50, l=35, r=35),
        showlegend=False, **scene_layout,
    )
    # 3D scenes are small and qualitative: hide the perspective tick labels, which
    # otherwise pile up and bury the surface. The y/z axis titles are supplied as
    # paper-coord annotations (the widened scene clips the scene's own titles).
    fig.update_scenes(
        xaxis_title_font_size=_SURFACE_AXIS_LABEL_PX,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
    )
    fig.update_layout(annotations=list(fig.layout.annotations) + _surface_scene_annotations(n_params))
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'alpha/beta panel saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar panel como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_model2_resolution_panel(x_list, t_list, eta_true_list, eta_pred_list, resolutions, outdir, filename, title,
                                 eval_alpha_beta=None, param_label=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x_list = [np.asarray(x, dtype=float) for x in x_list]
    t_list = [np.asarray(t, dtype=float) for t in t_list]
    resolutions = np.asarray(resolutions, dtype=float)
    n_res = len(resolutions)
    if n_res == 0:
        raise ValueError('resolutions must contain at least one value')
    if len(x_list) != n_res or len(t_list) != n_res:
        raise ValueError('x_list and t_list must match the number of resolutions')

    time_rel_norms = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)

    z_global_min = min(float(np.asarray(e).min()) for e in eta_pred_list)
    z_global_max = max(float(np.asarray(e).max()) for e in eta_pred_list)
    z_margin = (z_global_max - z_global_min) * 0.05
    z_lim = [z_global_min - z_margin, z_global_max + z_margin]

    camera = dict(eye=dict(x=-1.5, y=-1.8, z=0.8), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
    scene_kw = dict(
        xaxis_title='Time (t)', yaxis_title='', zaxis_title='',
        zaxis=dict(range=z_lim), aspectmode='manual', aspectratio=_SURFACE_ASPECTRATIO,
        domain=dict(x=_SURFACE_SCENE_DOMAIN_X),
    )

    # ---------- main panel ----------
    specs = [[{'type': 'scene'}, {'type': 'xy'}]] * n_res + [[{'type': 'xy', 'colspan': 2}, None]]
    subplot_titles = []
    for res in resolutions:
        subplot_titles += [f'Predicted η(x,t),  res={int(res)}', f'Relative error,  res={int(res)}']
    subplot_titles.append('Relative error distribution vs. resolution')

    fig = make_subplots(
        rows=n_res + 1, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[0.6, 0.4],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    S = _style_thesis(fig, 1900, frac=1.0)

    scene_layout = {}
    for i, (res, x_res, t_res, eta_pred) in enumerate(zip(resolutions, x_list, t_list, eta_pred_list)):
        eta_pred = np.asarray(eta_pred, dtype=float)
        fig.add_trace(go.Surface(
            x=t_res, y=x_res, z=eta_pred,
            colorscale='Inferno', cmin=z_global_min, cmax=z_global_max,
            showscale=False,
        ), row=i + 1, col=1)
        fig.add_trace(go.Scatter(
            x=t_res, y=time_rel_norms[i],
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ), row=i + 1, col=2)
        fig.update_xaxes(title_text='Time (t)', row=i + 1, col=2)
        fig.update_yaxes(title_text='Relative error', row=i + 1, col=2)
        scene_key = 'scene' if i == 0 else f'scene{i + 1}'
        scene_layout[scene_key] = dict(**scene_kw, camera=camera)

    for res, rel_norm in zip(resolutions, time_rel_norms):
        fig.add_trace(go.Box(
            y=rel_norm, name=str(int(res)),
            marker_color='#e6550d', fillcolor='#fdae6b',
            line_color='#e6550d', showlegend=False,
            width=0.2,
        ), row=n_res + 1, col=1)
    fig.update_xaxes(title_text='Resolution', row=n_res + 1, col=1)
    fig.update_yaxes(title_text='Relative error', row=n_res + 1, col=1)

    fig.update_layout(
        width=1900, height=680 * n_res + 360,
        margin=dict(t=70, b=50, l=35, r=35),
        showlegend=False, **scene_layout,
    )
    # 3D scenes are small and qualitative: hide the perspective tick labels, which
    # otherwise pile up and bury the surface. The y/z axis titles are supplied as
    # paper-coord annotations (the widened scene clips the scene's own titles).
    fig.update_scenes(
        xaxis_title_font_size=_SURFACE_AXIS_LABEL_PX,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
    )
    fig.update_layout(annotations=list(fig.layout.annotations) + _surface_scene_annotations(n_res))
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'resolution panel saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar panel como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_model2_spectral_panel(x, t, eta_true, eta_pred, outdir, filename, title,
                               n_times=3, eval_resolution=None, res_label=None, param_label=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    kx = spectral_mode_index(x)
    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    rel_err = compute_relative_error(true_spec, pred_spec)
    mean_rel_err_over_time = np.mean(rel_err, axis=0)

    if len(t) < n_times:
        indices = list(np.arange(len(t), dtype=int))
        target_times = list(t)
    else:
        target_times_arr = np.linspace(t[0], t[-1], n_times)
        selected = []
        seen = set()
        for target in target_times_arr:
            idx = int(np.argmin(np.abs(t - target)))
            if idx not in seen:
                seen.add(idx)
                selected.append((idx, float(target)))
        indices, target_times = zip(*selected) if selected else ([], [])
        indices = list(indices)
        target_times = list(target_times)

    def _tl(tt):
        return f'{int(round(tt))}' if abs(tt - round(tt)) < 1e-6 else f'{tt:.2f}'

    n_snap = len(indices)
    n_rows = n_snap + 1
    specs = [[{'type': 'xy'}, {'type': 'xy'}]] * n_snap + [[{'type': 'xy', 'colspan': 2}, None]]
    subplot_titles = []
    for idx, tt in zip(indices, target_times):
        tl = _tl(tt)
        subplot_titles += [f'Spectrum at t = {tl}', f'Relative spectral error at t = {tl}']
    subplot_titles += ['Mean relative spectral error over time', '']

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )
    S = _style_thesis(fig, 1400, frac=1.0)

    for row, (idx, tt) in enumerate(zip(indices, target_times)):
        tl = _tl(tt)
        show_legend = (row == 0)
        fig.add_trace(go.Scatter(
            x=kx.tolist(), y=true_spec[:, idx].tolist(),
            mode='lines+markers', name='True', showlegend=show_legend,
            line=dict(color='#222222', width=2.0),
            marker=dict(size=3, symbol='circle'),
        ), row=row + 1, col=1)
        fig.add_trace(go.Scatter(
            x=kx.tolist(), y=pred_spec[:, idx].tolist(),
            mode='lines+markers', name='Predicted', showlegend=show_legend,
            line=dict(color='#ff7f0e', width=1.8, dash='dash'),
            marker=dict(size=3, symbol='diamond'),
        ), row=row + 1, col=1)
        fig.update_xaxes(title_text='Spectral index n', row=row + 1, col=1)
        fig.update_yaxes(title_text='Amplitude', row=row + 1, col=1)

        mean_rel = float(mean_rel_err_over_time[idx])
        fig.add_trace(go.Scatter(
            x=kx.tolist(), y=rel_err[:, idx].tolist(),
            mode='lines+markers', showlegend=False,
            line=dict(color='crimson', width=1.8),
            marker=dict(size=3, symbol='circle'),
        ), row=row + 1, col=2)
        fig.add_annotation(
            text=f'Mean: {mean_rel:.2e}',
            xref=f'x{(row * 2 + 2) if row > 0 else "2"} domain', yref=f'y{(row * 2 + 2) if row > 0 else "2"} domain',
            x=0.97, y=0.97, xanchor='right', yanchor='top',
            showarrow=False, font=dict(size=S['annot']),
            bgcolor='white', bordercolor='#cccccc', borderwidth=1, borderpad=4,
        )
        fig.update_xaxes(title_text='Spectral index n', row=row + 1, col=2)
        fig.update_yaxes(title_text='Relative error', row=row + 1, col=2)

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=mean_rel_err_over_time.tolist(),
        mode='lines+markers', showlegend=False,
        line=dict(color='#c28b00', width=2.0),
        marker=dict(size=5),
    ), row=n_rows, col=1)
    fig.update_xaxes(title_text='Time (t)', row=n_rows, col=1)
    fig.update_yaxes(title_text='Mean relative error', row=n_rows, col=1)

    fig.update_layout(
        width=1400, height=460 * n_snap + 300,
        margin=dict(t=50, b=55, l=60, r=30),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'spectral panel saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar spectral panel como PNG. Instale kaleido: pip install kaleido')
        raise e


def plot_error_heatmap(x, t, eta_true, eta_pred, outdir, filename, title):
    # render a heatmap of relative error over space and time
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    rel_diff = compute_relative_error(eta_true, eta_pred)
    rel_diff = np.clip(rel_diff, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        rel_diff.T,
        extent=[x[0], x[-1], t[0], t[-1]],
        origin='lower',
        aspect='auto',
        cmap='inferno',
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Relative Error')
    fig.subplots_adjust(right=0.88)
    plt.savefig(outpath, dpi=150)
    print(f'error heatmap saved to {outpath}')
    plt.close()


def plot_spectral_bias_evolution(ordered_metadata, outdir, filename='spectral_bias_evolution.png'):
    models_data = []
    for label, mf in ordered_metadata:
        if not os.path.exists(mf):
            print(f'  metadata not found: {mf}, skipping')
            continue
        md = torch.load(mf, map_location='cpu')
        sh = md.get('spectral_history')
        if sh is None or not sh.get('epochs'):
            print(f'  no spectral_history in {mf}, skipping (retrain to populate)')
            continue
        models_data.append((label, sh))
    if not models_data:
        print('  no spectral history data available - retrain models first')
        return
    plot_spectral_bias_panel(models_data, outdir=outdir, filename=filename)


def plot_spectral_bias_panel(models_data, outdir, filename='spectral_bias_evolution.png'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    band_colors = ['#1f77b4', '#9467bd', '#d62728']
    band_labels = ['Low freq (k < Nx/4)', 'Mid freq (Nx/4 <= k < Nx/2)', 'High freq (k >= Nx/2)']
    band_keys = ['low_band', 'mid_band', 'high_band']

    n = len(models_data)
    n_top = n - 1
    n_pair_rows = (n_top + 1) // 2 if n_top > 0 else 0
    n_rows = n_pair_rows + 1

    specs = []
    for r in range(n_pair_rows):
        has_right = (r * 2 + 1) < n_top
        specs.append([{'type': 'xy'}, {'type': 'xy'} if has_right else None])
    specs.append([{'type': 'xy', 'colspan': 2}, None])

    subplot_titles = []
    for r in range(n_pair_rows):
        li, ri = r * 2, r * 2 + 1
        subplot_titles.append(models_data[li][0] if li < n_top else '')
        subplot_titles.append(models_data[ri][0] if ri < n_top else '')
    subplot_titles += [models_data[-1][0], '']

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    _style_thesis(fig, 1400, frac=1.0)

    for i in range(n_top):
        row = i // 2 + 1
        col = i % 2 + 1
        label, sh = models_data[i]
        epochs = np.asarray(sh['epochs']).tolist()
        for color, blab, key in zip(band_colors, band_labels, band_keys):
            vals = np.asarray(sh[key]).tolist()
            fig.add_trace(go.Scatter(
                x=epochs, y=vals, mode='lines', name=blab,
                line=dict(color=color, width=2.0), opacity=0.9,
                showlegend=(i == 0),
            ), row=row, col=col)
        fig.update_yaxes(type='log', row=row, col=col)
        fig.update_xaxes(title_text='Epoch', row=row, col=col)
        fig.update_yaxes(title_text='Rel. spectral error', row=row, col=col)

    last_label, last_sh = models_data[-1]
    epochs = np.asarray(last_sh['epochs']).tolist()
    for color, blab, key in zip(band_colors, band_labels, band_keys):
        vals = np.asarray(last_sh[key]).tolist()
        fig.add_trace(go.Scatter(
            x=epochs, y=vals, mode='lines', name=blab,
            line=dict(color=color, width=2.0), opacity=0.9,
            showlegend=False,
        ), row=n_rows, col=1)
    fig.update_yaxes(type='log', row=n_rows, col=1)
    fig.update_xaxes(title_text='Epoch', row=n_rows, col=1)
    fig.update_yaxes(title_text='Rel. spectral error', row=n_rows, col=1)

    fig.update_layout(
        width=1400, height=480 * n_rows,
        margin=dict(t=50, b=45, l=60, r=30),
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'spectral bias panel saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar spectral bias panel como PNG. Instale kaleido: pip install kaleido')
        raise e


def save_solution_surface_3d_html(x, t, eta_true, eta_pred, outdir, filename, title):
    # plotly interactive html: side-by-side 3d surface of reference and predicted eta(x,t)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    outpath = os.path.join(outdir, base + '.html')

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    z_min = min(float(eta_true.min()), float(eta_pred.min()))
    z_max = max(float(eta_true.max()), float(eta_pred.max()))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Reference', 'Prediction'],
        horizontal_spacing=0.05,
    )

    surface_kw = dict(x=x.tolist(), y=t.tolist(), colorscale='Viridis', cmin=z_min, cmax=z_max)
    fig.add_trace(go.Surface(z=eta_true.T.tolist(), showscale=False, **surface_kw), row=1, col=1)
    fig.add_trace(go.Surface(z=eta_pred.T.tolist(), colorbar=dict(title='eta(x,t)', x=1.02), **surface_kw), row=1, col=2)

    camera = dict(eye=dict(x=1.8, y=-1.8, z=1.2))
    scene_common = dict(xaxis_title='Space (x)', yaxis_title='Time (t)', zaxis_title='eta(x,t)', camera=camera)
    fig.update_layout(
        title_text=title, title_x=0.5,
        width=1400, height=650,
        scene=scene_common, scene2=scene_common,
    )
    fig.write_html(outpath)
    print(f'solution surface html saved to {outpath}')


def save_solution_plotly_html(x, t, eta_true, eta_pred, outdir, filename, title):
    # plotly interactive html: reference + predicted eta(x,t) heatmaps + relative error
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    outpath = os.path.join(outdir, base + '.html')

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    rel_error = compute_relative_error(eta_true, eta_pred)
    rel_error = np.clip(rel_error, 0.0, 1.0)
    time_rel_norm = (
        np.linalg.norm(np.abs(eta_true - eta_pred), axis=0)
        / (np.linalg.norm(eta_true, axis=0) + 1e-8)
    )

    z_min = min(float(eta_true.min()), float(eta_pred.min()))
    z_max = max(float(eta_true.max()), float(eta_pred.max()))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Reference eta(x,t)', 'Predicted eta(x,t)', 'Relative Error', 'Error Norm Over Time'],
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )

    heatmap_kw = dict(x=x.tolist(), y=t.tolist(), colorscale='Viridis', zmin=z_min, zmax=z_max)
    fig.add_trace(go.Heatmap(z=eta_true.T.tolist(), showscale=False, **heatmap_kw), row=1, col=1)
    fig.add_trace(go.Heatmap(z=eta_pred.T.tolist(), colorbar=dict(title='eta', x=1.02, len=0.45, y=0.78), **heatmap_kw), row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=rel_error.T.tolist(), x=x.tolist(), y=t.tolist(),
        colorscale='Magma', zmin=0.0, zmax=1.0,
        colorbar=dict(title='Rel. err', x=1.02, len=0.45, y=0.22),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=time_rel_norm.tolist(),
        mode='lines', line=dict(color='crimson', width=2), name='error norm',
    ), row=2, col=2)

    fig.update_xaxes(title_text='Space (x)', row=1)
    fig.update_xaxes(title_text='Space (x)', row=2, col=1)
    fig.update_xaxes(title_text='Time (t)', row=2, col=2)
    fig.update_yaxes(title_text='Time (t)', col=1)
    fig.update_yaxes(title_text='Relative error', row=2, col=2)
    fig.update_layout(title_text=title, title_x=0.5, width=1200, height=900, showlegend=False)

    fig.write_html(outpath)
    print(f'solution html saved to {outpath}')


def save_solution_gif(x, t, eta_true, eta_pred, outdir, filename, title_prefix, fps=20):
    # animate eta(x, t) over time: reference (gray) vs prediction (orange line),
    # with the evolving time value shown in the title. saves a .gif via matplotlib.
    from matplotlib.animation import FuncAnimation, PillowWriter

    _ensure_outdir(outdir)
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    outpath = os.path.join(outdir, base + '.gif')

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    # eta_* are [space, time]; iterate over the (common) time axis
    n_frames = int(min(eta_true.shape[1], eta_pred.shape[1], t.shape[0]))

    y_min = float(min(eta_true.min(), eta_pred.min()))
    y_max = float(max(eta_true.max(), eta_pred.max()))
    pad = 0.08 * (y_max - y_min + 1e-9)
    y_min, y_max = y_min - pad, y_max + pad

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    line_true, = ax.plot([], [], color='0.55', lw=2.0, label='Referência (pseudoespectral)')
    line_pred, = ax.plot([], [], color='#E66C11', lw=2.4, label='Predição')
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\eta(x,t)$')
    ax.legend(loc='upper right', fontsize=9)
    # reserve room at the top so the per-frame title is never clipped
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.13)
    title = ax.set_title(f'{title_prefix},   t = {t[0]:.2f}', color='black')

    def _init():
        line_true.set_data([], [])
        line_pred.set_data([], [])
        return line_true, line_pred

    def _update(n):
        line_true.set_data(x, eta_true[:, n])
        line_pred.set_data(x, eta_pred[:, n])
        title.set_text(f'{title_prefix},   t = {t[n]:.2f}')
        return line_true, line_pred

    anim = FuncAnimation(fig, _update, frames=n_frames, init_func=_init, blit=False)
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'solution gif saved to {outpath}')


def plot_pinn_ntk_panel(theta_rel_histories, k_rel_histories, log_epochs,
                         eigenvalues_init, eigenvalues_final, widths, outdir, filename):
    # NTK diagnostic rendered as THREE standalone figures (parameter drift, kernel
    # drift, eigenvalue spectrum) instead of one wide three-column panel, so each
    # can be \includegraphics'd on its own with legible, print-sized fonts.
    import plotly.graph_objects as go

    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    _ensure_outdir(outdir)
    base, _, ext = filename.rpartition('.')
    base = base or filename
    ext = ext or 'png'

    W, H, FRAC = 860, 560, 0.48

    def _finish(fig, title, xlab, ylab, suffix, logx=False, logy=False):
        fig.update_layout(
            xaxis_title=xlab, yaxis_title=ylab,
            width=W, height=H, showlegend=True,
            margin=dict(t=30, b=45, l=60, r=20),
            legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
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