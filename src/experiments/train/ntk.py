"""The spectral-bias probe: MLPs of three widths fitting one Boussinesq profile.

The target is the pseudospectral reference eta(x, t_slice) at alpha = beta =
param_value on N_x grid points, fitted by full-batch gradient descent. The probe
measures the frozen-kernel predictions of Section 2 against a real (nonlinear)
network:

    e^{(n)} = (I - mu K_0)^n e_0                                (eq:error_discrete)
    n_k     = ln(|c_k(0)| / eps) / (-ln|1 - mu lambda_k|)       (discrete crossing time;
                                                                mu*lambda_k limit = eq:iteration_count)

with K_0 = J_0 J_0^T the empirical NTK, V Lambda V^T its eigendecomposition and
c_k = v_k^T e.

This module runs the probe and persists it; `experiments.plots.ntk` draws every
figure from the persisted file alone. The run costs ~1.6 h at 500k iterations, so
the split is what lets the figures be iterated on without repeating it. Anything
a figure consumes therefore has to reach the `.pth` here -- see the `arrays`
payload assembled at the end.
"""

import os

import numpy as np
import torch

from methods.mlp import MLP
from tools import (
    atomic_torch_save, band_spectral_error, error_spectrum, save_metadata_json,
    save_model, set_seed, to_symmetric,
)
from experiments.common import reference_solution, resolve_device, stage_paths

LABEL = 'mlp_ntk'


def _log_schedule(n_iterations, count):
    """`count` log-spaced iterations in [0, n_iterations], both ends included."""
    n_iterations = int(n_iterations)
    if n_iterations <= 0:
        return np.array([0], dtype=np.int64)
    marks = np.geomspace(1, n_iterations, max(int(count), 2)).astype(np.int64)
    return np.unique(np.concatenate(([0], marks, [n_iterations])))


def _reference_slice(x_limit, t_limit, param_value, n_x, t_slice, device, amplitude=1.0):
    """eta(x, t_slice) on n_x points, under the shared [-1, 1] convention.

    Both the coordinate and the target go through `tools.to_symmetric`: the
    coordinate on the domain envelope [-L, L], the target on the amplitude
    envelope [-A, A], exactly as the MLP and the operators use them. The
    spectral-bias quantities are invariant to the target scaling (the NTK is set
    by the network, and the crossing threshold eps scales with the target), and
    the fit figures undo it to display physical eta.
    """
    x_sol, t_sol, eta_sol, _ = reference_solution(
        param_value, x_limit, t_limit, nx=n_x, nt=n_x, device=device, amplitude=amplitude)
    t_all = torch.as_tensor(t_sol, dtype=torch.float64, device=device)
    j = int(torch.argmin((t_all - t_slice).abs()))
    a = float(amplitude)
    target = to_symmetric(torch.as_tensor(eta_sol, dtype=torch.float64, device=device)[j], -a, a)
    x = torch.as_tensor(x_sol, dtype=torch.float64, device=device)
    return to_symmetric(x, -x_limit, x_limit).reshape(-1, 1), target, x, float(t_all[j])


def _empirical_ntk(model, inputs):
    """K = J J^T in R^{N_x x N_x}, with J[i] = d f(x_i) / d theta.

    Assembled by one backward pass per grid point, so this costs N_x backward
    passes per call; callers rebuild it on a sparse schedule.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    out = model(inputs).reshape(-1)
    n_x, n_theta = out.shape[0], sum(p.numel() for p in params)
    # J is filled row by row rather than stacked from a list, so only one copy of
    # the 128 x N_theta Jacobian is ever resident
    J = torch.empty((n_x, n_theta), dtype=out.dtype, device=out.device)
    for i in range(n_x):
        grads = torch.autograd.grad(out[i], params, retain_graph=(i < n_x - 1))
        torch.cat([g.reshape(-1) for g in grads], out=J[i])
    K = J @ J.T
    del J
    return K


def _spectrum(K):
    """Descending eigenpairs (lambda, V) of a symmetric PSD kernel."""
    evals, evecs = torch.linalg.eigh(K)
    order = torch.argsort(evals, descending=True)
    return evals[order].clamp(min=0.0), evecs[:, order]


def _trusted_spectrum(ev, floor_rel):
    """Eigenvalues above floor_rel * lambda_max.

    Below that the eigenvalues of a double-precision Gram matrix are round-off:
    they stop decaying and flatten onto a plateau at ~1.5e-16 * lambda_max.
    Plotting the plateau draws a cliff that looks like physics but is not.
    """
    if ev.numel() == 0 or ev[0] <= 0:
        return ev
    return ev[ev > ev[0] * floor_rel]


def _train_one(model, inputs, target, mu, n_iterations, sample_iters, kernel_iters,
               K0, theta0, V_track, eps):
    """Gradient descent on 1/2 ||e||^2, so one step is theta <- theta - mu J^T e.

    Returns (e at `sample_iters`, theta drift at `kernel_iters`, kernel drift at
    `kernel_iters`, first iteration at which each tracked |c_k| < eps).

    Crossings are detected every iteration on device, so eq:iteration_count keeps
    exact resolution without a per-iteration history buffer.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=mu)
    theta0_norm = theta0.norm() + 1e-30
    K0_norm = K0.norm() + 1e-30
    sample_set = {int(n) for n in sample_iters}
    kernel_set = {int(n) for n in kernel_iters}

    e_samples, theta_rel, k_rel = [], [], []
    first_cross = torch.full((V_track.shape[1],), -1, dtype=torch.int64,
                             device=inputs.device)

    for n in range(n_iterations + 1):
        e = model(inputs).reshape(-1) - target

        c = (e.detach() @ V_track).abs()               # c_k^{(n)} = v_k^T e^{(n)}
        first_cross = torch.where((c < eps) & (first_cross < 0),
                                  torch.full_like(first_cross, n), first_cross)

        if n in sample_set:
            e_samples.append(e.detach().clone())
        if n in kernel_set:
            theta_n = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            theta_rel.append(float((theta_n - theta0).norm() / theta0_norm))
            k_rel.append(float((_empirical_ntk(model, inputs) - K0).norm() / K0_norm))

        loss = 0.5 * torch.sum(e ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.stack(e_samples, dim=0), theta_rel, k_rel, first_cross.cpu().numpy()


def _run_width(model, inputs, target, mu_frac, n_iterations, sample_iters, kernel_iters,
               tolerance, eig_floor_rel):
    """Kernel, training run and per-mode bookkeeping for one network.

    The returned dict is exactly what the plotting stage reads back: it is stored
    verbatim (tensors moved to host) and reconstructed key for key, so a figure
    cannot tell a reloaded run from a live one.
    """
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    K0 = _empirical_ntk(model, inputs)
    lam, V = _spectrum(K0)
    ev_ok = _trusted_spectrum(lam, eig_floor_rel)

    # mu is a fixed fraction of each network's own stability bound mu < 2/lambda_max.
    # lambda_max grows with width, so one nominal mu would place the narrow and wide
    # networks at different fractions of their bounds and the runs would not be
    # comparable. mu_frac stays below 0.5, where 1 - mu*lambda_1 = 0 would predict a
    # one-step annihilation no nonlinear network reproduces.
    mu = float(mu_frac) * 2.0 / float(lam[0])

    e0 = model(inputs).reshape(-1).detach() - target
    c0 = V.T @ e0
    # eps is a fraction of the largest initial coefficient, so the threshold scales
    # with the amplitude of eta rather than with its units
    eps = float(tolerance) * float(c0.abs().max())

    n_track = int(ev_ok.numel())
    V_track = V[:, :n_track].contiguous()

    e_meas, theta_rel, k_rel, first_cross = _train_one(
        model, inputs, target, mu, n_iterations, sample_iters, kernel_iters,
        K0, theta0, V_track, eps)
    ev_final = _trusted_spectrum(_spectrum(_empirical_ntk(model, inputs))[0], eig_floor_rel)

    # eq:error_discrete evaluated at the sampled iterations, in the eigenbasis
    n_axis = torch.as_tensor(np.asarray(sample_iters, dtype=np.float64),
                             dtype=torch.float64, device=e_meas.device)
    C_pred = c0.unsqueeze(0) * (1.0 - mu * lam).unsqueeze(0) ** n_axis.unsqueeze(1)
    C_meas = e_meas @ V
    e_pred = C_pred @ V.T

    meas_n, theo_n = [], []
    for k in range(n_track):
        if lam[k] <= 0.0 or float(c0[k].abs()) <= eps or first_cross[k] < 0:
            continue
        meas_n.append(int(first_cross[k]))
        # exact discrete crossing of |c_k(0)| |1 - mu lambda_k|^n = eps, so the count
        # uses the same (1 - mu lambda_k)^n law as the decay panel; the continuous
        # mu*lambda_k denominator of eq:iteration_count is its small-step limit
        theo_n.append(float(np.log(float(c0[k].abs()) / eps)
                            / -np.log(abs(1.0 - mu * float(lam[k])))))

    return dict(
        lam=lam, mu=mu, ev_ok=ev_ok, ev_final=ev_final,
        sample_iters=np.asarray(sample_iters, dtype=np.int64),
        kernel_iters=np.asarray(kernel_iters, dtype=np.int64),
        e_meas=e_meas, e_pred=e_pred, C_meas=C_meas, C_pred=C_pred, c0=c0,
        eta_fit=(e_meas[-1] + target).detach(),
        theta_rel=theta_rel, k_rel=k_rel,
        eps=eps, meas_n=meas_n, theo_n=theo_n,
        n_theta=int(sum(p.numel() for p in model.parameters())),
    )


def _to_host(run):
    """One run's arrays, detached onto the host, ready for torch.save."""
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in run.items()}


def train_ntk(x_limit, t_limit, param_value, results_dir, n_x=128, t_slice=15.0,
              widths=None, hidden_layers=4, activation='tanh', mu_frac=0.2,
              n_iterations=500000, tolerance=0.05, reference_width=512,
              eig_floor_rel=1e-15, kernel_log_points=60, sample_log_points=600,
              amplitude=1.0, seed=37):
    """Run the probe at every width, save weights and metadata, return the metrics.

    Instrumentation is log-spaced in n because every panel is read on a log axis:
    `kernel_log_points` NTK rebuilds (N_x backward passes each) and
    `sample_log_points` stored error vectors. `eig_floor_rel` is the relative
    eigenvalue floor of _trusted_spectrum.
    """
    widths = list(widths) if widths is not None else [8, 64, 512]

    # reseeded locally: the probe trains its own networks and must reproduce
    # independently of what earlier pipeline stages drew from the RNG
    set_seed(seed)
    weights_dir, metadata_dir = stage_paths(results_dir, os.path.join('mlp', 'ntk'))

    device = resolve_device()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)   # the small eigenvalues need float64

    sample_iters = _log_schedule(n_iterations, sample_log_points)
    kernel_iters = _log_schedule(n_iterations, kernel_log_points)

    try:
        inputs, target, x, t_used = _reference_slice(
            x_limit, t_limit, param_value, n_x, t_slice, device, amplitude)
        print(f'spectral-bias probe: Boussinesq eta(x, t={t_used:g}) at '
              f'alpha=beta={param_value}, {n_x} grid points, widths {widths}, '
              f'{hidden_layers} hidden layers, {n_iterations:,} iterations, '
              f'mu = {mu_frac:g} * 2/lambda_max')
        print(f'  instrumentation: {len(sample_iters)} error snapshots, '
              f'{len(kernel_iters)} kernel rebuilds')

        results = {}
        for w in widths:
            model = MLP(input_size=1, output_size=1, neurons=w,
                        hidden_layers=hidden_layers, activation=activation, device=device)
            print(f'  width={w}: {sum(p.numel() for p in model.parameters()):,} parameters...')
            res = _run_width(model, inputs, target, mu_frac, n_iterations,
                             sample_iters, kernel_iters, tolerance, eig_floor_rel)
            lam_max = float(res['lam'][0])
            print(f'    lambda_max={lam_max:.4g}, mu={res["mu"]:.4g}, '
                  f'mu*lambda_max={res["mu"] * lam_max:.4g}, '
                  f'tau={res["mu"] * n_iterations:.4g}, '
                  f'modes above floor={res["ev_ok"].numel()}, crossed={len(res["theo_n"])}, '
                  f'theta_drift={res["theta_rel"][-1]:.4g}, k_drift={res["k_rel"][-1]:.4g}, '
                  f'final rel. error={float(res["e_meas"][-1].norm()) / float(target.norm()):.4f}')
            results[w] = res
            save_model(model, os.path.join(weights_dir, f'{LABEL}_w{w}_weights.pth'))
            del model

        ref_w = reference_width if reference_width in results else widths[-1]
        target_norm = float(target.norm())
        target_spectrum = torch.fft.rfft(target).abs() / target.numel()

        metrics = {
            'model': 'MLP-NTK probe',
            'label': LABEL,
            'param_value': float(param_value),
            't_slice': t_used,
            'n_x': int(n_x),
            'hidden_layers': int(hidden_layers),
            'activation': activation,
            # mu is per width (mu_frac * 2/lambda_max), so only the fraction is a
            # global constant; the resolved step is under metrics['widths'][w]['mu']
            'mu_frac': float(mu_frac),
            'mu_rule': 'mu = mu_frac * 2 / lambda_max, per width',
            'n_iterations': int(n_iterations),
            'tolerance': float(tolerance),
            'eig_floor_rel': float(eig_floor_rel),
            'reference_width': int(ref_w),
            'n_error_snapshots': int(len(sample_iters)),
            'n_kernel_rebuilds': int(len(kernel_iters)),
            'target_norm': target_norm,
            'seed': int(seed),
            'widths': {},
        }
        for w in widths:
            r = results[w]
            t_arr = np.asarray(r['theo_n'], dtype=np.float64)
            m_arr = np.maximum(np.asarray(r['meas_n'], dtype=np.float64), 1.0)
            band_meas = band_spectral_error(error_spectrum(r['e_meas'] + target), target_spectrum)
            metrics['widths'][str(w)] = {
                'width': int(w),
                'hidden_layers': int(hidden_layers),
                'num_params': r['n_theta'],
                'lambda_max': float(r['lam'][0]),
                'mu': float(r['mu']),
                'mu_lambda_max': float(r['mu']) * float(r['lam'][0]),
                'tau_final': float(r['mu']) * int(n_iterations),
                'n_modes_above_floor': int(r['ev_ok'].numel()),
                'eigenvalue_decades': float(np.log10(float(r['ev_ok'][0] / r['ev_ok'][-1]))),
                'eps': r['eps'],
                'n_modes_crossed': int(t_arr.size),
                'iteration_count_log_corr': (float(np.corrcoef(np.log10(t_arr), np.log10(m_arr))[0, 1])
                                             if t_arr.size > 2 else None),
                'iteration_count_median_ratio': (float(np.median(m_arr / t_arr))
                                                 if t_arr.size else None),
                'measured_crossings': [int(v) for v in r['meas_n']],
                'theoretical_crossings': [float(v) for v in r['theo_n']],
                'final_theta_rel': r['theta_rel'][-1],
                'final_k_rel': r['k_rel'][-1],
                'initial_error_norm': float(r['e_meas'][0].norm()),
                'final_error_norm': float(r['e_meas'][-1].norm()),
                'final_relative_error': float(r['e_meas'][-1].norm()) / target_norm,
                'band_error_first': [float(b[0]) for b in band_meas],
                'band_error_final': [float(b[-1]) for b in band_meas],
            }

        # everything the plotting stage consumes. The per-width entries are the
        # `_run_width` dicts verbatim, so the plotter needs no translation layer.
        arrays = {
            'x': x.cpu(),
            'target': target.cpu(),          # normalized (eta / A); figures undo it
            'amplitude': float(amplitude),
            't_slice': t_used,
            'widths': [int(w) for w in widths],
            'sample_iters': sample_iters,
            'kernel_iters': kernel_iters,
            'runs': {str(w): _to_host(results[w]) for w in widths},
        }
    finally:
        torch.set_default_dtype(prev_dtype)

    metadata_file = atomic_torch_save({**metrics, 'arrays': arrays},
                                      os.path.join(metadata_dir,
                                                   f'{LABEL}_model_metadata.pth'))
    print(f'MLP-NTK probe metadata saved to {metadata_file}')
    save_metadata_json(metrics, os.path.join(metadata_dir, f'{LABEL}_metadata.json'))
    return metrics
