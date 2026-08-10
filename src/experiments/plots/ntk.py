"""Figures for the spectral-bias probe, drawn from its saved metadata alone.

Nothing here touches a live model or any training state: the entry point takes
the `.pth` written by `experiments.train.ntk` and emits every panel. The probe
run costs ~1.6 h at 500k iterations, so this separation is what lets the figures
be reworked without repeating it.

Emits, into `outdir`:
    ntk_error_decay.png      per-eigendirection decay against eq:error_discrete
    ntk_spectral_decay.png   the same in Fourier space, at fixed wavenumbers
    ntk_band_decay.png       low/mid/high band averages of eq:m_bands
    ntk_iteration_count.png  measured vs predicted n_k of eq:iteration_count
    ntk_fit.png              final output at every width vs the reference
    ntk_fit_evolution.png    one run's output at log-spaced checkpoints
and into `drift_outdir`, three panels named `{filename stem}_{param,kernel,spectrum}`.
"""

import os

import numpy as np
import torch

from tools import band_relative_error, error_spectrum, save_metadata_json
from experiments.plots.figures import THESIS_BAND_COLORS, THESIS_PALETTE
from experiments.plots.figures.style import ensure_outdir, save_thesis_fig

_PALETTE = THESIS_PALETTE

# included at \linewidth inside 0.48\textwidth subfigures; frac is raised so the
# printed in-figure text stays around 7 pt
_W, _H, _FRAC = 820, 470, 0.48   # match the SciML pair panels: same printed font size
_LEGEND_ABOVE = dict(orientation='h', xanchor='center', x=0.5, yanchor='bottom', y=1.02)
_MARGIN = dict(t=46, b=62, l=96, r=20)   # margins scale with the print font size


def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def _marker_positions(iters, count=45):
    """Indices into `iters` at which the closed form is overlaid as open markers."""
    if len(iters) <= count:
        return np.arange(len(iters))
    return np.unique(np.geomspace(1, len(iters), count).astype(int) - 1)


def _decay_spread_indices(lam, mu, n_iterations, count=4, n_decay_min=2.0,
                          window_slack=8.0):
    """Eigendirections whose decay times 1/(mu lambda_k) are log-spaced over the run.

    The slack past `n_iterations` keeps one tracked direction that starts to decay
    without finishing, which is what spectral bias looks like at the slow end.
    """
    n_decay = 1.0 / np.maximum(mu * _np(lam), 1e-300)
    usable = np.where((n_decay > n_decay_min) & (n_decay < window_slack * n_iterations))[0]
    if usable.size == 0:
        usable = np.where((n_decay > 1.0) & (n_decay < n_iterations))[0]
    if usable.size == 0:
        return list(range(1, min(count + 1, lam.numel())))
    picked = []
    for tgt in np.geomspace(n_decay[usable].min(), n_decay[usable].max(), count):
        k = int(usable[np.argmin(np.abs(n_decay[usable] - tgt))])
        if k not in picked:
            picked.append(k)
    return picked


def _log_range(curves, decades=9.0, pad_hi=0.35):
    """log10 y-range clamped to `decades`; the prediction underflows and would
    otherwise set the axis and squash every measured curve into the top pixels."""
    hi = max(float(np.max(_np(c))) for c in curves if np.size(_np(c)))
    return [np.log10(max(hi, 1e-300)) - decades, np.log10(max(hi, 1e-300)) + pad_hi]


def _band_curves(res, target):
    """Band-averaged E_spec (eq:m_rel_spec) of the measured and the frozen-kernel
    predicted field, each against the reference. The fields are the errors put
    back on the target, e + target, so the metric matches the six-model bands."""
    ref = torch.fft.rfft(target).abs()
    pred_meas = error_spectrum(res['e_meas'] + target)   # |rfft(measured field)|
    pred_theo = error_spectrum(res['e_pred'] + target)   # |rfft(frozen-kernel field)|
    return band_relative_error(pred_meas, ref), band_relative_error(pred_theo, ref)


def _decay_figure(res, series, ylabel, outpath, y_range=None):
    """Measured lines with the closed form overlaid as sparse open markers."""
    import plotly.graph_objects as go

    iters = np.maximum(res['sample_iters'], 1)
    marks = _marker_positions(res['sample_iters'])

    fig = go.Figure()
    for j, (name, measured, predicted, color) in enumerate(series):
        color = color or _PALETTE[j % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=iters.tolist(), y=np.asarray(measured).tolist(),
            mode='lines', name=name, line=dict(color=color, width=2.4),
        ))
        # the prediction is kept out of the legend: it shares the band's colour and
        # is told apart by being open markers rather than a line, which every
        # caption using these panels already states. Listing it doubles the legend
        # onto three rows and takes a third of the panel height.
        fig.add_trace(go.Scatter(
            x=iters[marks].tolist(), y=np.asarray(predicted)[marks].tolist(),
            mode='markers', name=f'{name} theory', showlegend=False,
            marker=dict(color=color, size=5, symbol='circle-open', line=dict(width=1.4)),
        ))
    fig.update_xaxes(title_text='Iteration n  (&#964; = n&#956;)', type='log')
    fig.update_yaxes(title_text=ylabel, type='log',
                     **({'range': y_range} if y_range else {}))
    save_thesis_fig(fig, outpath, _W, _H, _FRAC,
                    extra_layout=dict(margin=_MARGIN, showlegend=True,
                                      legend=_LEGEND_ABOVE))


def _fig_error_decay(res, n_iterations, outdir):
    """|c_k^{(n)}| against |c_k(0)||1 - mu lambda_k|^n (eq:error_discrete)."""
    C_meas, C_pred = _np(res['C_meas']), _np(res['C_pred'])
    eig_idx = [k for k in _decay_spread_indices(res['lam'], res['mu'], n_iterations)
               if k < res['ev_ok'].numel()]
    series = [(f'k={k + 1}', np.abs(C_meas[:, k]), np.abs(C_pred[:, k]), None)
              for k in eig_idx]
    _decay_figure(res, series, '|v<sub>k</sub><sup>T</sup> e<sup>(n)</sup>|',
                  os.path.join(outdir, 'ntk_error_decay.png'),
                  y_range=_log_range([np.abs(C_meas[:, k]) for k in eig_idx]))
    return [int(k) + 1 for k in eig_idx]


def _error_spectra(res):
    """|ehat_k^{(n)}| of the measured and predicted error, rfft along space.

    This is the error-field spectrum used by the frequency-principle decay panel;
    the band metric of _band_curves instead uses the predicted-field spectrum.
    """
    return error_spectrum(res['e_meas']), error_spectrum(res['e_pred'])


def _fig_spectral_decay(res, outdir):
    """Fixed wavenumbers, so the panel reads the frequency principle literally."""
    ehat_meas, ehat_pred = (_np(a) for a in _error_spectra(res))
    modes = [k for k in (1, 4, 9, 15) if k < ehat_meas.shape[1]]
    series = [(f'k={k}', ehat_meas[:, k], ehat_pred[:, k], None) for k in modes]
    _decay_figure(res, series, '|&#234;<sub>k</sub><sup>(n)</sup>|',
                  os.path.join(outdir, 'ntk_spectral_decay.png'),
                  y_range=_log_range([ehat_meas[:, k] for k in modes], decades=3.0))


def _fig_band_decay(res, target, outdir):
    """Every mode averaged into the low/mid/high split of eq:m_bands."""
    meas, pred = _band_curves(res, target)
    # band names only; the index ranges are eq:m_bands and are restated in the
    # caption, and spelling them out here doubles the legend onto three rows
    labels = ['Low', 'Mid', 'High']
    series = [(lab, _np(m), _np(p), color)
              for color, lab, m, p in zip(THESIS_BAND_COLORS, labels, meas, pred)]
    _decay_figure(res, series, 'Rel. spectral error',
                  os.path.join(outdir, 'ntk_band_decay.png'))


def _fig_fit(results, widths, x, target, t_used, outdir, amplitude=1.0):
    """Final network output at every width against the reference profile, shown
    in physical eta (the training target is normalized by A)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_np(x).tolist(), y=(_np(target) * amplitude).tolist(), mode='lines',
        name='reference', line=dict(color='#000000', width=2.6),
    ))
    for i, w in enumerate(widths):
        fig.add_trace(go.Scatter(
            x=_np(x).tolist(), y=(_np(results[w]['eta_fit']) * amplitude).tolist(), mode='lines',
            name=f'GD, width={w}',
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.0, dash='dash'),
        ))
    fig.update_xaxes(title_text='x')
    fig.update_yaxes(title_text=f'&#951;(x, {t_used:g})')
    save_thesis_fig(fig, os.path.join(outdir, 'ntk_fit.png'), _W, _H, _FRAC,
                    extra_layout=dict(margin=_MARGIN, showlegend=True,
                                      legend=_LEGEND_ABOVE))


def _fig_fit_evolution(res, x, target, t_used, n_iterations, outdir, amplitude=1.0,
                       checkpoints=(0, 100, 1000, 10000, 100000, 500000)):
    """One run's output at log-spaced checkpoints, from eta^{(n)} = e^{(n)} + eta_true,
    shown in physical eta (the training target is normalized by A).

    Returns the iterations actually drawn (the nearest stored snapshots).
    """
    import plotly.graph_objects as go

    sample_iters = res['sample_iters']
    wanted = [c for c in checkpoints if c <= n_iterations] + [int(n_iterations)]

    picked, seen = [], set()
    for c in wanted:
        j = int(np.argmin(np.abs(sample_iters - c)))
        if j not in seen:
            seen.add(j)
            picked.append(j)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_np(x).tolist(), y=(_np(target) * amplitude).tolist(), mode='lines',
        name='reference', line=dict(color='#000000', width=2.6),
    ))
    for i, j in enumerate(picked):
        fig.add_trace(go.Scatter(
            x=_np(x).tolist(), y=(_np(res['e_meas'][j] + target) * amplitude).tolist(), mode='lines',
            name=f'n = {int(sample_iters[j]):,}',
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.9, dash='dash'),
        ))
    fig.update_xaxes(title_text='x')
    fig.update_yaxes(title_text=f'&#951;(x, {t_used:g})')
    save_thesis_fig(fig, os.path.join(outdir, 'ntk_fit_evolution.png'), _W, _H, _FRAC,
                    extra_layout=dict(margin=dict(t=49, b=62, l=96, r=20),
                                      showlegend=True, legend=_LEGEND_ABOVE))
    return [int(sample_iters[j]) for j in picked]


def _fig_iteration_count(results, widths, n_iterations, outdir):
    """Measured vs predicted n_k (eq:iteration_count); the closed form carries no
    width, so all three networks should fall on one diagonal.

    Only modes whose predicted n_k lies within the run are drawn. With n_iterations
    gradient steps the run cannot reach a direction the frozen kernel places beyond
    it, so that direction's closed-form n_k (up to ~1e14 for the slowest modes) is
    an extrapolation the measurement never tests; plotting it would stretch the
    axis by ten decades past any point that can sit on it."""
    import plotly.graph_objects as go

    x_cap = float(n_iterations)
    # (theoretical, measured) pairs the run can actually test, per width
    kept = {w: [(t, m) for t, m in zip(results[w]['theo_n'], results[w]['meas_n'])
                if t <= x_cap]
            for w in widths}

    symbols = ['circle', 'square', 'diamond']
    fig = go.Figure()
    lo, hi = np.inf, 0.0
    for pairs in kept.values():
        for t, m in pairs:
            lo, hi = min(lo, t, m), max(hi, t, m)
    if np.isfinite(lo):
        diag = [max(lo, 1.0) / 1.8, hi * 1.8]
        fig.add_trace(go.Scatter(
            x=diag, y=diag, mode='lines', name='y = x',
            line=dict(color='#888888', width=1.6, dash='dash'),
        ))
    for i, w in enumerate(widths):
        pairs = kept[w]
        if not pairs:
            continue
        fig.add_trace(go.Scatter(
            x=[t for t, _ in pairs], y=[max(m, 1) for _, m in pairs],
            mode='markers', name=f'width={w}',
            marker=dict(size=8, color=_PALETTE[i % len(_PALETTE)],
                        symbol=symbols[i % len(symbols)],
                        line=dict(width=0.6, color='#333333')),
        ))
    fig.update_xaxes(
        title_text='Theoretical n<sub>k</sub> = ln(|c<sub>k</sub>(0)|/&#949;) / (&#956; &#955;<sub>k</sub>)',
        type='log')
    fig.update_yaxes(title_text='Measured n<sub>k</sub>', type='log')
    save_thesis_fig(fig, os.path.join(outdir, 'ntk_iteration_count.png'), _W, _H, _FRAC,
                    extra_layout=dict(margin=dict(t=40, b=62, l=104, r=20), showlegend=True,
                                      legend=_LEGEND_ABOVE))


def _fig_drift_panels(results, widths, outdir, filename):
    """Parameter drift, kernel drift and eigenvalue spectrum, one PNG each."""
    import plotly.graph_objects as go

    ensure_outdir(outdir)
    base, _, ext = filename.rpartition('.')
    base, ext = base or filename, ext or 'png'

    def _finish(fig, xlab, ylab, suffix, logx=False, logy=False, legend=True):
        save_thesis_fig(
            fig, os.path.join(outdir, f'{base}_{suffix}.{ext}'), _W, _H, _FRAC,
            extra_layout=dict(
                xaxis_title=xlab, yaxis_title=ylab,
                margin=dict(t=42 if legend else 14, b=62, l=96, r=20),
                showlegend=legend,
                **({'xaxis_type': 'log'} if logx else {}),
                **({'yaxis_type': 'log'} if logy else {}),
            ),
        )

    # y stays linear, following \textcite{wang2022ntk}: the point is how small the
    # drift is and how it orders by width, which a log y would flatten. x is log
    # because the kernel is rebuilt on a log schedule, and log-spaced samples drawn
    # on a linear axis join into visible straight segments.
    for key, ylab, suffix in (('theta_rel', 'Rel. parameter change', 'param'),
                              ('k_rel', 'Rel. NTK change', 'kernel')):
        fig = go.Figure()
        for i, w in enumerate(widths):
            fig.add_trace(go.Scatter(
                x=np.maximum(results[w]['kernel_iters'], 1).tolist(), y=results[w][key],
                mode='lines', name=f'width={w}',
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.6),
            ))
        # the legend for this LaTeX figure is carried once, by the bottom panel
        _finish(fig, 'Iteration', ylab, suffix, logx=True, legend=False)

    fig_c = go.Figure()
    for i, w in enumerate(widths):
        r = results[w]
        color = _PALETTE[i % len(_PALETTE)]
        for ev, dash, sym, tag in ((r['ev_ok'], 'solid', 'circle', 'init'),
                                   (r['ev_final'], 'dash', 'diamond', 'final')):
            fig_c.add_trace(go.Scatter(
                x=np.arange(1, ev.numel() + 1).tolist(), y=_np(ev).tolist(),
                mode='lines+markers', name=f'width={w} ({tag})',
                line=dict(color=color, width=2.4, dash=dash),
                marker=dict(size=5, symbol=sym),
            ))
    # linear in the index: only a dozen modes clear the floor, and a log index axis
    # would crowd them into the left edge
    _finish(fig_c, 'Eigenvalue index', 'Eigenvalue', 'spectrum', logy=True)


def load_probe_metadata(model_metadata_file):
    """The probe's saved run: (metrics, shared arrays, per-width run dicts).

    Each run dict comes back with the keys `train.ntk._run_width` produced, so the
    figure functions are indifferent to whether they were handed a live run.
    """
    if not os.path.exists(model_metadata_file):
        raise FileNotFoundError(
            f'probe metadata not found: {model_metadata_file}. Run the training '
            'stage (experiments.train.ntk) before this one.')
    data = torch.load(model_metadata_file, map_location='cpu', weights_only=False)
    arrays = data.get('arrays')
    if not arrays or 'runs' not in arrays:
        raise KeyError(f'{model_metadata_file} carries no per-width run arrays; it '
                       'predates the train/plot split. Re-run the training stage.')

    metrics = {k: v for k, v in data.items() if k != 'arrays'}
    widths = [int(w) for w in arrays['widths']]
    results = {w: arrays['runs'][str(w)] for w in widths}

    required = ('lam', 'mu', 'ev_ok', 'ev_final', 'sample_iters', 'kernel_iters',
                'e_meas', 'e_pred', 'C_meas', 'C_pred', 'eta_fit', 'theta_rel',
                'k_rel', 'meas_n', 'theo_n')
    missing = {key for w in widths for key in required if key not in results[w]}
    if missing:
        raise KeyError(f'{model_metadata_file} is missing run arrays {sorted(missing)}; '
                       're-run the training stage.')
    return metrics, arrays, widths, results


def plot_ntk(model_metadata_file, outdir, drift_outdir, filename='mlp_ntk_panel.png'):
    """Emit every probe figure from the saved metadata, and record what they show.

    `displayed_eigenindices` and `fit_evolution_checkpoints` are selections this
    stage makes, so it appends them to the metrics JSON. The base is always the
    `.pth` the trainer wrote, never a previous run of this function, so repeated
    calls are idempotent.
    """
    metrics, arrays, widths, results = load_probe_metadata(model_metadata_file)
    ensure_outdir(outdir)
    ensure_outdir(drift_outdir)

    x, target, t_used = arrays['x'], arrays['target'], arrays['t_slice']
    amplitude = float(arrays.get('amplitude', 1.0))     # target is stored as eta / A
    n_iterations = int(metrics['n_iterations'])
    ref_w = int(metrics['reference_width'])
    if ref_w not in results:
        ref_w = widths[-1]
    print(f'  probe figures from {model_metadata_file}')
    print(f'  reference width for the decay panels: {ref_w}')

    eig_shown = _fig_error_decay(results[ref_w], n_iterations, outdir)
    _fig_spectral_decay(results[ref_w], outdir)
    _fig_band_decay(results[ref_w], target, outdir)
    _fig_iteration_count(results, widths, n_iterations, outdir)
    _fig_fit(results, widths, x, target, t_used, outdir, amplitude)
    fit_checkpoints = _fig_fit_evolution(results[ref_w], x, target, t_used,
                                         n_iterations, outdir, amplitude)
    _fig_drift_panels(results, widths, drift_outdir, filename)

    metrics['displayed_eigenindices'] = eig_shown
    metrics['fit_evolution_checkpoints'] = fit_checkpoints
    label = metrics.get('label', 'mlp_ntk')
    save_metadata_json(metrics, os.path.join(os.path.dirname(model_metadata_file),
                                             f'{label}_metadata.json'))
    print(f'spectral-bias NTK figures saved to {outdir} and {drift_outdir}')
    return metrics
