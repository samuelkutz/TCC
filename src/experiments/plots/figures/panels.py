"""One function per LaTeX figure of the results chapter.

Multi-part figures are written as one PNG per sub-panel, `{stem}_{suffix}.png`,
which LaTeX reassembles with `subfigure`. Each also writes `{stem}_metrics.json`
with every number the panel displays, so the text quotes measured values.
"""

import json
import os

import numpy as np

from experiments.plots.figures.metrics import (
    compute_relative_error, distribution_summary, spectral_mode_index,
    time_relative_error_norm,
)
from experiments.plots.figures.style import (
    THESIS_BAND_COLORS, THESIS_ERROR_COLOR, THESIS_LEGEND_ABOVE, THESIS_PALETTE,
    THESIS_PREDICTION_COLOR, THESIS_REFERENCE_COLOR,
    autocrop_white, ensure_outdir, panel_stem, save_thesis_fig, slugify,
    thesis_px_sizes,
)


def _write_metrics(outdir, stem, payload):
    path = os.path.join(outdir, f'{stem}_metrics.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    return path


def plot_training_statistics(histories, labels, outdir, filename, log_scale=True,
                             duration_seconds=None, final_loss=None, num_params=None):
    """Loss against iteration, annotated with parameter count, wall clock and final loss."""
    import plotly.graph_objects as go

    ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    fig = go.Figure()
    for i, (history, label) in enumerate(zip(histories, labels)):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)), y=list(history),
            mode='lines', name=label,
            line=dict(color=THESIS_PALETTE[i % len(THESIS_PALETTE)], width=2.2),
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
        legend=THESIS_LEGEND_ABOVE,
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
    except Exception:
        print('  failed to write PNG; kaleido is required: pip install kaleido')
        raise
    print(f'training statistics plot saved to {outpath}')
    return outpath


def plot_soliton_profile(outdir, filename='soliton_profile.png', amplitude=1.0):
    """The initial elevation eta(x, 0) = A sech^2(x)."""
    import plotly.graph_objects as go

    ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.linspace(-6, 6, 600)
    y = amplitude / np.cosh(x) ** 2

    fig = go.Figure(go.Scatter(
        x=x.tolist(), y=y.tolist(), mode='lines',
        line=dict(color=THESIS_PREDICTION_COLOR, width=2.2), showlegend=False,
    ))
    fig.update_xaxes(title_text='x')
    fig.update_yaxes(title_text='η(x,0)')
    return save_thesis_fig(fig, outpath, 1000, 400, 1.0,
                           extra_layout=dict(margin=dict(t=15, b=45, l=55, r=25)))


# All surfaces share one z-range and one camera, so the tiles stay comparable
# once LaTeX reassembles them.
_SURFACE_CAMERA = dict(eye=dict(x=-1.15, y=-1.40, z=0.95),
                       center=dict(x=0, y=0, z=-0.05), up=dict(x=0, y=0, z=1))
_SURFACE_ASPECTRATIO = dict(x=1.9, y=1.0, z=0.52)
# Viridis is perceptually uniform, so it shows the elevation field without the
# false banding a rainbow scale introduces; matte lighting keeps the ripple
# facets defined.
_SURFACE_COLORSCALE = 'Viridis'
_SURFACE_LIGHTING = dict(ambient=0.30, diffuse=0.95, specular=0.05, roughness=0.9, fresnel=0.1)
_SURFACE_LIGHTPOSITION = dict(x=-1.6, y=-1.0, z=0.25)
# fraction of the canvas width surviving autocrop_white, deterministic because the
# 3D box projection is fixed by camera+aspect. Used to size the scene axis titles
# so they print like the 2D tiles once cropped and included at \linewidth.
_SURFACE_CROP_W_RATIO = 0.681

_SURF_W, _SURF_H, _SURF_FRAC = 1000, 500, 0.48
_LINE_W, _LINE_H, _LINE_FRAC = 820, 470, 0.48
_BOX_W, _BOX_H, _BOX_FRAC = 1200, 430, 0.68


def _emit_surface_panel(stem, outdir, x_list, t_list, eta_pred_list, time_rel_norms,
                        box_names, box_fillcolor, box_xaxis_title):
    import plotly.graph_objects as go

    z_min = min(float(np.asarray(e).min()) for e in eta_pred_list)
    z_max = max(float(np.asarray(e).max()) for e in eta_pred_list)
    z_margin = (z_max - z_min) * 0.05
    z_lim = [z_min - z_margin, z_max + z_margin]

    axis_px = thesis_px_sizes(_SURF_W * _SURFACE_CROP_W_RATIO, _SURF_FRAC)['axis']
    scene_axis = dict(title_font=dict(size=axis_px, color='#000000'), showticklabels=False)

    for i, (x_res, t_res, eta_pred, rel_norm) in enumerate(
            zip(x_list, t_list, eta_pred_list, time_rel_norms), start=1):
        # ---- predicted surface eta(x, t): x-axis is time, y-axis is space ----
        fig = go.Figure(go.Surface(
            x=np.asarray(t_res, dtype=float), y=np.asarray(x_res, dtype=float),
            z=np.asarray(eta_pred, dtype=float),
            colorscale=_SURFACE_COLORSCALE, cmin=z_min, cmax=z_max, showscale=False,
            lighting=_SURFACE_LIGHTING, lightposition=_SURFACE_LIGHTPOSITION,
        ))
        # tick labels hidden: in perspective they pile up and bury the surface
        fig.update_layout(scene=dict(
            xaxis=dict(title='Time (t)', **scene_axis),
            yaxis=dict(title='Space (x)', **scene_axis),
            zaxis=dict(title='η', range=z_lim, **scene_axis),
            aspectmode='manual', aspectratio=_SURFACE_ASPECTRATIO, camera=_SURFACE_CAMERA,
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        ))
        surface_path = save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_surface_{i}.png'),
            _SURF_W, _SURF_H, _SURF_FRAC,
            extra_layout=dict(margin=dict(t=8, b=16, l=16, r=14)),
        )
        autocrop_white(surface_path)

        # ---- relative error over time ----
        fig = go.Figure(go.Scatter(
            x=np.asarray(t_res, dtype=float).tolist(),
            y=np.asarray(rel_norm, dtype=float).tolist(),
            mode='lines', line=dict(color=THESIS_ERROR_COLOR, width=1.8), showlegend=False,
        ))
        fig.update_xaxes(title_text='Time (t)')
        fig.update_yaxes(title_text='Relative error')
        save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_relerr_{i}.png'),
            _LINE_W, _LINE_H, _LINE_FRAC,
            extra_layout=dict(margin=dict(t=15, b=55, l=70, r=20)),
        )

    # ---- error distribution across items ----
    fig = go.Figure()
    for name, rel_norm in zip(box_names, time_rel_norms):
        fig.add_trace(go.Box(
            y=np.asarray(rel_norm, dtype=float).tolist(), name=str(name),
            marker_color='#e6550d', fillcolor=box_fillcolor,
            line_color='#e6550d', showlegend=False, width=0.2,
        ))
    fig.update_xaxes(title_text=box_xaxis_title)
    fig.update_yaxes(title_text='Relative error')
    save_thesis_fig(
        fig, os.path.join(outdir, f'{stem}_box.png'),
        _BOX_W, _BOX_H, _BOX_FRAC,
        extra_layout=dict(margin=dict(t=15, b=55, l=70, r=25)),
    )

    _write_metrics(outdir, stem, {
        'xaxis': box_xaxis_title,
        'items': [{'name': str(name), **distribution_summary(rel_norm)}
                  for name, rel_norm in zip(box_names, time_rel_norms)],
    })
    print(f'surface panel figures saved to {outdir} ({stem}_*)')


def plot_alpha_beta_panel(x, t, eta_true_list, eta_pred_list, param_values, outdir, filename):
    """Surface and error-vs-time tile per alpha=beta value, plus a box plot.

    Every item shares one spatial/temporal grid.
    """
    ensure_outdir(outdir)
    stem = panel_stem(filename)

    param_values = np.asarray(param_values, dtype=float)
    if param_values.size == 0:
        raise ValueError('param_values must contain at least one value')
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    time_rel_norms = [time_relative_error_norm(a, b)
                      for a, b in zip(eta_true_list, eta_pred_list)]
    _emit_surface_panel(
        stem, outdir,
        x_list=[x] * param_values.size, t_list=[t] * param_values.size,
        eta_pred_list=eta_pred_list, time_rel_norms=time_rel_norms,
        box_names=[f'α=β={v:.2f}' for v in param_values],
        box_fillcolor='#fdd0a2', box_xaxis_title='α=β value',
    )


def plot_resolution_panel(x_list, t_list, eta_true_list, eta_pred_list, resolutions,
                          outdir, filename):
    """Surface and error-vs-time tile per resolution, plus a box plot.

    Each item carries its own spatial/temporal grid.
    """
    ensure_outdir(outdir)
    stem = panel_stem(filename)

    resolutions = np.asarray(resolutions, dtype=float)
    if resolutions.size == 0:
        raise ValueError('resolutions must contain at least one value')
    if len(x_list) != resolutions.size or len(t_list) != resolutions.size:
        raise ValueError('x_list and t_list must match the number of resolutions')

    time_rel_norms = [time_relative_error_norm(a, b)
                      for a, b in zip(eta_true_list, eta_pred_list)]
    _emit_surface_panel(
        stem, outdir,
        x_list=[np.asarray(v, dtype=float) for v in x_list],
        t_list=[np.asarray(v, dtype=float) for v in t_list],
        eta_pred_list=eta_pred_list, time_rel_norms=time_rel_norms,
        box_names=[str(int(r)) for r in resolutions],
        box_fillcolor='#fdae6b', box_xaxis_title='Resolution',
    )


_PAIR_W, _PAIR_H, _PAIR_FRAC = 820, 470, 0.48
_MEAN_W, _MEAN_H, _MEAN_FRAC = 1200, 430, 0.68


def plot_spectral_panel(x, t, eta_true, eta_pred, outdir, filename, n_times=3):
    """Spatial spectra at n_times snapshots, from the rfft of the fields along x.

    Emits {stem}_spectrum_{i}, {stem}_relerr_{i} and {stem}_mean.
    """
    import plotly.graph_objects as go

    ensure_outdir(outdir)
    stem = panel_stem(filename)

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
    else:
        indices, seen = [], set()
        for target in np.linspace(t[0], t[-1], n_times):
            idx = int(np.argmin(np.abs(t - target)))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)

    _write_metrics(outdir, stem, {
        'time': t.tolist(),
        'mean_rel_spectral_error_over_time': mean_rel_err_over_time.tolist(),
        'snapshots': [
            {'t': float(t[idx]), 'mean_rel_spectral_error': float(mean_rel_err_over_time[idx])}
            for idx in indices
        ],
    })

    annot_px = thesis_px_sizes(_PAIR_W, _PAIR_FRAC)['annot']
    kxl = kx.tolist()
    for i, idx in enumerate(indices, start=1):
        # ---- reference vs predicted spectrum ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kxl, y=true_spec[:, idx].tolist(),
            mode='lines+markers', name='True',
            line=dict(color=THESIS_REFERENCE_COLOR, width=2.0),
            marker=dict(size=3, symbol='circle'),
        ))
        fig.add_trace(go.Scatter(
            x=kxl, y=pred_spec[:, idx].tolist(),
            mode='lines+markers', name='Predicted',
            line=dict(color=THESIS_PREDICTION_COLOR, width=1.8, dash='dash'),
            marker=dict(size=3, symbol='diamond'),
        ))
        fig.update_xaxes(title_text='Spectral index n')
        fig.update_yaxes(title_text='Amplitude')
        save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_spectrum_{i}.png'),
            _PAIR_W, _PAIR_H, _PAIR_FRAC,
            extra_layout=dict(margin=dict(t=42, b=55, l=70, r=20),
                              showlegend=True, legend=THESIS_LEGEND_ABOVE),
        )

        # ---- relative spectral error, with time-mean annotation ----
        fig = go.Figure(go.Scatter(
            x=kxl, y=rel_err[:, idx].tolist(),
            mode='lines+markers', showlegend=False,
            line=dict(color=THESIS_ERROR_COLOR, width=1.8),
            marker=dict(size=3, symbol='circle'),
        ))
        fig.update_xaxes(title_text='Spectral index n')
        fig.update_yaxes(title_text='Relative error')
        save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_relerr_{i}.png'),
            _PAIR_W, _PAIR_H, _PAIR_FRAC,
            extra_layout=dict(
                margin=dict(t=15, b=55, l=70, r=20),
                annotations=[dict(
                    text=f'Mean: {float(mean_rel_err_over_time[idx]):.2e}',
                    xref='paper', yref='paper', x=0.97, y=0.97,
                    xanchor='right', yanchor='top', showarrow=False,
                    font=dict(size=annot_px),
                    bgcolor='white', bordercolor='#cccccc', borderwidth=1, borderpad=4,
                )],
            ),
        )

    # ---- mean relative spectral error over time ----
    fig = go.Figure(go.Scatter(
        x=t.tolist(), y=mean_rel_err_over_time.tolist(),
        mode='lines+markers', showlegend=False,
        line=dict(color='#c28b00', width=2.0), marker=dict(size=5),
    ))
    fig.update_xaxes(title_text='Time (t)')
    fig.update_yaxes(title_text='Mean relative error')
    save_thesis_fig(
        fig, os.path.join(outdir, f'{stem}_mean.png'),
        _MEAN_W, _MEAN_H, _MEAN_FRAC,
        extra_layout=dict(margin=dict(t=15, b=55, l=70, r=25)),
    )
    print(f'spectral figures saved to {outdir} ({stem}_*)')


# the index ranges are defined once in eq:m_bands and restated in every caption
# that uses these curves, so the legend carries only the band name; spelling the
# ranges out here wraps the legend onto three rows and costs a third of the panel
_BAND_LABELS = ['Low', 'Mid', 'High']
_BAND_KEYS = ['low_band', 'mid_band', 'high_band']

# frac raised above the 0.48 subfigure width so the in-figure text prints at ~7 pt
BAND_W, BAND_H, BAND_FRAC = 820, 470, 0.48   # match the SciML pair panels


def plot_spectral_bias_panel(models_data, outdir, filename='spectral_bias_evolution.png'):
    """Low/mid/high band relative spectral error vs epoch, one PNG per model."""
    import plotly.graph_objects as go

    ensure_outdir(outdir)
    stem = panel_stem(filename)

    metrics = {}
    for label, history in models_data:
        epochs = np.asarray(history['epochs']).tolist()
        fig = go.Figure()
        for color, band_label, key in zip(THESIS_BAND_COLORS, _BAND_LABELS, _BAND_KEYS):
            values = np.asarray(history[key]).tolist()
            fig.add_trace(go.Scatter(
                x=epochs, y=values, mode='lines', name=band_label,
                line=dict(color=color, width=2.0), opacity=0.9,
            ))
        save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_{slugify(label)}.png'),
            BAND_W, BAND_H, BAND_FRAC,
            extra_layout=dict(
                xaxis_title='Epoch', yaxis_title='Rel. spectral error', yaxis_type='log',
                margin=dict(t=40, b=62, l=86, r=20), showlegend=True,
                legend=THESIS_LEGEND_ABOVE,
            ),
        )
        metrics[slugify(label)] = {
            'label': label,
            'epochs': epochs,
            **{key: [float(v) for v in np.asarray(history[key])] for key in _BAND_KEYS},
            'first': {key: float(np.asarray(history[key])[0]) for key in _BAND_KEYS},
            'final': {key: float(np.asarray(history[key])[-1]) for key in _BAND_KEYS},
        }

    _write_metrics(outdir, stem, metrics)
    print(f'spectral bias figures saved to {outdir} ({stem}_*)')
