import os

import numpy as np
import torch

from experiments.plots_common import _ensure_outdir, _panel_stem, _slugify, _save_thesis_fig, THESIS_BAND_COLORS


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
    # Emits one standalone image per model (assembled in LaTeX via subfigure):
    #   {stem}_{model-slug}.png : low/mid/high band relative error vs epoch (log y)
    import plotly.graph_objects as go

    _ensure_outdir(outdir)
    stem = _panel_stem(filename)

    band_colors = THESIS_BAND_COLORS
    band_labels = ['Low (k < Nx/4)', 'Mid (Nx/4 &#8804; k < Nx/2)', 'High (k &#8805; Nx/2)']
    band_keys = ['low_band', 'mid_band', 'high_band']

    # frac raised well above the 0.48 subfigure width so the in-figure text prints
    # about half its former size (ticks ~6.5 pt, axis ~7.4 pt, legend ~6.9 pt) and
    # no longer dominates the panel; the legend is moved out below the axes.
    W, H, FRAC = 820, 520, 1.1

    for label, sh in models_data:
        epochs = np.asarray(sh['epochs']).tolist()
        fig = go.Figure()
        for color, blab, key in zip(band_colors, band_labels, band_keys):
            vals = np.asarray(sh[key]).tolist()
            fig.add_trace(go.Scatter(
                x=epochs, y=vals, mode='lines', name=blab,
                line=dict(color=color, width=2.0), opacity=0.9,
            ))
        fig.update_xaxes(title_text='Epoch')
        fig.update_yaxes(title_text='Rel. spectral error', type='log')
        _save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_{_slugify(label)}.png'),
            W, H, FRAC,
            extra_layout=dict(
                margin=dict(t=12, b=78, l=64, r=16),
                showlegend=True,
                legend=dict(orientation='h', xanchor='center', x=0.5,
                            yanchor='top', y=-0.22),
            ),
        )
    print(f'spectral bias figures saved to {outdir} ({stem}_*)')
