import os
import json

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

# in-figure text is pure black to match the black LaTeX body text (plotly's
# default is a dark slate blue, which reads lighter than the surrounding prose).
_THESIS_FONT_COLOR = '#000000'

# Shared colourblind-safe qualitative palette (Okabe--Ito), used everywhere a set
# of categorical series needs distinct colours, replacing the Matplotlib/Plotly
# tab10 defaults that are not safe under deuteranopia/protanopia. Assign in this
# fixed order; do not cycle a longer set of series through it.
THESIS_PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']

# Three-class band palette for the low/mid/high spectral bands, drawn from the
# same Okabe--Ito set so the band figures match the rest of the document.
THESIS_BAND_COLORS = ['#0072B2', '#E69F00', '#D55E00']  # low, mid, high


def _thesis_px(pt, fig_width_px, frac):
    # plotly font size (logical px) that prints at `pt` when a figure of logical
    # width `fig_width_px` is \includegraphics'd at `frac`*\textwidth
    return pt * fig_width_px / (frac * THESIS_TEXTWIDTH_IN * 72.0)


def _thesis_px_sizes(fig_width_px, frac=1.0):
    # resolved in-figure font sizes (px) for a figure of logical width
    # `fig_width_px` shown at `frac`*\textwidth, keyed by _THESIS_PT roles
    return {k: _thesis_px(v, fig_width_px, frac) for k, v in _THESIS_PT.items()}


def _style_thesis(fig, fig_width_px, frac=1.0):
    # apply consistent, print-sized fonts to a plotly figure and return the
    # resolved px sizes so callers can size their own annotations to match.
    px = _thesis_px_sizes(fig_width_px, frac)
    fig.update_layout(
        template='plotly_white',
        font=dict(size=px['tick'], color=_THESIS_FONT_COLOR),
        title_font_size=px['title'],
        legend_font_size=px['legend'],
    )
    fig.update_xaxes(title_font_size=px['axis'], tickfont_size=px['tick'],
                     title_font_color=_THESIS_FONT_COLOR, tickfont_color=_THESIS_FONT_COLOR)
    fig.update_yaxes(title_font_size=px['axis'], tickfont_size=px['tick'],
                     title_font_color=_THESIS_FONT_COLOR, tickfont_color=_THESIS_FONT_COLOR)
    # subplot titles (and any annotations already present) come from make_subplots
    # as layout annotations; size them to the subplot-title target
    for ann in fig.layout.annotations:
        ann.font.size = px['subplot']
    return px


def _ensure_outdir(outdir):
    # make sure output directory exists before saving files
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ---------------------------------------------------------------------------
# Split-panel helpers
# ---------------------------------------------------------------------------
# The multi-panel figures are broken into one standalone image per subplot so a
# single sub-figure can be regenerated in isolation (e.g. after a reviewer note)
# without rebuilding the whole panel. LaTeX reassembles them with `subfigure`.
# Every emitted file keeps the panel's `filename` stem plus a descriptive
# suffix, so callers keep their existing signatures.

def _panel_stem(filename):
    # 'fno_model2_spectral_panel.png' -> 'fno_model2_spectral_panel'
    return os.path.splitext(os.path.basename(filename))[0]


def _slugify(text):
    # 'PINN (no data)' -> 'pinn_no_data'
    import re
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def _save_thesis_fig(fig, outpath, width_px, height_px, frac, extra_layout=None, scale=2.0):
    # style a standalone plotly figure at print size and write it as PNG. Font
    # sizes print consistently across every emitted sub-figure because they are
    # normalized to (width_px, frac); see _style_thesis. `extra_layout` is applied
    # after styling, so annotations passed there keep the font sizes set on them.
    _style_thesis(fig, width_px, frac=frac)
    layout = dict(width=width_px, height=height_px)
    if extra_layout:
        layout.update(extra_layout)
    fig.update_layout(**layout)
    try:
        fig.write_image(outpath, scale=scale)
        print(f'  saved {outpath}')
    except Exception as e:
        print('Erro ao salvar figura como PNG. Instale kaleido: pip install kaleido')
        raise e


def _autocrop_white(path, pad=6):
    # trim the uniform white margins plotly pads around a 3D scene. Only vertical
    # slack is really removed here (the surface already spans the width), so the
    # image stays width-dominated and width=\linewidth inclusion keeps its text at
    # the intended print size. No-op if Pillow is unavailable.
    try:
        from PIL import Image, ImageChops
    except ImportError:
        print('  (autocrop skipped: Pillow not installed)')
        return
    img = Image.open(path).convert('RGB')
    bg = Image.new('RGB', img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox is None:
        return
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img.width, right + pad)
    bottom = min(img.height, bottom + pad)
    img.crop((left, top, right, bottom)).save(path)


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


def plot_training_statistics(histories, labels, outdir, filename, log_scale=True, duration_seconds=None, final_loss=None, num_params=None):
    import plotly.graph_objects as go
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    _colors = THESIS_PALETTE
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
# Shared builder for the stacked 3D-surface panels (alpha/beta and resolution)
# ---------------------------------------------------------------------------
# Each item (a parameter value or a resolution) yields one standalone 3D-surface
# image and one relative-error-vs-time image; a single box-plot image summarizes
# the error distribution across items. All surfaces share one z-range and one
# camera so they stay directly comparable when reassembled in LaTeX.
_SURFACE_CAMERA = dict(eye=dict(x=-1.15, y=-1.40, z=0.95),
                       center=dict(x=0, y=0, z=-0.05), up=dict(x=0, y=0, z=1))
_SURFACE_ASPECTRATIO = dict(x=1.9, y=1.0, z=0.52)
# Viridis is perceptually uniform and colourblind-safe, unlike the rainbow-family
# Turbo, so it renders the smooth elevation field honestly without false banding;
# matte directional lighting (low ambient, high diffuse, almost no specular) casts
# clean shadows on the ripple facets so they stay well defined.
_SURFACE_COLORSCALE = 'Viridis'
_SURFACE_LIGHTING = dict(ambient=0.30, diffuse=0.95, specular=0.05, roughness=0.9, fresnel=0.1)
_SURFACE_LIGHTPOSITION = dict(x=-1.6, y=-1.0, z=0.25)
# fraction of the rendered canvas width that survives _autocrop_white (the 3D
# box projection is fixed by camera+aspect, so this is deterministic and equal
# for every surface). Used to size the scene axis titles so that, after cropping
# and inclusion at \linewidth, they print at the same point size as the 2D tiles.
_SURFACE_CROP_W_RATIO = 0.681


def _emit_surface_panel_figures(stem, outdir, x_list, t_list, eta_pred_list, time_rel_norms,
                                box_names, box_fillcolor, box_xaxis_title):
    import plotly.graph_objects as go

    z_global_min = min(float(np.asarray(e).min()) for e in eta_pred_list)
    z_global_max = max(float(np.asarray(e).max()) for e in eta_pred_list)
    z_margin = (z_global_max - z_global_min) * 0.05
    z_lim = [z_global_min - z_margin, z_global_max + z_margin]

    SURF_W, SURF_H, SURF_FRAC = 1000, 500, 0.48
    LINE_W, LINE_H, LINE_FRAC = 820, 470, 0.48
    BOX_W, BOX_H, BOX_FRAC = 1200, 430, 0.68
    # the surface is autocropped to ~SURF_W*_SURFACE_CROP_W_RATIO wide before it is
    # shown at 0.48\textwidth, so size the axis titles against that effective width.
    axis_px = _thesis_px_sizes(SURF_W * _SURFACE_CROP_W_RATIO, SURF_FRAC)['axis']

    for i, (x_res, t_res, eta_pred, rel_norm) in enumerate(
            zip(x_list, t_list, eta_pred_list, time_rel_norms), start=1):
        # ---- predicted surface eta(x, t): x-axis is time, y-axis is space ----
        fig = go.Figure(go.Surface(
            x=np.asarray(t_res, dtype=float), y=np.asarray(x_res, dtype=float),
            z=np.asarray(eta_pred, dtype=float),
            colorscale=_SURFACE_COLORSCALE, cmin=z_global_min, cmax=z_global_max, showscale=False,
            lighting=_SURFACE_LIGHTING, lightposition=_SURFACE_LIGHTPOSITION,
        ))
        # 3D scenes are qualitative here: perspective tick labels pile up and bury
        # the surface, so they are hidden and only the axis titles are kept.
        fig.update_layout(scene=dict(
            xaxis=dict(title='Time (t)', title_font=dict(size=axis_px, color=_THESIS_FONT_COLOR), showticklabels=False),
            yaxis=dict(title='Space (x)', title_font=dict(size=axis_px, color=_THESIS_FONT_COLOR), showticklabels=False),
            zaxis=dict(title='η', title_font=dict(size=axis_px, color=_THESIS_FONT_COLOR), showticklabels=False, range=z_lim),
            aspectmode='manual', aspectratio=_SURFACE_ASPECTRATIO, camera=_SURFACE_CAMERA,
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        ))
        surface_path = os.path.join(outdir, f'{stem}_surface_{i}.png')
        _save_thesis_fig(
            fig, surface_path,
            SURF_W, SURF_H, SURF_FRAC,
            extra_layout=dict(margin=dict(t=8, b=16, l=16, r=14)),
        )
        _autocrop_white(surface_path)

        # ---- relative error over time ----
        fig = go.Figure(go.Scatter(
            x=np.asarray(t_res, dtype=float).tolist(),
            y=np.asarray(rel_norm, dtype=float).tolist(),
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ))
        fig.update_xaxes(title_text='Time (t)')
        fig.update_yaxes(title_text='Relative error')
        _save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_relerr_{i}.png'),
            LINE_W, LINE_H, LINE_FRAC,
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
    _save_thesis_fig(
        fig, os.path.join(outdir, f'{stem}_box.png'),
        BOX_W, BOX_H, BOX_FRAC,
        extra_layout=dict(margin=dict(t=15, b=55, l=70, r=25)),
    )

    # persist the per-item time-resolved relative-error distribution so the numbers
    # quoted in the text come from data rather than being read back off the PNG box plot
    metrics = {
        'xaxis': box_xaxis_title,
        'items': [
            {
                'name': str(name),
                'median': float(np.median(rel_norm)),
                'mean': float(np.mean(rel_norm)),
                'q1': float(np.percentile(rel_norm, 25)),
                'q3': float(np.percentile(rel_norm, 75)),
                'min': float(np.min(rel_norm)),
                'max': float(np.max(rel_norm)),
            }
            for name, rel_norm in zip(box_names, time_rel_norms)
        ],
    }
    with open(os.path.join(outdir, f'{stem}_metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(f'surface panel figures saved to {outdir} ({stem}_*)')


def plot_model2_alpha_beta_panel(x, t, eta_true_list, eta_pred_list, param_values, outdir, filename, title,
                                 eval_resolution=None, res_label=None):
    # Emits, per parameter value, a {stem}_surface_{i}.png and {stem}_relerr_{i}.png,
    # plus one {stem}_box.png for the error distribution across parameters. All
    # items share the same spatial/temporal grid.
    _ensure_outdir(outdir)
    stem = _panel_stem(filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    param_values = np.asarray(param_values, dtype=float)
    n_params = len(param_values)
    if n_params == 0:
        raise ValueError('param_values must contain at least one value')

    time_rel_norms = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        eta_true = np.asarray(eta_true, dtype=float)
        eta_pred = np.asarray(eta_pred, dtype=float)
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)

    _emit_surface_panel_figures(
        stem, outdir,
        x_list=[x] * n_params, t_list=[t] * n_params,
        eta_pred_list=eta_pred_list, time_rel_norms=time_rel_norms,
        box_names=[f'α=β={alpha:.2f}' for alpha in param_values],
        box_fillcolor='#fdd0a2', box_xaxis_title='α=β value',
    )


def plot_model2_resolution_panel(x_list, t_list, eta_true_list, eta_pred_list, resolutions, outdir, filename, title,
                                 eval_alpha_beta=None, param_label=None):
    # Emits, per resolution, a {stem}_surface_{i}.png and {stem}_relerr_{i}.png,
    # plus one {stem}_box.png for the error distribution across resolutions. Each
    # item carries its own spatial/temporal grid.
    _ensure_outdir(outdir)
    stem = _panel_stem(filename)

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
        eta_true = np.asarray(eta_true, dtype=float)
        eta_pred = np.asarray(eta_pred, dtype=float)
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)

    _emit_surface_panel_figures(
        stem, outdir,
        x_list=x_list, t_list=t_list,
        eta_pred_list=eta_pred_list, time_rel_norms=time_rel_norms,
        box_names=[str(int(res)) for res in resolutions],
        box_fillcolor='#fdae6b', box_xaxis_title='Resolution',
    )


def plot_model2_spectral_panel(x, t, eta_true, eta_pred, outdir, filename, title,
                               n_times=3, eval_resolution=None, res_label=None, param_label=None):
    # Emits one standalone image per subplot (assembled in LaTeX via subfigure):
    #   {stem}_spectrum_{i}.png : reference vs predicted spectrum at snapshot i
    #   {stem}_relerr_{i}.png   : relative spectral error at snapshot i
    #   {stem}_mean.png         : mean relative spectral error over time
    import plotly.graph_objects as go

    _ensure_outdir(outdir)
    stem = _panel_stem(filename)

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

    # persist the mean relative spectral error over time (and its value at each
    # displayed snapshot) so the text quotes measured numbers, not pixel reads
    metrics = {
        'time': t.tolist(),
        'mean_rel_spectral_error_over_time': mean_rel_err_over_time.tolist(),
        'snapshots': [
            {'t': float(t[idx]), 'mean_rel_spectral_error': float(mean_rel_err_over_time[idx])}
            for idx in indices
        ],
    }
    with open(os.path.join(outdir, f'{stem}_metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # display geometry: spectrum/error tiles go two-per-row at 0.48\textwidth,
    # the mean-over-time plot spans a single wider centred slot at 0.68\textwidth.
    PAIR_W, PAIR_H, PAIR_FRAC = 820, 470, 0.48
    MEAN_W, MEAN_H, MEAN_FRAC = 1200, 430, 0.68
    annot_px = _thesis_px_sizes(PAIR_W, PAIR_FRAC)['annot']

    kxl = kx.tolist()
    for i, idx in enumerate(indices, start=1):
        # ---- reference vs predicted spectrum ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kxl, y=true_spec[:, idx].tolist(),
            mode='lines+markers', name='True',
            line=dict(color='#222222', width=2.0),
            marker=dict(size=3, symbol='circle'),
        ))
        fig.add_trace(go.Scatter(
            x=kxl, y=pred_spec[:, idx].tolist(),
            mode='lines+markers', name='Predicted',
            line=dict(color='#ff7f0e', width=1.8, dash='dash'),
            marker=dict(size=3, symbol='diamond'),
        ))
        fig.update_xaxes(title_text='Spectral index n')
        fig.update_yaxes(title_text='Amplitude')
        _save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_spectrum_{i}.png'),
            PAIR_W, PAIR_H, PAIR_FRAC,
            extra_layout=dict(
                margin=dict(t=15, b=55, l=70, r=20),
                showlegend=True,
                legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top',
                            bgcolor='rgba(255,255,255,0.7)'),
            ),
        )

        # ---- relative spectral error, with time-mean annotation ----
        mean_rel = float(mean_rel_err_over_time[idx])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kxl, y=rel_err[:, idx].tolist(),
            mode='lines+markers', showlegend=False,
            line=dict(color='crimson', width=1.8),
            marker=dict(size=3, symbol='circle'),
        ))
        fig.update_xaxes(title_text='Spectral index n')
        fig.update_yaxes(title_text='Relative error')
        _save_thesis_fig(
            fig, os.path.join(outdir, f'{stem}_relerr_{i}.png'),
            PAIR_W, PAIR_H, PAIR_FRAC,
            extra_layout=dict(
                margin=dict(t=15, b=55, l=70, r=20),
                annotations=[dict(
                    text=f'Mean: {mean_rel:.2e}',
                    xref='paper', yref='paper', x=0.97, y=0.97,
                    xanchor='right', yanchor='top', showarrow=False,
                    font=dict(size=annot_px),
                    bgcolor='white', bordercolor='#cccccc', borderwidth=1, borderpad=4,
                )],
            ),
        )

    # ---- mean relative spectral error over time ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=mean_rel_err_over_time.tolist(),
        mode='lines+markers', showlegend=False,
        line=dict(color='#c28b00', width=2.0),
        marker=dict(size=5),
    ))
    fig.update_xaxes(title_text='Time (t)')
    fig.update_yaxes(title_text='Mean relative error')
    _save_thesis_fig(
        fig, os.path.join(outdir, f'{stem}_mean.png'),
        MEAN_W, MEAN_H, MEAN_FRAC,
        extra_layout=dict(margin=dict(t=15, b=55, l=70, r=25)),
    )
    print(f'spectral figures saved to {outdir} ({stem}_*)')


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
    plt.colorbar(im, cax=cax, label='Relative Error (clipped at 1.0)')
    fig.subplots_adjust(right=0.88)
    plt.savefig(outpath, dpi=150)
    print(f'error heatmap saved to {outpath}')
    plt.close()


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
        colorbar=dict(title='Rel. err (clip 1.0)', x=1.02, len=0.45, y=0.22),
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
    line_true, = ax.plot([], [], color='0.55', lw=2.0, label='Reference (pseudospectral)')
    line_pred, = ax.plot([], [], color='#E66C11', lw=2.4, label='Prediction')
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
