"""Shared figure style: palette, print-sized fonts, output paths, PNG writer."""

import os
import re

import numpy as np


# A plotly figure of logical width W px included at frac*\textwidth prints a font
# of f px at f * frac * THESIS_TEXTWIDTH_IN * 72 / W points. thesis_px inverts
# that, so in-figure text lands at a chosen printed point size regardless of the
# figure's pixel dimensions. The document's text block is ~160 mm wide.
THESIS_TEXTWIDTH_IN = 6.30
_THESIS_PT = dict(tick=7.5, axis=8.5, subplot=9.0, title=10.0, legend=8.0, annot=7.0)

# black, to match the LaTeX body text; plotly's default is a lighter slate blue
_THESIS_FONT_COLOR = '#000000'

# Okabe--Ito, which is safe under deuteranopia and protanopia unlike the tab10
# defaults. Assign in this fixed order.
THESIS_PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                  '#CC79A7', '#56B4E9', '#F0E442', '#000000']
THESIS_BAND_COLORS = ['#0072B2', '#E69F00', '#D55E00']  # low, mid, high

THESIS_REFERENCE_COLOR = '#222222'
THESIS_PREDICTION_COLOR = '#ff7f0e'
THESIS_ERROR_COLOR = 'crimson'

THESIS_LEGEND_BELOW = dict(orientation='h', xanchor='center', x=0.5,
                           yanchor='top', y=-0.20)


def mpl_legend_below(ax, ncol=4):
    """Matplotlib counterpart of THESIS_LEGEND_BELOW."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=min(ncol, len(handles)), frameon=False, fontsize='small')


def thesis_px(pt, fig_width_px, frac):
    return pt * fig_width_px / (frac * THESIS_TEXTWIDTH_IN * 72.0)


def thesis_px_sizes(fig_width_px, frac=1.0):
    """Font sizes in px, keyed by _THESIS_PT role, for one figure geometry."""
    return {k: thesis_px(v, fig_width_px, frac) for k, v in _THESIS_PT.items()}


def style_thesis(fig, fig_width_px, frac=1.0):
    """Apply the house style to a plotly figure; returns the resolved px sizes."""
    px = thesis_px_sizes(fig_width_px, frac)
    fig.update_layout(
        template='plotly_white',
        font=dict(size=px['tick'], color=_THESIS_FONT_COLOR),
        title_font_size=px['title'],
        legend_font_size=px['legend'],
        legend=THESIS_LEGEND_BELOW,
    )
    # exponentformat='power' keeps ticks in scientific notation; plotly's default
    # renders 1e-6 as "1u" and 1e4 as "10k", which is engineering shorthand
    axis_kw = dict(title_font_size=px['axis'], tickfont_size=px['tick'],
                   title_font_color=_THESIS_FONT_COLOR, tickfont_color=_THESIS_FONT_COLOR,
                   exponentformat='power', showexponent='all')
    fig.update_xaxes(**axis_kw)
    fig.update_yaxes(**axis_kw)
    for ann in fig.layout.annotations:      # subplot titles from make_subplots
        ann.font.size = px['subplot']
    return px


def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir


def panel_stem(filename):
    """'fno_model2_spectral_panel.png' -> 'fno_model2_spectral_panel'."""
    return os.path.splitext(os.path.basename(filename))[0]


def slugify(text):
    """'PINN (pure physics)' -> 'pinn_pure_physics'."""
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def save_thesis_fig(fig, outpath, width_px, height_px, frac, extra_layout=None, scale=2.0):
    """Style a plotly figure at print size and write it as PNG.

    `extra_layout` is applied after styling, so annotations passed there keep the
    font sizes set on them.
    """
    ensure_outdir(os.path.dirname(outpath) or '.')
    style_thesis(fig, width_px, frac=frac)
    layout = dict(width=width_px, height=height_px)
    if extra_layout:
        layout.update(extra_layout)
    fig.update_layout(**layout)
    try:
        fig.write_image(outpath, scale=scale)
    except Exception:
        print('  failed to write PNG; kaleido is required: pip install kaleido')
        raise
    print(f'  saved {outpath}')
    return outpath


def autocrop_white(path, pad=6):
    """Trim the white margin plotly pads around a 3D scene. No-op without Pillow.

    Only vertical slack is really removed (the surface already spans the width),
    so the image stays width-dominated and \\linewidth inclusion preserves the
    intended print font size.
    """
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
    img.crop((max(0, left - pad), max(0, top - pad),
              min(img.width, right + pad), min(img.height, bottom + pad))).save(path)


def log_marker_iters(n_iterations, count=45):
    """Log-spaced iteration marks for overlaying a closed form on a measured line."""
    return np.unique(np.geomspace(1, max(int(n_iterations), 2), count).astype(int))
