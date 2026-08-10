"""Shared figure style: palette, print-sized fonts, output paths, PNG writer."""

import os
import re

import numpy as np


# A plotly figure of logical width W px included at frac*\textwidth prints a font
# of f px at f * frac * THESIS_TEXTWIDTH_IN * 72 / W points. thesis_px inverts
# that, so in-figure text lands at a chosen printed point size regardless of the
# figure's pixel dimensions. The document's text block is ~160 mm wide.
THESIS_TEXTWIDTH_IN = 6.30
_THESIS_PT = dict(tick=6.0, axis=7.0, subplot=8.0, title=9.0, legend=6.5, annot=6.0)

# black, to match the LaTeX body text; plotly's default is a lighter slate blue
_THESIS_FONT_COLOR = '#000000'

# Okabe--Ito, assigned in this fixed order and never cycled. Only the first six
# steps are kept: measured against a white page they all clear the lightness band,
# the chroma floor, and a colour-vision-deficiency separation of dE >= 9.6 on the
# worst adjacent pair. The two remaining Okabe--Ito steps do not. The yellow
# #F0E442 sits at L 0.90, far above the band, and reaches only 1.29:1 against the
# page, so a line in it is close to invisible in print; #000000 has no chroma at
# all and reads as the axis rather than as a series. Six is therefore the ceiling:
# a seventh series wants small multiples or a composite encoding, not a new hue.
THESIS_PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                  '#CC79A7', '#56B4E9']
THESIS_BAND_COLORS = ['#0072B2', '#E69F00', '#D55E00']  # low, mid, high

THESIS_REFERENCE_COLOR = '#222222'
THESIS_PREDICTION_COLOR = '#ff7f0e'
THESIS_ERROR_COLOR = 'crimson'

THESIS_LEGEND_ABOVE = dict(orientation='h', xanchor='center', x=0.5,
                           yanchor='bottom', y=1.02)


def mpl_legend_above(ax, ncol=4):
    """Matplotlib counterpart of THESIS_LEGEND_ABOVE."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
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
        legend=THESIS_LEGEND_ABOVE,
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


def _data_log_spans(fig):
    """Decades spanned by the positive x and y data across every trace."""
    spans = {}
    for kind in ('x', 'y'):
        vals = []
        for tr in fig.data:
            v = getattr(tr, kind, None)
            if v is None:
                continue
            arr = np.asarray(v, dtype=float).ravel()
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size:
                vals.append((arr.min(), arr.max()))
        if vals:
            lo = min(v[0] for v in vals)
            hi = max(v[1] for v in vals)
            spans[kind] = np.log10(hi) - np.log10(lo)
    return spans


def _axis_log_span(axis, data_span):
    """Decades the axis shows: its explicit range if set, else the data's."""
    if axis.range is not None and len(axis.range) == 2:
        return abs(float(axis.range[1]) - float(axis.range[0]))   # already log10
    return data_span


def _tidy_ticks(fig):
    """Tick formatting that depends on whether the axis is logarithmic.

    Two defaults misfire at thesis font sizes. Plotly labels the 2x and 5x minor
    ticks of a log axis, which collide with the decade labels and read as noise
    ("10^2 2 5 10^3 2 5"), so log axes are cut to whole decades. And
    `exponentformat='power'` on a *linear* axis turns an epoch count into
    "0.2x10^4"; those labels are long enough to be rotated and then overlap, so
    linear axes get plain separated thousands instead.
    """
    spans = _data_log_spans(fig)
    for kind, axis in ([('x', a) for a in fig.select_xaxes()]
                       + [('y', a) for a in fig.select_yaxes()]):
        if axis.type == 'log':
            # one label per decade only while the axis is short. The kernel
            # spectrum runs fifteen decades, and a label on each of them overlaps
            # into an unreadable stack, so the step widens with the span.
            span = _axis_log_span(axis, spans.get(kind))
            axis.dtick = max(1, int(np.ceil(span / 6.0))) if span else 1
            axis.exponentformat = 'power'
        else:
            axis.exponentformat = 'none'
            axis.separatethousands = True
        # horizontal labels or none: left to itself plotly rotates them to 45
        # degrees once they crowd, which costs vertical space and is harder to
        # read than simply showing fewer ticks
        axis.tickangle = 0
        if axis.nticks in (None, 0):
            axis.nticks = 6


def _legend_rows(fig, legend_font_px, plot_width_px):
    """How many rows a horizontal legend wraps into, from its entry labels."""
    labels = [str(tr.name) for tr in fig.data
              if getattr(tr, 'showlegend', None) is not False and tr.name]
    if not labels:
        return 0
    # a glyph is roughly 0.55 em wide; each entry also carries its marker and gap
    widths = [(len(lab) * 0.55 + 4.0) * legend_font_px for lab in labels]
    rows, used = 1, 0.0
    for w in widths:
        if used + w > plot_width_px and used > 0:
            rows, used = rows + 1, w
        else:
            used += w
    return rows


def save_thesis_fig(fig, outpath, width_px, height_px, frac, extra_layout=None, scale=2.0):
    """Style a plotly figure at print size and write it as PNG.

    `extra_layout` is applied after styling, so annotations passed there keep the
    font sizes set on them.
    """
    ensure_outdir(os.path.dirname(outpath) or '.')
    px = style_thesis(fig, width_px, frac=frac)
    layout = dict(width=width_px, height=height_px)
    if extra_layout:
        layout.update(extra_layout)
    fig.update_layout(**layout)
    _tidy_ticks(fig)
    # the legend sits above the axes, so the top margin has to clear however many
    # rows it wraps into; a fixed margin clips the last row of a six-entry legend
    if fig.layout.showlegend is not False:
        rows = _legend_rows(fig, px['legend'], width_px)
        if rows:
            margin = fig.layout.margin
            need = 14 + rows * px['legend'] * 1.9
            if margin.t is None or margin.t < need:
                fig.update_layout(margin=dict(t=need))
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
