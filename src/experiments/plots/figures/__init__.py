"""Figure generation for the thesis.

    style.py      house style: palette, printed font sizes, PNG writer
    metrics.py    the error measures every panel displays
    panels.py     one function per LaTeX figure
    animation.py  the animated (.gif) solution figures

Import from this package rather than from the submodules, so the split stays an
implementation detail.
"""

from experiments.plots.figures.animation import save_solution_gif
from experiments.plots.figures.metrics import (
    band_slices, compute_relative_error, distribution_summary,
    spatial_wavenumbers, spectral_mode_index, time_relative_error_norm,
)
from experiments.plots.figures.panels import (
    plot_alpha_beta_panel, plot_resolution_panel, plot_soliton_profile,
    plot_spectral_bias_panel, plot_spectral_panel, plot_training_statistics,
)
from experiments.plots.figures.style import (
    THESIS_BAND_COLORS, THESIS_LEGEND_BELOW, THESIS_PALETTE,
    ensure_outdir, log_marker_iters, panel_stem, save_thesis_fig, slugify,
)

__all__ = [
    'THESIS_BAND_COLORS', 'THESIS_LEGEND_BELOW', 'THESIS_PALETTE',
    'band_slices', 'compute_relative_error', 'distribution_summary',
    'ensure_outdir', 'log_marker_iters', 'panel_stem', 'plot_alpha_beta_panel',
    'plot_resolution_panel', 'plot_soliton_profile', 'plot_spectral_bias_panel',
    'plot_spectral_panel', 'plot_training_statistics', 'save_solution_gif',
    'save_thesis_fig', 'slugify', 'spatial_wavenumbers', 'spectral_mode_index',
    'time_relative_error_norm',
]
