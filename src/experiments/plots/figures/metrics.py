"""Error measures shared by the evaluation figures.

The band split lives in `tools` because the training stages need it too; it is
re-exported here so figure code has one import for the measures it draws.
"""

import numpy as np

from tools import band_slices, spectral_amplitude  # noqa: F401  (re-exported for figure modules)


def compute_spectral_error(true_spec, pred_spec):
    """Per-mode absolute spectral error of eq:m_abs_spec.

    Both arguments are amplitude spectra already normalized by N_x (see
    `tools.spectral_amplitude`), so the difference is read in the units of eta
    and does not change when the grid is refined. There is no denominator and
    therefore no floor: the reference spectrum passes through near-nulls, and
    dividing by one of those turns an unremarkable amplitude error into a spike
    that says nothing about the model.
    """
    true_spec = np.asarray(true_spec, dtype=float)
    pred_spec = np.asarray(pred_spec, dtype=float)
    return np.abs(true_spec - pred_spec)


def time_relative_error_norm(eta_true, eta_pred):
    """||e(.,t)||_2 / ||eta_true(.,t)||_2 at each instant; fields are (Nx, Nt).

    The reference is a travelling wave of fixed amplitude on a periodic domain,
    so its spatial norm stays of order one over the whole horizon and the
    quotient needs no regularization.
    """
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    return (np.linalg.norm(eta_true - eta_pred, axis=0)
            / np.linalg.norm(eta_true, axis=0))


def _half_width(x):
    """L for a periodic grid on [-L, L].

    The grid drops its right endpoint, so it spans 2L - dx rather than 2L and the
    missing cell has to be added back before halving; reading L off the samples
    alone would understate it by dx/2.
    """
    return (x[-1] - x[0] + (x[1] - x[0])) / 2.0


def spatial_wavenumbers(x):
    """kx = n*pi/L on the domain [-L, L], one per retained real-DFT mode."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)
    return np.arange(len(x) // 2 + 1, dtype=float) * np.pi / _half_width(x)


def spectral_mode_index(x):
    """kx rescaled by L/pi, so the axis reads as the real-DFT bin number n.

    The fundamental of a 2L-periodic field is pi/L, so n = kx * L/pi and the axis
    runs 0 .. Nx/2, matching the bins `np.fft.rfft` returns.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)
    return spatial_wavenumbers(x) * (_half_width(x) / np.pi)


def distribution_summary(values):
    """The five numbers a box plot draws, so the text can quote them."""
    values = np.asarray(values, dtype=float)
    return {
        'median': float(np.median(values)),
        'mean': float(np.mean(values)),
        'q1': float(np.percentile(values, 25)),
        'q3': float(np.percentile(values, 75)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }
