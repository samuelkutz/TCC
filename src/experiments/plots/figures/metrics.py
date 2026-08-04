"""Error measures shared by the evaluation figures.

The band split lives in `tools` because the training stages need it too; it is
re-exported here so figure code has one import for the measures it draws.
"""

import numpy as np

from tools import band_slices  # noqa: F401  (re-exported for figure modules)


def compute_relative_error(eta_true, eta_pred):
    """Per-mode relative error |eta_true - eta_pred| / |eta_true|.

    Used for the spectral panel, whose reference amplitudes stay well away from
    zero over the retained modes, so no denominator floor is needed. The
    non-finite guard only cleans an exact-zero reference mode, should one occur.
    """
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(eta_true - eta_pred) / np.abs(eta_true)
    return np.nan_to_num(rel_error, nan=0.0, posinf=0.0, neginf=0.0)


def time_relative_error_norm(eta_true, eta_pred, eps=1e-12):
    """||e(.,t)||_2 / ||eta_true(.,t)||_2 at each instant; fields are (Nx, Nt)."""
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    return (np.linalg.norm(eta_true - eta_pred, axis=0)
            / (np.linalg.norm(eta_true, axis=0) + eps))


def spatial_wavenumbers(x):
    """kx = n*pi/L on the domain [-L, L]."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)
    L = (x[-1] - x[0]) / 2.0
    return np.arange(len(x) // 2 + 1, dtype=float) * np.pi / L


def spectral_mode_index(x):
    """kx rescaled by 2L/pi, so the axis reads as integer mode numbers."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)
    L = (x[-1] - x[0]) / 2.0
    return spatial_wavenumbers(x) * (2.0 * L / np.pi)


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
