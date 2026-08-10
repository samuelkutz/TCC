"""Error measures shared by the evaluation figures.

The band split lives in `tools` because the training stages need it too; it is
re-exported here so figure code has one import for the measures it draws.
"""

import numpy as np

from tools import SPECTRAL_FLOOR, band_slices  # noqa: F401  (re-exported for figure modules)


def compute_relative_error(eta_true, eta_pred, delta=SPECTRAL_FLOOR):
    """Per-mode relative spectral error of eq:m_rel_spec, with a relative floor.

    The denominator is |eta_true| + delta * max_k |eta_true|, the maximum taken
    over modes at each time level (axis 0), so the floor tracks the amplitude the
    spectrum actually carries at that instant. An absolute floor cannot do this
    job: the reference spectrum passes through near-nulls whose amplitude is many
    orders above any fixed epsilon but negligible against the spectrum, and
    dividing a small absolute error by one of those produces a spike that is an
    artefact of the metric rather than a property of the model.
    """
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    mag = np.abs(eta_true)
    floor = delta * mag.max(axis=0, keepdims=True)
    return np.abs(eta_true - eta_pred) / (mag + floor)


def time_relative_error_norm(eta_true, eta_pred, eps=1e-12):
    """||e(.,t)||_2 / (||eta_true(.,t)||_2 + eps) at each instant; fields are (Nx, Nt).

    `eps` keeps the quotient defined for a reference that vanishes at some instant.
    It does not bias the reported numbers: for these solutions the reference norm
    stays of order one over the horizon, so the shift is a relative 1e-12.
    """
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    return (np.linalg.norm(eta_true - eta_pred, axis=0)
            / (np.linalg.norm(eta_true, axis=0) + eps))


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
