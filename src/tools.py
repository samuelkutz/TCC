"""Seeding, model/dataset I/O, min-max normalization and the band-error metric."""

import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed every random source and put cuDNN in deterministic mode.

    Called once by `main`, and again at the entry of any stage that must
    reproduce on its own regardless of what earlier stages drew from the RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_log_epoch(epoch_index, epochs, print_interval):
    """`epoch_index` is 0-based; true every print_interval epochs and on the last."""
    return (epoch_index + 1) % print_interval == 0 or epoch_index == epochs - 1


def log_epoch(epoch, epochs, elapsed, **losses):
    """One progress format for every training loop."""
    parts = ', '.join(f'{k} {v:.4e}' for k, v in losses.items())
    print(f'  epoch {epoch}/{epochs}, elapsed {elapsed:.1f}s, {parts}')


def atomic_torch_save(payload, filepath):
    """torch.save via a temporary file and os.replace.

    The pipeline is one long sequential run in which later stages load what
    earlier ones wrote, so a crash mid-write must not leave a truncated .pth that
    a plotting stage would then load as if it were complete.
    """
    filepath = os.fspath(filepath)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    tmp = f'{filepath}.tmp'
    try:
        torch.save(payload, tmp)
        os.replace(tmp, filepath)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return filepath


def save_model(model, filepath):
    atomic_torch_save(model.state_dict(), filepath)
    print(f'model saved to {filepath}')
    return filepath


def load_model(filepath, model, device):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_dataset(x_train, y_train, filepath, norm_stats=None):
    payload = {'x_train': x_train, 'y_train': y_train}
    if norm_stats is not None:
        payload['norm_stats'] = norm_stats
    atomic_torch_save(payload, filepath)
    print(f'dataset saved to {filepath}')
    return filepath


def load_dataset(filepath):
    data = torch.load(filepath, weights_only=False)
    return data['x_train'], data['y_train'], data.get('norm_stats', None)


def compute_norm_stats(x_train, y_train, amplitude, param_min, param_max, eps=1e-12):
    """Fixed, data-free channel-wise normalization constants.

    The scale is set from quantities the problem already provides, never from the
    reference solution fields, so a physics-only (no-data) model inherits no
    information about the dataset. The elevation and velocity channels, and the
    initial-condition inputs that feed them, use the known initial amplitude A;
    the parameter channels alpha, beta use the known sampling range. Inputs are
    ordered [eta(.,0), u(.,0), alpha, beta], outputs [eta, u]. The tensors borrow
    x_train / y_train's device and dtype but none of their values.
    """
    a = float(amplitude)
    pmin, pmax = float(param_min), float(param_max)
    input_min = x_train.new_tensor([-a, -a, pmin, pmin]).reshape(1, -1, 1, 1)
    input_max = x_train.new_tensor([a, a, pmax, pmax]).reshape(1, -1, 1, 1)
    output_min = y_train.new_tensor([-a, -a]).reshape(1, -1, 1, 1)
    output_max = y_train.new_tensor([a, a]).reshape(1, -1, 1, 1)
    return {
        'input_min': input_min,
        'input_max': input_max,
        'output_min': output_min,
        'output_max': output_max,
        'eps': eps,
    }


def normalize_tensor(tensor, min_val, max_val, eps=1e-12):
    """(x - min) / (max - min); eps guards a constant channel."""
    return (tensor - min_val) / (max_val - min_val + eps)


def unnormalize_tensor(tensor, min_val, max_val, eps=1e-12):
    return tensor * (max_val - min_val + eps) + min_val


def normalize_dataset(x_train, y_train, norm_stats):
    return (normalize_tensor(x_train, norm_stats['input_min'], norm_stats['input_max'],
                             norm_stats['eps']),
            normalize_tensor(y_train, norm_stats['output_min'], norm_stats['output_max'],
                             norm_stats['eps']))


def save_metadata_json(metadata, filepath):
    """Write metadata as JSON, coercing numpy and torch values to plain Python."""
    def _coerce(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(v) for v in obj]
        return obj

    filepath = os.fspath(filepath)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    tmp = f'{filepath}.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(_coerce(metadata), f, indent=2)
        os.replace(tmp, filepath)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f'metadata json saved to {filepath}')
    return filepath


def band_slices(n_k):
    """The low / mid / high wavenumber split of eq:m_bands.

    Defined once here because both the training stages and the figures split the
    spectrum, and the bands have to mean the same thing in every number the
    thesis quotes.
    """
    return (slice(0, n_k // 4), slice(n_k // 4, n_k // 2), slice(n_k // 2, n_k))


def compute_spectral_band_errors(eta_pred, eta_true):
    """Band-averaged relative spectral error (eq:m_band_err), the time-average of
    the per-mode E_spec of eq:m_rel_spec.

    Fields are (Nx, Nt); the rfft runs over axis 0 (space). For every mode and
    time level the relative spectral error | |eta_pred_hat| - |eta_true_hat| | /
    |eta_true_hat| is formed, averaged over the time levels, then averaged within
    each of the three bands of eq:m_bands. Returns (low, mid, high).
    """
    pred_hat = np.abs(np.fft.rfft(eta_pred, axis=0))    # (Nk, Nt)
    true_hat = np.abs(np.fft.rfft(eta_true, axis=0))    # (Nk, Nt)
    with np.errstate(divide='ignore', invalid='ignore'):
        e_spec = np.abs(pred_hat - true_hat) / true_hat  # eq:m_rel_spec per (k, t_n)
    e_spec = np.nan_to_num(e_spec, nan=0.0, posinf=0.0, neginf=0.0)
    rel_per_mode = e_spec.mean(axis=1)                   # average E_spec over time
    return tuple(float(rel_per_mode[s].mean())
                 for s in band_slices(rel_per_mode.shape[0]))


def error_spectrum(e):
    """|ehat_k| of a stack of error vectors; `e` is (n_samples, N_x), rfft over x."""
    return torch.fft.rfft(e, dim=1).abs()


def band_relative_error(pred_spectrum, reference_spectrum, eps=1e-12):
    """Band-averaged E_spec (eq:m_rel_spec), one curve per band.

    `pred_spectrum` is (n_samples, N_k) = |rfft(predicted field)| and
    `reference_spectrum` is (N_k,) = |rfft(reference field)|. Each band is the
    mean over its modes of | |pred_hat| - |ref_hat| | / |ref_hat|, the same
    relative spectral error the six-model band tracking uses.
    """
    ref = reference_spectrum.unsqueeze(0)
    e_spec = (pred_spectrum - ref).abs() / (ref + eps)
    return [e_spec[:, s].mean(dim=1) for s in band_slices(pred_spectrum.shape[1])]
