"""Scaffolding shared by the experiment stages: paths, references, band tracking,
metadata writing and the operator input layout."""

import os
from timeit import default_timer

import numpy as np
import torch

from methods.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from tools import atomic_torch_save, compute_spectral_band_errors, save_metadata_json

# training regime is named by one taxonomy everywhere: the mode string, the
# on-disk directory, and the figure folder all use these. FNO and the data-only
# MLP are the 'pure_data' members of their families and take no mode.
_MODE_DIR = {'data_and_physics': 'data_and_physics', 'pure_physics': 'pure_physics'}


def resolve_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mode_dirname(mode):
    if mode not in _MODE_DIR:
        raise ValueError(f'mode must be one of {sorted(_MODE_DIR)}, got {mode!r}')
    return _MODE_DIR[mode]


def stage_paths(results_dir, family, mode=None):
    """(weights_dir, metadata_dir) for one experiment, created if absent."""
    parts = [family] + ([mode_dirname(mode)] if mode is not None else [])
    weights_dir = os.path.join(results_dir, 'models', 'weights', *parts)
    metadata_dir = os.path.join(results_dir, 'models', 'metadata', *parts)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    return weights_dir, metadata_dir


def boussinesq_at(param_value, x_limit, t_limit, amplitude=1.0):
    """The system with coupled coefficients alpha = beta = param_value."""
    return Boussinesq(-x_limit, x_limit, 0, t_limit, param_value, param_value, amplitude)


def reference_solution(param_value, x_limit, t_limit, nx, nt, device, amplitude=1.0):
    """Pseudospectral ground truth; returns (x, t, eta, u) with eta, u of shape (nt+1, nx).

    `nt` is the number of RK4 steps, so the solver returns nt+1 time levels. Every
    caller passes it explicitly because the pointwise models and the operator
    dataset use different conventions (see train_mlp).
    """
    solver = PseudoSpectralBoussinesq(
        boussinesq_at(param_value, x_limit, t_limit, amplitude),
        Nx=nx, Nt=nt, device=device)
    return solver.solve()


class BandTracker:
    """Low/mid/high relative spectral error (eq:m_band_err), sampled during training.

    `eta_true` is (Nx, Nt): the spatial axis comes first so the FFT runs over space.
    """

    def __init__(self, eta_true):
        self.eta_true = np.asarray(eta_true, dtype=np.float64)
        self.history = {'epochs': [], 'low_band': [], 'mid_band': [], 'high_band': []}

    def record(self, epoch, eta_pred):
        low, mid, high = compute_spectral_band_errors(
            np.asarray(eta_pred, dtype=np.float64), self.eta_true)
        self.history['epochs'].append(int(epoch))
        self.history['low_band'].append(low)
        self.history['mid_band'].append(mid)
        self.history['high_band'].append(high)
        return low, mid, high

    def as_dict(self):
        return self.history


class StageTimer:
    def __init__(self):
        self.start = default_timer()

    @property
    def elapsed(self):
        return default_timer() - self.start


def write_stage_metadata(metadata, metadata_dir, label, model_name):
    """Write `<label>_model_metadata.pth` (tensors included) and `<label>_metadata.json`.

    The JSON is the .pth minus the tensor-valued `norm_stats`; the evaluation
    stages reload the .pth, the thesis text quotes the JSON.
    """
    payload = dict(metadata)
    payload.setdefault('model', model_name)
    payload.setdefault('label', label)

    pth_path = atomic_torch_save(payload, os.path.join(metadata_dir,
                                                       f'{label}_model_metadata.pth'))
    print(f'{model_name} metadata saved to {pth_path}')

    json_payload = {k: v for k, v in payload.items() if k != 'norm_stats'}
    save_metadata_json(json_payload, os.path.join(metadata_dir, f'{label}_metadata.json'))
    return pth_path


def operator_input_tensor(eta_true, u_true, param_value, nx, nt, device=None):
    """The operator input, shape (1, 4, nx, nt).

    Channels are [eta(.,0), u(.,0), alpha, beta], the first two tiled along time.
    `eta_true` and `u_true` are the solver's (Nt+1, Nx) histories. The dataset
    builder and every evaluation path go through here, so training and evaluation
    cannot desynchronize on the channel layout.
    """
    eta0 = np.asarray(eta_true[0, :], dtype=np.float32)
    u0 = np.asarray(u_true[0, :], dtype=np.float32)
    channels = np.stack([
        np.tile(eta0[:, None], (1, nt)),
        np.tile(u0[:, None], (1, nt)),
        np.full((nx, nt), param_value, dtype=np.float32),
        np.full((nx, nt), param_value, dtype=np.float32),
    ], axis=-1)
    tensor = torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0).float()
    return tensor if device is None else tensor.to(device)


def device_norm_stats(norm_stats, device):
    """Move the min/max normalization tensors onto `device`, leaving eps scalar."""
    if norm_stats is None:
        raise RuntimeError('model metadata must carry norm_stats for evaluation')
    moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in norm_stats.items()}
    moved['eps'] = norm_stats.get('eps', 1e-12)
    return moved
