import os
import json
import random
import numpy as np
import torch


def set_seed(seed):
    # seed every random source and put the cuDNN backend in deterministic mode, so
    # that a stage re-run in isolation reproduces its results regardless of how much
    # randomness earlier pipeline stages consumed. Call this at the entry of any
    # training or diagnostic routine that must be reproducible on its own.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, filepath):
    # save model state dict to disk
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f'model saved to {filepath}')
    return filepath


def load_model(filepath, model, device):
    # load model weights and set network to evaluation mode
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_dataset(x_train, y_train, filepath, norm_stats=None):
    # save dataset tensors and optional normalization statistics to a pytorch .pth file
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    payload = {
        'x_train': x_train,
        'y_train': y_train,
    }
    if norm_stats is not None:
        payload['norm_stats'] = norm_stats
    torch.save(payload, filepath)
    print(f"dataset saved to {filepath}")


def load_dataset(filepath):
    # load dataset tensors and optional normalization statistics from a pytorch .pth file
    data = torch.load(filepath)
    x_train = data['x_train']
    y_train = data['y_train']
    norm_stats = data.get('norm_stats', None)
    return x_train, y_train, norm_stats


def compute_norm_stats(x_train, y_train, eps=1e-12):
    # compute channel-wise min/max ranges for input and output tensors
    input_min = x_train.amin(dim=(0, 2, 3), keepdim=True)
    input_max = x_train.amax(dim=(0, 2, 3), keepdim=True)
    output_min = y_train.amin(dim=(0, 2, 3), keepdim=True)
    output_max = y_train.amax(dim=(0, 2, 3), keepdim=True)
    return {
        'input_min': input_min,
        'input_max': input_max,
        'output_min': output_min,
        'output_max': output_max,
        'eps': eps,
    }


def normalize_tensor(tensor, min_val, max_val, eps=1e-12):
    # map tensor values to [0, 1] range per channel: (x - min) / (max - min)
    return (tensor - min_val) / (max_val - min_val + eps)


def unnormalize_tensor(tensor, min_val, max_val, eps=1e-12):
    # map tensor from [0, 1] back to original scale: x * (max - min) + min
    return tensor * (max_val - min_val + eps) + min_val


def normalize_dataset(x_train, y_train, norm_stats):
    x_norm = normalize_tensor(x_train, norm_stats['input_min'], norm_stats['input_max'], norm_stats['eps'])
    y_norm = normalize_tensor(y_train, norm_stats['output_min'], norm_stats['output_max'], norm_stats['eps'])
    return x_norm, y_norm


class L2_loss(object):
    # standard mean-squared error loss for model output compared to target data
    def __init__(self, eps=1e-10):
        super(L2_loss, self).__init__()
        self.eps = eps

    def __call__(self, x, y):
        diff = x - y
        loss = torch.mean(diff * diff)
        return loss


def save_metadata_json(metadata, filepath):
    # save json-serializable training metadata to disk
    def _coerce(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
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

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(_coerce(metadata), f, indent=2)
    print(f'metadata json saved to {filepath}')


def compute_spectral_band_errors(eta_pred, eta_true):
    # relative spectral error per spatial mode (rfft along axis=0), averaged over time, split into 3 bands
    error = eta_pred - eta_true
    fft_err = np.abs(np.fft.rfft(error, axis=0))    # (Nk, Nt)
    fft_ref = np.abs(np.fft.rfft(eta_true, axis=0)) # (Nk, Nt)
    rel_per_mode = fft_err.mean(axis=1) / (fft_ref.mean(axis=1) + 1e-12)  # (Nk,)
    Nk = rel_per_mode.shape[0]
    return (
        float(rel_per_mode[:Nk // 4].mean()),          # low  k in [0, Nk/4)
        float(rel_per_mode[Nk // 4: Nk // 2].mean()),  # mid  k in [Nk/4, Nk/2)
        float(rel_per_mode[Nk // 2:].mean()),           # high k in [Nk/2, Nk)
    )