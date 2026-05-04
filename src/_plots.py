import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable



def _ensure_outdir(outdir):
    # make sure output directory exists before saving files
    os.makedirs(outdir, exist_ok=True)
    return outdir


# plot training loss
def plot_training_loss(train_loss_history, outdir, filename, duration_seconds=None, final_loss=None, num_params=None):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    epochs = np.arange(1, len(train_loss_history) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss_history, color='#1f77b4', lw=2.2)
    ax.set_title('Training L2 loss history', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 loss', fontsize=12)
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


# plot training statistics across runs
def plot_training_statistics(histories, labels, outdir, filename, log_scale=True, duration_seconds=None, final_loss=None, num_params=None):
    # compare multiple loss curves for the given runs
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    epochs = np.arange(1, len(histories[0]) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for history, label in zip(histories, labels):
        ax.plot(epochs, history, alpha=0.85, lw=2.0, label=label)

    ax.set_title('Training L2 loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    if log_scale:
        ax.set_yscale('log')
    if len(labels) > 1:
        ax.legend(fontsize='small', loc='upper right')

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
    print(f'training statistics plot saved to {outpath}')
    plt.close(fig)


# spectral relative error metrics
def compute_spectral_relative_error(eta_true, eta_pred):
    # compute relative error in fourier spectrum, using rfft2 over space-time
    true_fft = np.fft.rfft2(eta_true)
    pred_fft = np.fft.rfft2(eta_pred)
    diff_fft = true_fft - pred_fft
    rel_error = np.linalg.norm(diff_fft) / (np.linalg.norm(true_fft) + 1e-12)
    return float(rel_error)


def compute_relative_error(eta_true, eta_pred, floor_ratio=1e-2, floor_min=1e-3):
    """compute a robust relative error map.

    this avoids huge spikes from dividing by values near zero in the true field.
    """
    # relative error = |eta_true - eta_pred| / max(|eta_true|, floor)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    abs_true = np.abs(eta_true)
    floor = np.maximum(np.max(abs_true) * floor_ratio, floor_min)
    denom = np.maximum(abs_true, floor)
    rel_error = np.abs(eta_true - eta_pred) / denom
    return np.nan_to_num(rel_error, posinf=1e3, neginf=0.0)


def spatial_wavenumbers(x):
    """return spatial wavenumbers kx normalized as n * π / L.

    This maps the FFT frequency index to the standard integer-indexed
    wavenumbers for a domain [-L, L].
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)

    domain_length = x[-1] - x[0]
    L = domain_length / 2.0
    n_freq = len(x) // 2 + 1
    return np.arange(n_freq, dtype=float) * np.pi / L


def spectral_mode_index(x):
    """return spectral mode indices scaled to the spatial resolution.

    This rescales the physical wavenumber axis by 2L/pi so that the
    displayed values follow the discrete mode index convention.
    """
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
    axes[0].set_title('True spatial spectrum over time')
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
    axes[1].set_title('Relative spatial spectrum error')
    axes[1].set_xlabel('Spectral index n')
    plt.colorbar(im1, ax=axes[1], label='Relative Error')

    fig.suptitle(title)
    plt.savefig(outpath, dpi=150)
    print(f'spectral summary saved to {outpath}')
    plt.close()


def plot_relative_error_panel(x, t, eta_true, eta_pred, times, outdir, filename, title):
    # plot solution, pointwise error, and spectrum comparison for selected times
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

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=False)

    for idx, time_index in enumerate(indices):
        ax = axes[0, idx]
        ax.plot(x, eta_true[:, time_index], color='black', lw=1.8, label='True')
        ax.plot(x, eta_pred[:, time_index], color='#ff7f0e', lw=1.8, linestyle='--', label='predicted')
        ax.set_title(f'Prediction at t={t[time_index]:.2f}')
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('η(x,t)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize='small', loc='best')

    im = axes[1, 0].imshow(
        rel_error.T,
        extent=[x[0], x[-1], t[0], t[-1]],
        origin='lower',
        aspect='auto',
        cmap='magma',
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title('Relative error heatmap')
    axes[1, 0].set_xlabel('Space (x)')
    axes[1, 0].set_ylabel('Time (t)')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Relative Error')

    time_plot = t + 5.0
    axes[1, 1].plot(time_plot, time_relative_norm, color='crimson', lw=2)
    axes[1, 1].axvline(
        x=time_plot[-1],
        color='black',
        linestyle='--',
        linewidth=1.0,
        alpha=0.7,
        label='evaluation stop',
    )
    axes[1, 1].set_title('Time-resolved relative error norm')
    axes[1, 1].set_xlabel('Time (t + 5)')
    axes[1, 1].set_ylabel('Relative error')
    axes[1, 1].grid(True, alpha=0.3)
    mean_err = np.mean(time_relative_norm)
    axes[1, 1].text(
        0.5,
        0.9,
        f'Mean: {mean_err:.2e}',
        transform=axes[1, 1].transAxes,
        ha='center',
        va='center',
        bbox=dict(facecolor='white', alpha=0.8),
    )

    im2 = axes[2, 0].imshow(
        true_spec.T,
        extent=[kx[0], kx[-1], t[0], t[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
    )
    axes[2, 0].set_title('True spatial spectrum over time')
    axes[2, 0].set_xlabel('Spectral index n')
    axes[2, 0].set_ylabel('Time')
    divider2 = make_axes_locatable(axes[2, 0])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Amplitude')

    im3 = axes[2, 1].imshow(
        rel_spec.T,
        extent=[kx[0], kx[-1], t[0], t[-1]],
        origin='lower',
        aspect='auto',
        cmap='inferno',
        vmin=0.0,
        vmax=1.0,
    )
    axes[2, 1].set_title('Relative spatial spectrum error')
    axes[2, 1].set_xlabel('Spectral index n')
    divider3 = make_axes_locatable(axes[2, 1])
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im3, cax=cax3, label='Relative Error')

    fig.suptitle(title)
    fig.subplots_adjust(top=0.95, bottom=0.03, hspace=0.35, wspace=0.28)
    fig.savefig(outpath, dpi=150)
    print(f'relative error summary saved to {outpath}')
    plt.close()


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
        axes[row, 0].set_ylabel('η(x,t)')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)

        rel_error_line = compute_relative_error(eta_true[:, idx], eta_pred[:, idx])
        axes[row, 1].plot(x, rel_error_line, color='crimson')
        axes[row, 1].set_title(f'Relative error at t = {t[idx]:.2f}')
        axes[row, 1].set_ylabel('Relative error')
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Space (x)')
    axes[-1, 1].set_xlabel('Space (x)')
    fig.suptitle(title)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98)
    plt.savefig(outpath, dpi=150)
    print(f'solution snapshots saved to {outpath}')
    plt.close()


def plot_stacked_solution_curves(x, t, eta_true, eta_pred, outdir, filename, title,
                                 n_curves=5, offset_factor=1.3):
    # stacked 2d curves for a few time slices, matching the gif style with reference solution
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    if len(t) < n_curves:
        indices = np.arange(len(t), dtype=int)
    else:
        fractions = np.linspace(0.0, 1.0, n_curves)
        indices = [int(np.round(frac * (len(t) - 1))) for frac in fractions]

    amp_true = np.max(eta_true) - np.min(eta_true)
    amp_pred = np.max(eta_pred) - np.min(eta_pred)
    amplitude = max(amp_true, amp_pred, 1.0)
    offset = amplitude * offset_factor

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, idx in enumerate(indices):
        baseline = i * offset
        ax.plot(x, eta_true[:, idx] + baseline, color='black', lw=1.8,
                label='Reference' if i == 0 else '')
        ax.plot(x, eta_pred[:, idx] + baseline, color='#1f77b4', lw=1.5,
                linestyle='--', label='Predicted' if i == 0 else '')
        ax.hlines(baseline, x[0], x[-1], color='gray', alpha=0.25, linewidth=0.7)

    y_ticks = [i * offset for i in range(len(indices))]
    y_tick_labels = [f'{t[idx]:.2f}' for idx in indices]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_title(title)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.25)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98)
    fig.savefig(outpath, dpi=150)
    print(f'stacked solution curves saved to {outpath}')
    plt.close(fig)


def _five_time_slice_indices_labels(t_array):
    """return indices and labels for t=min, 1/4, 1/2, 3/4, max."""
    t_array = np.asarray(t_array, dtype=float)
    if t_array.size == 0:
        return [], []

    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    tags = ['min', '1/4', '1/2', '3/4', 'max']
    idx_labels = []
    for frac, tag in zip(fractions, tags):
        target = t_array[0] + frac * (t_array[-1] - t_array[0])
        idx = int(np.argmin(np.abs(t_array - target)))
        idx_labels.append((idx, tag))

    seen = set()
    indices = []
    labels = []
    for idx, tag in idx_labels:
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
        labels.append(tag)
    return indices, labels


def plot_model2_alpha_beta_panel(x, t, eta_true_list, eta_pred_list, param_values, outdir, filename, title,
                                 n_curves=5, offset_factor=1.3, eval_resolution=None, res_label=None):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    param_values = np.asarray(param_values, dtype=float)
    n_params = len(param_values)
    if n_params == 0:
        raise ValueError('param_values must contain at least one value')

    indices, _ = _five_time_slice_indices_labels(t)

    amplitude = 0.0
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        amplitude = max(amplitude, np.max(eta_true) - np.min(eta_true), np.max(eta_pred) - np.min(eta_pred))
    amplitude = max(amplitude, 1.0)
    offset = amplitude * offset_factor

    time_rel_norms = []
    mean_rel_errors = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)
        mean_rel_errors.append(np.mean(rel_norm))

    fig = plt.figure(figsize=(14, 4 * n_params + 4))
    gs = fig.add_gridspec(n_params + 1, 2, height_ratios=[1] * n_params + [0.8], hspace=0.4, wspace=0.3)

    for i, (alpha, eta_true, eta_pred) in enumerate(zip(param_values, eta_true_list, eta_pred_list)):
        ax_left = fig.add_subplot(gs[i, 0])
        ax_right = fig.add_subplot(gs[i, 1])

        for j, idx in enumerate(indices):
            baseline = j * offset
            ax_left.plot(x, eta_true[:, idx] + baseline, color='#2b1b17', lw=1.6,
                         label='Reference' if j == 0 else '')
            ax_left.plot(x, eta_pred[:, idx] + baseline, color='#d94801', lw=1.4, linestyle='--',
                         label='Predicted' if j == 0 else '')
            ax_left.hlines(baseline, x[0], x[-1], color='#8c564b', alpha=0.25, linewidth=0.7)
        y_ticks = [j * offset for j in range(len(indices))]
        y_tick_labels = [f'{t[idx]:.2f}' if idx < len(t) else '' for idx in indices]
        ax_left.set_yticks(y_ticks)
        ax_left.set_yticklabels(y_tick_labels)
        ax_left.set_title(f'Stacked solutions for α = β = {alpha:.3f}')
        ax_left.set_xlabel('Space (x)')
        ax_left.set_ylabel('Time (t)')
        ax_left.grid(True, alpha=0.2)
        # show evaluation resolution label if provided (prefer explicit res_label)
        if res_label is not None:
            res_text = str(res_label)
        elif eval_resolution is not None:
            res_text = f'{int(eval_resolution)}'
        else:
            res_text = None
        if res_text is not None:
            ax_left.text(
                0.98,
                0.02,
                f'eval res = {res_text}',
                transform=ax_left.transAxes,
                ha='right',
                va='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            )
        if i == 0:
            ax_left.legend(loc='upper left', fontsize='small')

        ax_right.plot(t, time_rel_norms[i], lw=1.8, color='#b22222')
        ax_right.set_title(f'Relative error over time for α = β = {alpha:.3f}')
        ax_right.set_xlabel('Time (t)')
        ax_right.set_ylabel('Relative error')
        ax_right.grid(True, alpha=0.2)

    ax_bottom = fig.add_subplot(gs[-1, :])
    box = ax_bottom.boxplot(
        time_rel_norms,
        positions=param_values,
        widths=np.maximum(0.05 * (param_values.max() - param_values.min()), 0.05),
        patch_artist=True,
        showmeans=True,
        meanline=True,
        labels=[f'{val:.2f}' for val in param_values],
    )
    for patch in box['boxes']:
        patch.set_facecolor('#fdd0a2')
        patch.set_edgecolor('#e6550d')
        patch.set_alpha(0.7)
    ax_bottom.set_title('Relative error distribution vs. α = β')
    ax_bottom.set_xlabel('α = β value')
    ax_bottom.set_ylabel('Relative error')
    ax_bottom.grid(True, alpha=0.2)
    ax_bottom.set_xticks(param_values)
    ax_bottom.set_xticklabels([f'{val:.2f}' for val in param_values], rotation=15, ha='right')

    suptitle = title
    if res_label is not None:
        suptitle = f"{suptitle} (eval res = {res_label})"
    elif eval_resolution is not None:
        suptitle = f'{suptitle} (eval res = {int(eval_resolution)})'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.98, hspace=0.35, wspace=0.3)
    fig.savefig(outpath, dpi=150)
    print(f'alpha/beta panel saved to {outpath}')
    plt.close(fig)


def plot_model2_resolution_panel(x_list, t_list, eta_true_list, eta_pred_list, resolutions, outdir, filename, title,
                                 n_curves=5, offset_factor=1.3, eval_alpha_beta=None, param_label=None):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x_list = [np.asarray(x, dtype=float) for x in x_list]
    t_list = [np.asarray(t, dtype=float) for t in t_list]
    resolutions = np.asarray(resolutions, dtype=float)
    n_res = len(resolutions)
    if n_res == 0:
        raise ValueError('resolutions must contain at least one value')
    if len(x_list) != n_res or len(t_list) != n_res:
        raise ValueError('x_list and t_list must match the number of resolutions')

    amplitude = 0.0
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        amplitude = max(amplitude, np.max(eta_true) - np.min(eta_true), np.max(eta_pred) - np.min(eta_pred))
    amplitude = max(amplitude, 1.0)
    offset = amplitude * offset_factor

    time_rel_norms = []
    mean_rel_errors = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)
        mean_rel_errors.append(np.mean(rel_norm))

    fig = plt.figure(figsize=(14, 4 * n_res + 4))
    gs = fig.add_gridspec(n_res + 1, 2, height_ratios=[1] * n_res + [0.8], hspace=0.4, wspace=0.3)

    for i, (res, x_res, t_res, eta_true, eta_pred) in enumerate(zip(resolutions, x_list, t_list, eta_true_list, eta_pred_list)):
        ax_left = fig.add_subplot(gs[i, 0])
        ax_right = fig.add_subplot(gs[i, 1])

        indices, _ = _five_time_slice_indices_labels(t_res)

        for j, idx in enumerate(indices):
            if idx >= eta_true.shape[1]:
                continue
            baseline = j * offset
            ax_left.plot(x_res, eta_true[:, idx] + baseline, color='#2b1b17', lw=1.6,
                         label='Reference' if j == 0 else '')
            ax_left.plot(x_res, eta_pred[:, idx] + baseline, color='#d94801', lw=1.4, linestyle='--',
                         label='Predicted' if j == 0 else '')
            ax_left.hlines(baseline, x_res[0], x_res[-1], color='#8c564b', alpha=0.25, linewidth=0.7)
        y_ticks = [j * offset for j in range(len(indices))]
        y_tick_labels = [f'{t_res[idx]:.2f}' if idx < len(t_res) else '' for idx in indices]
        ax_left.set_yticks(y_ticks)
        ax_left.set_yticklabels(y_tick_labels)
        ax_left.set_title(f'Stacked solutions for res = {int(res)}')
        ax_left.set_xlabel('Space (x)')
        ax_left.set_ylabel('Time (t)')
        ax_left.grid(True, alpha=0.2)
        # show evaluated alpha/beta label if provided (prefer explicit param_label)
        if param_label is not None:
            p_text = str(param_label)
        elif eval_alpha_beta is not None:
            p_text = f'{eval_alpha_beta:.3f}'
        else:
            p_text = None
        if p_text is not None:
            ax_left.text(
                0.98,
                0.02,
                f'eval α=β = {p_text}',
                transform=ax_left.transAxes,
                ha='right',
                va='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            )
        if i == 0:
            ax_left.legend(loc='upper left', fontsize='small')

        ax_right.plot(t_res, time_rel_norms[i], lw=1.8, color='#b22222')
        ax_right.set_title(f'Relative error over time for res = {int(res)}')
        ax_right.set_xlabel('Time (t)')
        ax_right.set_ylabel('Relative error')
        ax_right.grid(True, alpha=0.2)

    ax_bottom = fig.add_subplot(gs[-1, :])
    box = ax_bottom.boxplot(
        time_rel_norms,
        positions=resolutions,
        widths=np.maximum(0.05 * (resolutions.max() - resolutions.min()), 1.0),
        patch_artist=True,
        showmeans=True,
        meanline=True,
        labels=[str(int(res)) for res in resolutions],
    )
    for patch in box['boxes']:
        patch.set_facecolor('#fdae6b')
        patch.set_edgecolor('#e6550d')
        patch.set_alpha(0.7)
    ax_bottom.set_title('Relative error distribution vs. resolution')
    ax_bottom.set_xlabel('Resolution')
    ax_bottom.set_ylabel('Relative error')
    ax_bottom.grid(True, alpha=0.2)
    ax_bottom.set_xticks(resolutions)
    ax_bottom.set_xticklabels([str(int(res)) for res in resolutions], rotation=15, ha='right')

    suptitle = title
    if param_label is not None:
        suptitle = f"{suptitle} (eval α=β = {param_label})"
    elif eval_alpha_beta is not None:
        suptitle = f'{suptitle} (eval α=β = {eval_alpha_beta:.3f})'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.98, hspace=0.35, wspace=0.3)
    fig.savefig(outpath, dpi=150)
    print(f'resolution panel saved to {outpath}')
    plt.close(fig)


def plot_model2_spectral_panel(x, t, eta_true, eta_pred, outdir, filename, title,
                               n_times=3, eval_resolution=None, res_label=None):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    kx = spectral_mode_index(x)

    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    abs_err = np.abs(pred_spec - true_spec)
    mean_abs_err_over_time = np.mean(abs_err, axis=0)

    if len(t) < n_times:
        indices = np.arange(len(t), dtype=int)
        target_times = t
    else:
        target_times = np.linspace(t[0], t[-1], n_times)
        selected = []
        seen = set()
        for target in target_times:
            idx = int(np.argmin(np.abs(t - target)))
            if idx not in seen:
                seen.add(idx)
                selected.append((idx, target))
        if selected:
            indices, target_times = zip(*selected)
        else:
            indices, target_times = [], []

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(n_times + 1, 2, height_ratios=[1] * n_times + [0.8], hspace=0.35, wspace=0.3)

    for row, (idx, target_time) in enumerate(zip(indices, target_times)):
        ax_spec = fig.add_subplot(gs[row, 0])
        ax_err = fig.add_subplot(gs[row, 1])

        ax_spec.plot(kx, true_spec[:, idx], color='black', lw=1.8, label='True spectrum')
        ax_spec.plot(kx, pred_spec[:, idx], color='#ff7f0e', lw=1.6, linestyle='--', label='predicted spectrum')
        if abs(target_time - round(target_time)) < 1e-6:
            time_label = f'{int(round(target_time))}'
        else:
            time_label = f'{target_time:.2f}'
        ax_spec.set_title(f'Spectrum at t = {time_label}')
        ax_spec.set_xlabel('Spectral index n')
        ax_spec.set_ylabel('Amplitude')
        ax_spec.grid(True, alpha=0.2)
        if row == 0:
            ax_spec.legend(fontsize='small', loc='upper right')

        ax_err.plot(kx, abs_err[:, idx], color='crimson', lw=1.6)
        mean_abs = mean_abs_err_over_time[idx]
        ax_err.set_title(f'Absolute spectral error at t = {time_label}')
        ax_err.set_xlabel('Spectral index n')
        ax_err.set_ylabel('Absolute error')
        ax_err.grid(True, alpha=0.2)
        ax_err.text(0.05, 0.9, f'Mean abs error: {mean_abs:.2e}', transform=ax_err.transAxes,
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax_bottom = fig.add_subplot(gs[-1, :])
    ax_bottom.plot(t, mean_abs_err_over_time, color='#2ca02c', lw=2, marker='o')
    ax_bottom.set_title('Mean absolute spectral error over time')
    ax_bottom.set_xlabel('Time (t)')
    ax_bottom.set_ylabel('Mean absolute error')
    ax_bottom.grid(True, alpha=0.2)

    suptitle = title
    if res_label is not None:
        suptitle = f"{suptitle} (eval res = {res_label})"
    elif eval_resolution is not None:
        suptitle = f'{suptitle} (eval res = {int(eval_resolution)})'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.98, hspace=0.35, wspace=0.3)
    fig.savefig(outpath, dpi=150)
    print(f'spectral panel saved to {outpath}')
    plt.close(fig)


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
    plt.colorbar(im, cax=cax, label='Relative Error')
    fig.subplots_adjust(right=0.88)
    plt.savefig(outpath, dpi=150)
    print(f'error heatmap saved to {outpath}')
    plt.close()


def save_solution_gif(x, t, eta_true, eta_pred, outdir, filename, title):
    # create animation of predicted and true eta(x,t) over time
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    ax.set_title(title)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('η(x,t)')
    ax.set_xlim(x[0], x[-1])
    ymin = min(np.nanmin(eta_true), np.nanmin(eta_pred))
    ymax = max(np.nanmax(eta_true), np.nanmax(eta_pred))
    ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))
    ax.grid(True, alpha=0.3)

    true_line, = ax.plot([], [], color='black', lw=2, label='True')
    pred_line, = ax.plot([], [], color='#ff7f0e', lw=2, linestyle='--', label='predicted')
    time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.legend()

    def init():
        true_line.set_data([], [])
        pred_line.set_data([], [])
        time_text.set_text('')
        return true_line, pred_line, time_text

    def update(frame):
        true_line.set_data(x, eta_true[:, frame])
        pred_line.set_data(x, eta_pred[:, frame])
        time_text.set_text(f'Time: {t[frame]:.2f}')
        return true_line, pred_line, time_text

    # extend animation runtime by +5 seconds for evaluation
    original_duration = len(t) / 20.0
    extended_duration = original_duration + 5.0
    fps = max(1, int(round(len(t) / extended_duration)))
    interval = 1000.0 / fps

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=interval)
    ani.save(outpath, writer='pillow', fps=fps)
    print(f'animation saved to {outpath}')
    plt.close(fig)