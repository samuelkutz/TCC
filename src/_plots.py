import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LightSource
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
    ax.set_title('Training $L^2$ Loss History', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('$L^2$ Loss', fontsize=12)
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


def plot_soliton_profile(outdir, filename='soliton_profile.png', A=1.0):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.linspace(-6, 6, 600)
    y = A / np.cosh(x) ** 2

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color='#ff7f0e', lw=2.2)
    ax.set_title(r'Initial soliton profile $\eta(x,0) = A\,\mathrm{sech}^2(x)$, $A=1$', fontsize=14)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$\eta(x,0)$', fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f'soliton profile saved to {outpath}')
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

    ax.set_title('Training $L^2$ Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('$L^2$ Loss', fontsize=12)
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
    axes[0].set_title('True Spatial Spectrum Over Time')
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
    axes[1].set_title('Relative Spatial Spectrum Error')
    axes[1].set_xlabel('Spectral index n')
    plt.colorbar(im1, ax=axes[1], label='Relative Error')

    fig.suptitle(title)
    plt.savefig(outpath, dpi=150)
    print(f'spectral summary saved to {outpath}')
    plt.close()


def plot_relative_error_panel(x, t, eta_true, eta_pred, times, outdir, filename, title):
    dir_text = os.path.join(outdir, 'text')
    dir_beamer = os.path.join(outdir, 'beamer')
    _ensure_outdir(dir_text)
    _ensure_outdir(dir_beamer)
    
    outpath_text = os.path.join(dir_text, filename)
    basename, ext = os.path.splitext(filename)

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

    fig = plt.figure(figsize=(14, 14), constrained_layout=False)
    # The original creates a 3x2 grid. We replicate using GridSpec for manual assignment
    gs = fig.add_gridspec(3, 2)
    axes = np.empty((3, 2), dtype=object)
    
    # row 0
    fig_b_row0, axes_b_0 = plt.subplots(1, 2, figsize=(14, 4))
    for idx, time_index in enumerate(indices):
        if idx >= 2: break
        
        ax = fig.add_subplot(gs[0, idx])
        axes[0, idx] = ax
        ax_b = axes_b_0[idx]
        
        for a in (ax, ax_b):
            a.plot(x, eta_true[:, time_index], color='black', lw=1.8, label='True')
            a.plot(x, eta_pred[:, time_index], color='#ff7f0e', lw=1.8, linestyle='--', label='predicted')
            a.set_title(f'Prediction at $t = {t[time_index]:.2f}$')
            a.set_xlabel('Space (x)')
            a.set_ylabel('η(x,t)')
            a.grid(True, alpha=0.3)
            if idx == 0:
                a.legend(fontsize='small', loc='best')
    fig_b_row0.tight_layout()
    fig_b_row0.savefig(os.path.join(dir_beamer, f'{basename}_row_0{ext}'), dpi=150)
    plt.close(fig_b_row0)

    # row 1
    fig_b_row1, (ax_b_10, ax_b_11) = plt.subplots(1, 2, figsize=(14, 4))
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1])

    for ax, ax_b in [(axes[1, 0], ax_b_10)]:
        im = ax.imshow(
            rel_error.T, extent=[x[0], x[-1], t[0], t[-1]],
            origin='lower', aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_title('Relative Error Heatmap')
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label='Relative Error')
        
        im_b = ax_b.imshow(
            rel_error.T, extent=[x[0], x[-1], t[0], t[-1]],
            origin='lower', aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
        ax_b.set_title('Relative Error Heatmap')
        ax_b.set_xlabel('Space (x)')
        ax_b.set_ylabel('Time (t)')
        divider_b = make_axes_locatable(ax_b)
        cax_b = divider_b.append_axes('right', size='5%', pad=0.05)
        fig_b_row1.colorbar(im_b, cax=cax_b, label='Relative Error')

    for ax, ax_b in [(axes[1, 1], ax_b_11)]:
        time_plot = t + 5.0
        ax.plot(time_plot, time_relative_norm, color='crimson', lw=2)
        ax.axvline(x=time_plot[-1], color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='evaluation stop')
        ax.set_title('Time-Resolved Relative Error Norm')
        ax.set_xlabel('Time (t + 5)')
        ax.set_ylabel('Relative error')
        ax.grid(True, alpha=0.3)
        mean_err = np.mean(time_relative_norm)
        ax.text(0.5, 0.9, f'Mean: {mean_err:.2e}', transform=ax.transAxes, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        
        ax_b.plot(time_plot, time_relative_norm, color='crimson', lw=2)
        ax_b.axvline(x=time_plot[-1], color='black', linestyle='--', linewidth=1.0, alpha=0.7, label='evaluation stop')
        ax_b.set_title('Time-Resolved Relative Error Norm')
        ax_b.set_xlabel('Time (t + 5)')
        ax_b.set_ylabel('Relative error')
        ax_b.grid(True, alpha=0.3)
        ax_b.text(0.5, 0.9, f'Mean: {mean_err:.2e}', transform=ax_b.transAxes, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

    fig_b_row1.tight_layout()
    fig_b_row1.savefig(os.path.join(dir_beamer, f'{basename}_row_1{ext}'), dpi=150)
    plt.close(fig_b_row1)

    # row 2
    fig_b_row2, (ax_b_20, ax_b_21) = plt.subplots(1, 2, figsize=(14, 4))
    axes[2, 0] = fig.add_subplot(gs[2, 0])
    axes[2, 1] = fig.add_subplot(gs[2, 1])

    for ax, ax_b in [(axes[2, 0], ax_b_20)]:
        im2 = ax.imshow(
            true_spec.T, extent=[kx[0], kx[-1], t[0], t[-1]],
            origin='lower', aspect='auto', cmap='viridis')
        ax.set_title('True Spatial Spectrum Over Time')
        ax.set_xlabel('Spectral index n')
        ax.set_ylabel('Time')
        divider2 = make_axes_locatable(ax)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax2, label='Amplitude')
        
        im2_b = ax_b.imshow(
            true_spec.T, extent=[kx[0], kx[-1], t[0], t[-1]],
            origin='lower', aspect='auto', cmap='viridis')
        ax_b.set_title('True Spatial Spectrum Over Time')
        ax_b.set_xlabel('Spectral index n')
        ax_b.set_ylabel('Time')
        divider2_b = make_axes_locatable(ax_b)
        cax2_b = divider2_b.append_axes('right', size='5%', pad=0.05)
        fig_b_row2.colorbar(im2_b, cax=cax2_b, label='Amplitude')

    for ax, ax_b in [(axes[2, 1], ax_b_21)]:
        im3 = ax.imshow(
            rel_spec.T, extent=[kx[0], kx[-1], t[0], t[-1]],
            origin='lower', aspect='auto', cmap='inferno', vmin=0.0, vmax=1.0)
        ax.set_title('Relative Spatial Spectrum Error')
        ax.set_xlabel('Spectral index n')
        divider3 = make_axes_locatable(ax)
        cax3 = divider3.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax3, label='Relative Error')
        
        im3_b = ax_b.imshow(
            rel_spec.T, extent=[kx[0], kx[-1], t[0], t[-1]],
            origin='lower', aspect='auto', cmap='inferno', vmin=0.0, vmax=1.0)
        ax_b.set_title('Relative Spatial Spectrum Error')
        ax_b.set_xlabel('Spectral index n')
        divider3_b = make_axes_locatable(ax_b)
        cax3_b = divider3_b.append_axes('right', size='5%', pad=0.05)
        fig_b_row2.colorbar(im3_b, cax=cax3_b, label='Relative Error')

    fig_b_row2.tight_layout()
    fig_b_row2.savefig(os.path.join(dir_beamer, f'{basename}_row_2{ext}'), dpi=150)
    plt.close(fig_b_row2)

    fig.suptitle(title)
    fig.subplots_adjust(top=0.95, bottom=0.03, hspace=0.35, wspace=0.28)
    fig.savefig(outpath_text, dpi=150)
    print(f'relative error summary text saved to {outpath_text} and beamer rows to {dir_beamer}')
    plt.close(fig)


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
        axes[row, 1].set_title(f'Relative Error at $t = {t[idx]:.2f}$')
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
    dir_text = os.path.join(outdir, 'text')
    dir_beamer = os.path.join(outdir, 'beamer')
    _ensure_outdir(dir_text)
    _ensure_outdir(dir_beamer)
    
    outpath_text = os.path.join(dir_text, filename)
    basename, ext = os.path.splitext(filename)

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

        # create a beamer figure
        fig_b, (ax_left_b, ax_right_b) = plt.subplots(1, 2, figsize=(14, 4))
        
        y_ticks = [j * offset for j in range(len(indices))]
        y_tick_labels = [f'{t[idx]:.2f}' if idx < len(t) else '' for idx in indices]
        
        if res_label is not None:
            res_text = str(res_label)
        elif eval_resolution is not None:
            res_text = f'{int(eval_resolution)}'
        else:
            res_text = None

        for ax_l in (ax_left, ax_left_b):
            for j, idx in enumerate(indices):
                baseline = j * offset
                ax_l.plot(x, eta_true[:, idx] + baseline, color='#2b1b17', lw=1.6,
                             label='Reference' if j == 0 else '')
                ax_l.plot(x, eta_pred[:, idx] + baseline, color='#d94801', lw=1.4, linestyle='--',
                             label='Predicted' if j == 0 else '')
                ax_l.hlines(baseline, x[0], x[-1], color='#8c564b', alpha=0.25, linewidth=0.7)
            
            ax_l.set_yticks(y_ticks)
            ax_l.set_yticklabels(y_tick_labels)
            ax_l.set_title(f'Stacked Solutions for $\\alpha = \\beta = {alpha:.3f}$')
            ax_l.set_xlabel('Space (x)')
            ax_l.set_ylabel('Time (t)')
            ax_l.grid(True, alpha=0.2)
            
            if res_text is not None:
                ax_l.text(
                    0.98,
                    0.02,
                    f'eval res = {res_text}',
                    transform=ax_l.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                )
            if i == 0 or ax_l == ax_left_b:
                ax_l.legend(loc='upper left', fontsize='small')

        for ax_r in (ax_right, ax_right_b):
            ax_r.plot(t, time_rel_norms[i], lw=1.8, color='#b22222')
            ax_r.set_title(f'Relative Error Over Time for $\\alpha = \\beta = {alpha:.3f}$')
            ax_r.set_xlabel('Time (t)')
            ax_r.set_ylabel('Relative error')
            ax_r.grid(True, alpha=0.2)

        fig_b.tight_layout()
        fig_b.savefig(os.path.join(dir_beamer, f'{basename}_row_{i}_a{alpha:.3f}{ext}'), dpi=150)
        plt.close(fig_b)

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
    ax_bottom.set_title('Relative Error Distribution vs. $\\alpha = \\beta$')
    ax_bottom.set_xlabel('$\\alpha = \\beta$ Value')
    ax_bottom.set_ylabel('Relative Error')
    ax_bottom.grid(True, alpha=0.2)
    ax_bottom.set_xticks(param_values)
    ax_bottom.set_xticklabels([f'{val:.2f}' for val in param_values], rotation=15, ha='right')
    
    # Save the boxplot part for beamer as well
    fig_b_bottom, ax_b_bottom = plt.subplots(figsize=(10, 4))
    box_b = ax_b_bottom.boxplot(
        time_rel_norms,
        positions=param_values,
        widths=np.maximum(0.05 * (param_values.max() - param_values.min()), 0.05),
        patch_artist=True,
        showmeans=True,
        meanline=True,
        labels=[f'{val:.2f}' for val in param_values],
    )
    for patch in box_b['boxes']:
        patch.set_facecolor('#fdd0a2')
        patch.set_edgecolor('#e6550d')
        patch.set_alpha(0.7)
    ax_b_bottom.set_title('Relative Error Distribution vs. $\\alpha = \\beta$')
    ax_b_bottom.set_xlabel('$\\alpha = \\beta$ Value')
    ax_b_bottom.set_ylabel('Relative Error')
    ax_b_bottom.grid(True, alpha=0.2)
    ax_b_bottom.set_xticks(param_values)
    ax_b_bottom.set_xticklabels([f'{val:.2f}' for val in param_values], rotation=15, ha='right')
    fig_b_bottom.tight_layout()
    fig_b_bottom.savefig(os.path.join(dir_beamer, f'{basename}_boxplot{ext}'), dpi=150)
    plt.close(fig_b_bottom)

    suptitle = title
    if res_label is not None:
        suptitle = f"{suptitle} (eval res = {res_label})"
    elif eval_resolution is not None:
        suptitle = f'{suptitle} (eval res = {int(eval_resolution)})'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.98, hspace=0.35, wspace=0.3)
    fig.savefig(outpath_text, dpi=150)
    print(f'alpha/beta panel text saved to {outpath_text} and beamer rows to {dir_beamer}')
    plt.close(fig)


def plot_model2_resolution_panel(x_list, t_list, eta_true_list, eta_pred_list, resolutions, outdir, filename, title,
                                 n_curves=5, offset_factor=1.3, eval_alpha_beta=None, param_label=None):
    dir_text = os.path.join(outdir, 'text')
    dir_beamer = os.path.join(outdir, 'beamer')
    _ensure_outdir(dir_text)
    _ensure_outdir(dir_beamer)
    
    outpath_text = os.path.join(dir_text, filename)
    basename, ext = os.path.splitext(filename)

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

        # create a beamer figure
        fig_b, (ax_left_b, ax_right_b) = plt.subplots(1, 2, figsize=(14, 4))

        indices, _ = _five_time_slice_indices_labels(t_res)
        y_ticks = [j * offset for j in range(len(indices))]
        y_tick_labels = [f'{t_res[idx]:.2f}' if idx < len(t_res) else '' for idx in indices]

        if param_label is not None:
            p_text = str(param_label)
        elif eval_alpha_beta is not None:
            p_text = f'{eval_alpha_beta:.3f}'
        else:
            p_text = None

        for ax_l in (ax_left, ax_left_b):
            for j, idx in enumerate(indices):
                if idx >= eta_true.shape[1]:
                    continue
                baseline = j * offset
                ax_l.plot(x_res, eta_true[:, idx] + baseline, color='#2b1b17', lw=1.6,
                             label='Reference' if j == 0 else '')
                ax_l.plot(x_res, eta_pred[:, idx] + baseline, color='#d94801', lw=1.4, linestyle='--',
                             label='Predicted' if j == 0 else '')
                ax_l.hlines(baseline, x_res[0], x_res[-1], color='#8c564b', alpha=0.25, linewidth=0.7)
            ax_l.set_yticks(y_ticks)
            ax_l.set_yticklabels(y_tick_labels)
            ax_l.set_title(f'Stacked solutions for res = {int(res)}')
            ax_l.set_xlabel('Space (x)')
            ax_l.set_ylabel('Time (t)')
            ax_l.grid(True, alpha=0.2)
            if p_text is not None:
                ax_l.text(
                    0.98,
                    0.02,
                    f'eval α=β = {p_text}',
                    transform=ax_l.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                )
            if i == 0 or ax_l == ax_left_b:
                ax_l.legend(loc='upper left', fontsize='small')

        for ax_r in (ax_right, ax_right_b):
            ax_r.plot(t_res, time_rel_norms[i], lw=1.8, color='#b22222')
            ax_r.set_title(f'Relative error over time for res = {int(res)}')
            ax_r.set_xlabel('Time (t)')
            ax_r.set_ylabel('Relative error')
            ax_r.grid(True, alpha=0.2)

        fig_b.tight_layout()
        fig_b.savefig(os.path.join(dir_beamer, f'{basename}_row_{i}_res{int(res)}{ext}'), dpi=150)
        plt.close(fig_b)

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

    fig_b_bottom, ax_b_bottom = plt.subplots(figsize=(10, 4))
    box_b = ax_b_bottom.boxplot(
        time_rel_norms,
        positions=resolutions,
        widths=np.maximum(0.05 * (resolutions.max() - resolutions.min()), 1.0),
        patch_artist=True,
        showmeans=True,
        meanline=True,
        labels=[str(int(res)) for res in resolutions],
    )
    for patch in box_b['boxes']:
        patch.set_facecolor('#fdae6b')
        patch.set_edgecolor('#e6550d')
        patch.set_alpha(0.7)
    ax_b_bottom.set_title('Relative error distribution vs. resolution')
    ax_b_bottom.set_xlabel('Resolution')
    ax_b_bottom.set_ylabel('Relative error')
    ax_b_bottom.grid(True, alpha=0.2)
    ax_b_bottom.set_xticks(resolutions)
    ax_b_bottom.set_xticklabels([str(int(res)) for res in resolutions], rotation=15, ha='right')
    fig_b_bottom.tight_layout()
    fig_b_bottom.savefig(os.path.join(dir_beamer, f'{basename}_boxplot{ext}'), dpi=150)
    plt.close(fig_b_bottom)

    suptitle = title
    if param_label is not None:
        suptitle = f"{suptitle} (eval α=β = {param_label})"
    elif eval_alpha_beta is not None:
        suptitle = f'{suptitle} (eval α=β = {eval_alpha_beta:.3f})'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.98, hspace=0.35, wspace=0.3)
    fig.savefig(outpath_text, dpi=150)
    print(f'resolution panel text saved to {outpath_text} and beamer rows to {dir_beamer}')
    plt.close(fig)


def plot_model2_spectral_panel(x, t, eta_true, eta_pred, outdir, filename, title,
                               n_times=3, eval_resolution=None, res_label=None, param_label=None):
    dir_text = os.path.join(outdir, 'text')
    dir_beamer = os.path.join(outdir, 'beamer')
    _ensure_outdir(dir_text)
    _ensure_outdir(dir_beamer)
    
    outpath_text = os.path.join(dir_text, filename)
    basename, ext = os.path.splitext(filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    kx = spectral_mode_index(x)

    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    rel_err = compute_relative_error(true_spec, pred_spec)
    mean_rel_err_over_time = np.mean(rel_err, axis=0)

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

        # create a beamer figure
        fig_b, (ax_spec_b, ax_err_b) = plt.subplots(1, 2, figsize=(14, 4))
        
        if abs(target_time - round(target_time)) < 1e-6:
            time_label = f'{int(round(target_time))}'
        else:
            time_label = f'{target_time:.2f}'

        for ax_s in (ax_spec, ax_spec_b):
            ax_s.plot(kx, true_spec[:, idx], color='black', lw=1.8, label='True spectrum')
            ax_s.plot(kx, pred_spec[:, idx], color='#ff7f0e', lw=1.6, linestyle='--', label='predicted spectrum')
            ax_s.set_title(f'Spectrum at t = {time_label}')
            ax_s.set_xlabel('Spectral index n')
            ax_s.set_ylabel('Amplitude')
            ax_s.grid(True, alpha=0.2)
            if row == 0 or ax_s == ax_spec_b:
                ax_s.legend(fontsize='small', loc='upper right')

        for ax_e in (ax_err, ax_err_b):
            ax_e.plot(kx, rel_err[:, idx], color='crimson', lw=1.6)
            mean_rel = mean_rel_err_over_time[idx]
            ax_e.set_title(f'Relative Spectral Error at $t = {time_label}$')
            ax_e.set_xlabel('Spectral index n')
            ax_e.set_ylabel('Relative error')
            ax_e.grid(True, alpha=0.2)
            ax_e.text(0.05, 0.9, f'Mean rel error: {mean_rel:.2e}', transform=ax_e.transAxes,
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        fig_b.tight_layout()
        fig_b.savefig(os.path.join(dir_beamer, f'{basename}_row_{row}_t{time_label}{ext}'), dpi=150)
        plt.close(fig_b)

    ax_bottom = fig.add_subplot(gs[-1, :])
    ax_bottom.plot(t, mean_rel_err_over_time, color='#c28b00', lw=2, marker='o')
    ax_bottom.set_title('Mean Relative Spectral Error Over Time')
    ax_bottom.set_xlabel('Time (t)')
    ax_bottom.set_ylabel('Mean Relative Error')
    ax_bottom.grid(True, alpha=0.2)
    
    fig_b_bottom, ax_b_bottom = plt.subplots(figsize=(10, 4))
    ax_b_bottom.plot(t, mean_rel_err_over_time, color='#c28b00', lw=2, marker='o')
    ax_b_bottom.set_title('Mean Relative Spectral Error Over Time')
    ax_b_bottom.set_xlabel('Time (t)')
    ax_b_bottom.set_ylabel('Mean Relative Error')
    ax_b_bottom.grid(True, alpha=0.2)
    fig_b_bottom.tight_layout()
    fig_b_bottom.savefig(os.path.join(dir_beamer, f'{basename}_bottom{ext}'), dpi=150)
    plt.close(fig_b_bottom)

    suptitle = title
    if param_label is not None:
        suptitle = f'{suptitle} (alpha=beta={param_label})'
    if res_label is not None:
        suptitle = f"{suptitle} (eval res = {res_label})"
    elif eval_resolution is not None:
        suptitle = f'{suptitle} (eval res = {int(eval_resolution)})'
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.05, right=0.98, hspace=0.35, wspace=0.3)
    fig.savefig(outpath_text, dpi=150)
    print(f'spectral panel text saved to {outpath_text} and beamer rows to {dir_beamer}')
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


def plot_spectral_bias_evolution(ordered_metadata, outdir, filename='spectral_bias_evolution.png'):
    models_data = []
    for label, mf in ordered_metadata:
        if not os.path.exists(mf):
            print(f'  metadata not found: {mf}, skipping')
            continue
        md = torch.load(mf, map_location='cpu')
        sh = md.get('spectral_history')
        if sh is None or not sh.get('epochs'):
            print(f'  no spectral_history in {mf}, skipping (retrain to populate)')
            continue
        models_data.append((label, sh))
    if not models_data:
        print('  no spectral history data available — retrain models first')
        return
    plot_spectral_bias_panel(models_data, outdir=outdir, filename=filename)


def plot_spectral_bias_panel(models_data, outdir, filename='spectral_bias_evolution.png'):
    """Plot per-frequency-band spectral error vs training epoch for each model.

    models_data: list of (label, spectral_history) tuples, where spectral_history is a dict
                 with keys 'epochs', 'low_band', 'mid_band', 'high_band'.
    Expected order: [pinn_no_data, pinn_data, pino_no_data, pino_data, fno].
    The last model is rendered spanning the full width of the bottom row.
    """
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    band_colors = ['#1f77b4', '#9467bd', '#d62728']   # blue, purple, red
    band_labels = [
        r'Low freq  ($k < N_x/4$)',
        r'Mid freq  ($N_x/4 \leq k < N_x/2$)',
        r'High freq ($k \geq N_x/2$)',
    ]
    band_keys = ['low_band', 'mid_band', 'high_band']

    n = len(models_data)
    # first n-1 models in 2-column pairs; last model spans full width
    n_pair_rows = (n - 1 + 1) // 2  # ceiling of (n-1)/2
    n_rows = n_pair_rows + 1         # +1 for the spanning bottom row

    fig = plt.figure(figsize=(14, 4.5 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.50, wspace=0.32)

    axes = []
    for i in range(n - 1):
        row = i // 2
        col = i % 2
        axes.append(fig.add_subplot(gs[row, col]))
    axes.append(fig.add_subplot(gs[-1, :]))  # last model spans both columns

    for i, (ax, (label, sh)) in enumerate(zip(axes, models_data)):
        epochs = np.asarray(sh['epochs'])
        for color, blab, key in zip(band_colors, band_labels, band_keys):
            vals = np.asarray(sh[key])
            ax.semilogy(epochs, vals, color=color, lw=2.0, label=blab, alpha=0.9)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Rel. Spectral Error', fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)
        if i == 0:
            ax.legend(fontsize='small', loc='upper right')

    fig.suptitle('Spectral Error Evolution by Frequency Band During Training', fontsize=14, y=1.01)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'spectral bias panel saved to {outpath}')


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


def plot_solution_surface_3d(x, t, eta_true, eta_pred, outdir, filename, title,
                              true_label='Reference', pred_label='Prediction', cmap='inferno'):
    # create side-by-side 3d surfaces for reference and prediction using Plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    # use shared z-limits for fair visual comparison
    z_min = min(np.nanmin(eta_true), np.nanmin(eta_pred))
    z_max = max(np.nanmax(eta_true), np.nanmax(eta_pred))
    z_margin = (z_max - z_min) * 0.05
    z_lim = [z_min - z_margin, z_max + z_margin]

    # Create subplots for 3D surfaces
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=(true_label, pred_label),
        horizontal_spacing=0.05
    )

    # True Surface
    fig.add_trace(
        go.Surface(
            x=t, y=x, z=eta_true,
            colorscale=cmap,
            cmin=z_min, cmax=z_max,
            colorbar=dict(title='η(x,t)', x=1.05, len=0.75),
            showscale=True
        ),
        row=1, col=1
    )

    # Predicted Surface
    fig.add_trace(
        go.Surface(
            x=t, y=x, z=eta_pred,
            colorscale=cmap,
            cmin=z_min, cmax=z_max,
            showscale=False
        ),
        row=1, col=2
    )

    # Camera view settings (similar to previous matplotlib angles)
    # x/y/z correspond to eye coordinates. 
    camera = dict(
        eye=dict(x=-1.5, y=-1.8, z=0.8),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )

    # Shared axis/scene configurations
    scene_config = dict(
        xaxis_title='Time (t)',
        yaxis_title='Space (x)',
        zaxis_title='η',
        zaxis=dict(range=z_lim),
        aspectmode='manual',
        # Set realistic proportional bounding box scale
        aspectratio=dict(x=1.5, y=1, z=0.4) 
    )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        scene=scene_config,
        scene2=scene_config,
        margin=dict(l=0, r=0, b=0, t=60), # Remove excessive white space
        width=1200,
        height=600
    )
    
    # Apply camera to both scenes
    fig.layout.scene.camera = camera
    fig.layout.scene2.camera = camera

    # Save to file (requires 'kaleido')
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'Plotly 3d surface plot saved to {outpath}')
    except Exception as e:
        print("Erro ao salvar Plotly em PNG. Certifique-se de ter rodado: pip install kaleido")
        raise e


def plot_radially_revolved_gif(x, t, eta_true, eta_pred, outdir, filename, title,
                               true_label='Reference', pred_label='Prediction', cmap='inferno'):
    # create a plotly 3D GIF animating the radial revolution of eta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from PIL import Image
    import io
    
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)
    if not outpath.endswith('.gif'):
        outpath = outpath.rsplit('.', 1)[0] + '.gif'

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    # 1. Create a 2D radial mesh (Full circle)
    x_center = (np.max(x) + np.min(x)) / 2.0
    r_max = np.max(np.abs(x - x_center))
    
    grid_pts = 160 # High resolution for "marolinhas"
    linspace = np.linspace(-r_max, r_max, grid_pts)
    X, Y = np.meshgrid(linspace, linspace)
    R = np.sqrt(X**2 + Y**2)
    R_clipped = np.clip(R, 0, np.max(np.abs(x - x_center)))

    # Since we are not plotting true solution, scale bounds based on prediction
    z_min = np.nanmin(eta_pred)
    z_max = np.nanmax(eta_pred)
    z_margin = (z_max - z_min) * 0.05
    z_lim = [z_min - z_margin, z_max + z_margin]

    def get_radially_revolved(eta_1d):
        idx_right = x >= x_center
        x_right = x[idx_right] - x_center
        eta_right = eta_1d[idx_right]
        Z = np.interp(R_clipped.ravel(), x_right, eta_right)
        
        # Zero out values outside the physical circle to make it look clean
        Z = Z.reshape(R.shape)
        Z[R > r_max] = np.nan
        return Z

    camera = dict(
        eye=dict(x=-1.2, y=-1.6, z=0.8), # Pulled back slightly for the wider frame
        center=dict(x=0, y=0, z=-0.1),
        up=dict(x=0, y=0, z=1)
    )
    scene_config = dict(
        xaxis_title='Space X',
        yaxis_title='Space Y',
        zaxis_title='η(x, y)',
        zaxis=dict(range=z_lim),
        xaxis=dict(range=[-r_max, r_max]),
        yaxis=dict(range=[-r_max, r_max]),
        aspectmode='manual',
        aspectratio=dict(x=1.8, y=1.8, z=0.3) # Increased domain X/Y and reduced Z to avoid pointiness
    )

    max_frames = 48
    step = max(1, len(t) // max_frames)
    frame_indices = list(range(0, len(t), step))
    # ensure last frame is included
    if frame_indices[-1] != len(t) - 1:
        frame_indices.append(len(t) - 1)

    print(f"Generating radial 3D GIF ({len(frame_indices)} frames)...")
    frames = []

    for idx in frame_indices:
        Z_pred = get_radially_revolved(eta_pred[:, idx])

        fig = go.Figure()

        # Added lighting properties to create shadows/specular reflections and 'Viridis' color
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_pred, 
                colorscale='Viridis', 
                cmin=z_min, cmax=z_max, showscale=False,
                lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.6, fresnel=0.2)
            )
        )

        fig.update_layout(
            title_text=f"{title} (t = {t[idx]:.2f})",
            title_x=0.5,
            scene=scene_config,
            margin=dict(l=0, r=0, b=0, t=60),
            width=1200,
            height=850,
            paper_bgcolor='white'
        )
        fig.layout.scene.camera = camera

        try:
            img_bytes = fig.to_image(format="png", width=1200, height=850, scale=1.5)
            image = Image.open(io.BytesIO(img_bytes))
            frames.append(image)
        except Exception as e:
            print("Erro ao gerar frame para o GIF. Verifique o pacote kaleido.")
            return

    if frames:
        # approx 5 seconds total duration
        duration_per_frame = max(50, 5000 // len(frames))
        frames[0].save(
            outpath,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration_per_frame
        )
        print(f"Radial evolution 3D GIF saved to: {outpath}")