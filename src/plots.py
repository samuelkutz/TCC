import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq


def _ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir


# Plot training loss
def plot_training_loss(train_loss_history, outdir='RESULTS', filename=None):
    _ensure_outdir(outdir)
    if filename is None:
        filename = 'training_loss.png'
    outpath = os.path.join(outdir, filename)

    plt.figure()
    plt.plot(train_loss_history, lw=1.8)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Loss')
    plt.grid(True, alpha=0.4)
    plt.savefig(outpath, dpi=150)
    print(f'Training loss plot saved to {outpath}')
    plt.close()


# Plot training statistics across seeds
def plot_training_statistics(histories, labels, outdir='RESULTS', filename='training_statistics.png', log_scale=True):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    histories_np = np.array(histories)
    mean_history = np.mean(histories_np, axis=0)
    std_history = np.std(histories_np, axis=0)
    epochs = np.arange(len(mean_history))

    plt.figure(figsize=(9, 5))
    for history, label in zip(histories, labels):
        plt.plot(epochs, history, alpha=0.35, label=label)

    plt.plot(epochs, mean_history, color='black', lw=2.5, label='Mean')
    plt.fill_between(epochs, mean_history - std_history, mean_history + std_history,
                     color='black', alpha=0.15, label='Std Dev')

    plt.title('Training Relative L2 Loss Across Runs')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Loss')
    plt.grid(True, alpha=0.3)
    if log_scale:
        plt.yscale('log')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'Training statistics plot saved to {outpath}')
    plt.close()


# Spectral relative error metrics
def compute_spectral_relative_error(eta_true, eta_pred):
    true_fft = np.fft.rfft2(eta_true)
    pred_fft = np.fft.rfft2(eta_pred)
    diff_fft = true_fft - pred_fft
    rel_error = np.linalg.norm(diff_fft) / (np.linalg.norm(true_fft) + 1e-12)
    return float(rel_error)


def compute_relative_error(eta_true, eta_pred, floor_ratio=1e-2, floor_min=1e-3):
    """Compute a robust relative error map.

    This avoids huge spikes from dividing by values near zero in the true field.
    """
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    abs_true = np.abs(eta_true)
    floor = np.maximum(np.max(abs_true) * floor_ratio, floor_min)
    denom = np.maximum(abs_true, floor)
    rel_error = np.abs(eta_true - eta_pred) / denom
    return np.nan_to_num(rel_error, posinf=1e3, neginf=0.0)


def plot_spectral_summary(eta_true, eta_pred, x, t, outdir='RESULTS', filename='spectral_summary.png', title='Spectrum Comparison'):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x)
    t = np.asarray(t)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    kx = np.fft.rfftfreq(len(x), d=dx)

    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    rel_spec = compute_relative_error(true_spec, pred_spec)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im0 = axes[0].imshow(true_spec.T,
                          extent=[kx[0], kx[-1], t[0], t[-1]],
                          origin='lower',
                          aspect='auto',
                          cmap='viridis')
    axes[0].set_title('True X-spectrum over time')
    axes[0].set_xlabel('Spatial frequency kx')
    axes[0].set_ylabel('Time')
    plt.colorbar(im0, ax=axes[0], label='Amplitude')

    im1 = axes[1].imshow(rel_spec.T,
                          extent=[kx[0], kx[-1], t[0], t[-1]],
                          origin='lower',
                          aspect='auto',
                          cmap='inferno',
                          vmin=0.0,
                          vmax=np.max(rel_spec))
    axes[1].set_title('Relative X-spectrum Error')
    axes[1].set_xlabel('Spatial frequency kx')
    plt.colorbar(im1, ax=axes[1], label='Relative Error')

    fig.suptitle(title)
    plt.savefig(outpath, dpi=150)
    print(f'Spectral summary saved to {outpath}')
    plt.close()


def plot_relative_error_panel(x, t, eta_true, eta_pred, times=None, outdir='RESULTS', filename='relative_error_summary.png', title='Relative Error Summary'):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    x = np.asarray(x)
    t = np.asarray(t)
    times = np.array(times if times is not None else [t[0], t[-1]], dtype=float)
    indices = [int(np.argmin(np.abs(t - ti))) for ti in times]

    rel_error = compute_relative_error(eta_true, eta_pred)
    time_relative_norm = np.linalg.norm(np.abs(eta_true - eta_pred), axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-8)

    dx = x[1] - x[0] if len(x) > 1 else 1.0
    kx = np.fft.rfftfreq(len(x), d=dx)
    true_spec = np.abs(np.fft.rfft(eta_true, axis=0))
    pred_spec = np.abs(np.fft.rfft(eta_pred, axis=0))
    rel_spec = np.abs(pred_spec - true_spec) / (true_spec + 1e-12)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

    for idx, time_index in enumerate(indices):
        ax = axes[0, idx]
        ax.plot(x, eta_true[:, time_index], color='black', lw=1.8, label='True')
        ax.plot(x, eta_pred[:, time_index], color='#ff7f0e', lw=1.8, linestyle='--', label='Predicted')
        ax.set_title(f'Prediction at t={t[time_index]:.2f}')
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('η(x,t)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize='small')

    im = axes[1, 0].imshow(rel_error.T,
                           extent=[x[0], x[-1], t[0], t[-1]],
                           origin='lower',
                           aspect='auto',
                           cmap='magma')
    axes[1, 0].set_title('Relative Error Heatmap')
    axes[1, 0].set_xlabel('Space (x)')
    axes[1, 0].set_ylabel('Time (t)')
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Relative Error')

    axes[1, 1].plot(t, time_relative_norm, color='crimson', lw=2)
    axes[1, 1].set_title('Time-resolved Relative Error Norm')
    axes[1, 1].set_xlabel('Time (t)')
    axes[1, 1].set_ylabel('Relative Error')
    axes[1, 1].grid(True, alpha=0.3)
    mean_err = np.mean(time_relative_norm)
    axes[1, 1].text(0.5, 0.9, f'Mean: {mean_err:.2e}', transform=axes[1, 1].transAxes,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

    im2 = axes[2, 0].imshow(true_spec.T,
                            extent=[kx[0], kx[-1], t[0], t[-1]],
                            origin='lower',
                            aspect='auto',
                            cmap='viridis')
    axes[2, 0].set_title('True X-spectrum over time')
    axes[2, 0].set_xlabel('Spatial frequency kx')
    axes[2, 0].set_ylabel('Time')
    divider2 = make_axes_locatable(axes[2, 0])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Amplitude')

    im3 = axes[2, 1].imshow(rel_spec.T,
                            extent=[kx[0], kx[-1], t[0], t[-1]],
                            origin='lower',
                            aspect='auto',
                            cmap='inferno',
                            vmin=0.0,
                            vmax=np.max(rel_spec))
    axes[2, 1].set_title('Relative X-spectrum Error')
    axes[2, 1].set_xlabel('Spatial frequency kx')
    divider3 = make_axes_locatable(axes[2, 1])
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im3, cax=cax3, label='Relative Error')

    fig.suptitle(title)
    fig.savefig(outpath, dpi=150)
    print(f'Relative error summary saved to {outpath}')
    plt.close()


# Plot solution snapshots for several instants
def plot_solution_snapshots(x, t, eta_true, eta_pred, times, outdir='RESULTS', filename='solution_snapshots.png', title='Solution Snapshots'):
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
        axes[row, 1].set_title(f'Relative Error at t={t[idx]:.2f}')
        axes[row, 1].set_ylabel('Relative Error')
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Space (x)')
    axes[-1, 1].set_xlabel('Space (x)')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath, dpi=150)
    print(f'Solution snapshots saved to {outpath}')
    plt.close()


def plot_error_heatmap(x, t, eta_true, eta_pred, outdir='RESULTS', filename='error_heatmap.png', title='Relative Error Heatmap'):
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    rel_diff = compute_relative_error(eta_true, eta_pred)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(rel_diff.T, extent=[x[0], x[-1], t[0], t[-1]], origin='lower', aspect='auto', cmap='inferno')
    ax.set_title(title)
    ax.set_xlabel('Space (x)')
    ax.set_ylabel('Time (t)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Relative Error')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'Error heatmap saved to {outpath}')
    plt.close()


def save_solution_gif(x, t, eta_true, eta_pred, outdir='RESULTS', filename='solution_animation.gif', title='Solution Animation'):
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
    pred_line, = ax.plot([], [], color='#ff7f0e', lw=2, linestyle='--', label='Predicted')
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

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=80)
    ani.save(outpath, writer='pillow', fps=20)
    print(f'Animation saved to {outpath}')
    plt.close(fig)


# Keep compatibility with the previous helper
def generate_animation(model, val_a=1, val_b=1, res_high=64, device='cpu', outdir='RESULTS'):
    _ensure_outdir(outdir)

    bsq_high = Boussinesq(x_min=-30, x_max=30, t_min=0, t_max=15, a=val_a, b=val_b)
    solver_high = PseudoSpectralBoussinesq(bsq_high, Nx=res_high, Nt=res_high-1, device=device)
    x_high, t_high, eta_true, u_true = solver_high.solve()

    ch0 = np.tile(eta_true[0:1, :].T, (1, res_high))
    ch1 = np.tile(u_true[0:1, :].T, (1, res_high))
    ch2 = np.ones((res_high, res_high)) * val_a
    ch3 = np.ones((res_high, res_high)) * val_b

    input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)
    input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_tensor = model(input_tensor)
        eta_pred = pred_tensor.squeeze().cpu().numpy()[0, :, :]

    eta_true = eta_true.T
    title = f'Boussinesq FNO (a={val_a:.2f}, b={val_b:.2f})'
    save_solution_gif(x_high, t_high, eta_true, eta_pred, outdir=outdir,
                      filename=f'boussinesq_a={val_a:.2f}_b={val_b:.2f}_res{res_high}.gif',
                      title=title)


# Evaluate errors for test parameters
def evaluate_errors(model, test_params, res_high=64, device='cpu', outdir='RESULTS', filename=None):
    _ensure_outdir(outdir)
    if filename is None:
        filename = 'error_analysis.png'
    outpath = os.path.join(outdir, filename)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(r'Error Analysis FNO: Boussinesq ($\eta$)', fontsize=18, y=0.95)

    print(f'Evaluating {len(test_params)} cases at resolution {res_high}x{res_high}...')

    for idx, val in enumerate(test_params):
        bsq_sim = Boussinesq(x_min=-30, x_max=30, t_min=0, t_max=15, a=val, b=val)
        solver = PseudoSpectralBoussinesq(bsq_sim, Nx=res_high, Nt=res_high-1, device=device)
        x_sol, t_sol, eta_true, u_true = solver.solve()

        eta_0 = eta_true[0, :]
        u_0 = u_true[0, :]

        ch0 = np.tile(eta_0[:, None], (1, res_high))
        ch1 = np.tile(u_0[:, None], (1, res_high))
        ch2 = np.ones((res_high, res_high), dtype=np.float32) * val
        ch3 = np.ones((res_high, res_high), dtype=np.float32) * val

        input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)
        input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            pred_tensor = model(input_tensor)
            eta_pred = pred_tensor.squeeze().cpu().numpy()[0, :, :]

        eta_true_t = eta_true.T

        diff = eta_true_t - eta_pred
        norm_diff = np.linalg.norm(diff, ord=2, axis=0)
        norm_true = np.linalg.norm(eta_true_t, ord=2, axis=0)
        relative_error_t = norm_diff / (norm_true + 1e-8)

        ax_map = axes[idx, 0]
        im = ax_map.imshow(np.abs(diff).T,
                           extent=[x_sol[0], x_sol[-1], t_sol[0], t_sol[-1]],
                           origin='lower', aspect='auto', cmap='inferno')
        ax_map.set_title(rf'Absolute Error $|\eta - \tilde{{\eta}}|$ ($\alpha=\beta={val}$)')
        ax_map.set_ylabel('Time (t)')
        if idx == 2:
            ax_map.set_xlabel('Space (x)')

        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax_err = axes[idx, 1]
        ax_err.plot(t_sol, relative_error_t, color='crimson', lw=2)
        ax_err.set_title(r'Relative Error: $\frac{||\eta(\cdot,t) - \tilde{\eta}(\cdot,t)||}{||\eta(\cdot,t)||}$')
        ax_err.grid(True, linestyle='--', alpha=0.6)
        ax_err.set_xlim(t_sol[0], t_sol[-1])
        ax_err.set_ylim(bottom=0)

        mean_err = np.mean(relative_error_t)
        ax_err.text(0.5, 0.9, f'Mean: {mean_err:.2e}', transform=ax_err.transAxes,
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

        if idx == 2:
            ax_err.set_xlabel('Time (t)')

        print(f'Case alpha={val}: Mean Relative Error = {mean_err:.4e}')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'Error analysis plot saved to {outpath}')
    plt.close()
