import os

import numpy as np
import torch
import matplotlib.pyplot as plt
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


def plot_training_statistics(histories, labels, outdir, filename, log_scale=True, duration_seconds=None, final_loss=None, num_params=None):
    import plotly.graph_objects as go
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig = go.Figure()
    for i, (history, label) in enumerate(zip(histories, labels)):
        epochs = list(range(1, len(history) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=list(history),
            mode='lines', name=label,
            line=dict(color=_colors[i % len(_colors)], width=2.2),
            opacity=0.9,
        ))

    annotation_lines = []
    if num_params is not None:
        annotation_lines.append(f'Params: {num_params:,}')
    if duration_seconds is not None:
        annotation_lines.append(f'Training time: {duration_seconds:.1f}s')
    if final_loss is not None:
        annotation_lines.append(f'Final loss: {final_loss:.2e}')

    fig.update_layout(
        title_text='Training L2 Loss', title_x=0.5,
        xaxis_title='Epoch',
        yaxis_title='L2 Loss',
        yaxis_type='log' if log_scale else 'linear',
        template='plotly_white',
        width=1000, height=500,
        showlegend=len(labels) > 1,
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
    )
    if annotation_lines:
        fig.add_annotation(
            text='<br>'.join(annotation_lines),
            xref='paper', yref='paper',
            x=0.02, y=0.98, xanchor='left', yanchor='top',
            showarrow=False,
            bordercolor='#aaaaaa', borderwidth=1, borderpad=8,
            bgcolor='white', opacity=0.92,
            font=dict(size=11),
        )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'training statistics plot saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar training statistics como PNG. Instale kaleido: pip install kaleido')
        raise e


# spectral relative error metrics
def compute_spectral_relative_error(eta_true, eta_pred):
    # compute relative error in fourier spectrum, using rfft2 over space-time
    true_fft = np.fft.rfft2(eta_true)
    pred_fft = np.fft.rfft2(eta_pred)
    diff_fft = true_fft - pred_fft
    rel_error = np.linalg.norm(diff_fft) / (np.linalg.norm(true_fft) + 1e-12)
    return float(rel_error)


def compute_relative_error(eta_true, eta_pred, floor_ratio=1e-2, floor_min=1e-3):
    # |eta_true - eta_pred| / max(|eta_true|, floor) - floor avoids spikes near zero
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)
    abs_true = np.abs(eta_true)
    floor = np.maximum(np.max(abs_true) * floor_ratio, floor_min)
    denom = np.maximum(abs_true, floor)
    rel_error = np.abs(eta_true - eta_pred) / denom
    return np.nan_to_num(rel_error, posinf=1e3, neginf=0.0)


def spatial_wavenumbers(x):
    # spatial wavenumbers kx = n*pi/L for domain [-L, L]
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.array([0.0], dtype=float)

    domain_length = x[-1] - x[0]
    L = domain_length / 2.0
    n_freq = len(x) // 2 + 1
    return np.arange(n_freq, dtype=float) * np.pi / L


def spectral_mode_index(x):
    # discrete mode index: kx * (2L/pi) so axis shows integer mode numbers
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
            a.set_ylabel('eta(x,t)')
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
        axes[row, 0].set_ylabel('eta(x,t)')
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


def plot_model2_alpha_beta_panel(x, t, eta_true_list, eta_pred_list, param_values, outdir, filename, title,
                                 eval_resolution=None, res_label=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dir_text = os.path.abspath(os.path.join(outdir, 'text'))
    dir_beamer = os.path.abspath(os.path.join(outdir, 'beamer'))
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

    time_rel_norms = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)

    z_global_min = min(float(np.asarray(e).min()) for e in eta_pred_list)
    z_global_max = max(float(np.asarray(e).max()) for e in eta_pred_list)
    z_margin = (z_global_max - z_global_min) * 0.05
    z_lim = [z_global_min - z_margin, z_global_max + z_margin]

    camera = dict(eye=dict(x=-1.5, y=-1.8, z=0.8), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
    scene_kw = dict(
        xaxis_title='Time (t)', yaxis_title='Space (x)', zaxis_title='eta',
        zaxis=dict(range=z_lim), aspectmode='manual', aspectratio=dict(x=2.8, y=1.8, z=0.6),
    )

    # ---------- main panel ----------
    specs = [[{'type': 'scene'}, {'type': 'xy'}]] * n_params + [[{'type': 'xy', 'colspan': 2}, None]]
    subplot_titles = []
    for alpha in param_values:
        subplot_titles += [f'Predicted eta(x,t) - alpha=beta={alpha:.3f}', f'Relative error - alpha=beta={alpha:.3f}']
    subplot_titles.append('Relative error distribution vs. alpha=beta')

    fig = make_subplots(
        rows=n_params + 1, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[0.6, 0.4],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    scene_layout = {}
    for i, (alpha, eta_pred) in enumerate(zip(param_values, eta_pred_list)):
        eta_pred = np.asarray(eta_pred, dtype=float)
        fig.add_trace(go.Surface(
            x=t, y=x, z=eta_pred,
            colorscale='Inferno', cmin=z_global_min, cmax=z_global_max,
            showscale=False,
        ), row=i + 1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=time_rel_norms[i],
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ), row=i + 1, col=2)
        fig.update_xaxes(title_text='Time (t)', row=i + 1, col=2)
        fig.update_yaxes(title_text='Relative error', row=i + 1, col=2)
        scene_key = 'scene' if i == 0 else f'scene{i + 1}'
        scene_layout[scene_key] = dict(**scene_kw, camera=camera)

    for alpha, rel_norm in zip(param_values, time_rel_norms):
        fig.add_trace(go.Box(
            y=rel_norm, name=f'alpha={alpha:.2f}',
            marker_color='#e6550d', fillcolor='#fdd0a2',
            line_color='#e6550d', showlegend=False,
            width=0.2,
        ), row=n_params + 1, col=1)
    fig.update_xaxes(title_text='alpha=beta value', row=n_params + 1, col=1)
    fig.update_yaxes(title_text='Relative error', row=n_params + 1, col=1)

    suptitle = title
    if res_label is not None:
        suptitle = f'{suptitle} (eval res = {res_label})'
    elif eval_resolution is not None:
        suptitle = f'{suptitle} (eval res = {int(eval_resolution)})'
    fig.update_layout(
        title_text=suptitle, title_x=0.5,
        width=1900, height=680 * n_params + 480,
        showlegend=False, **scene_layout,
    )
    try:
        fig.write_image(outpath_text, scale=2.0)
        print(f'alpha/beta panel saved to {outpath_text}')
    except Exception as e:
        print('Erro ao salvar panel como PNG. Instale kaleido: pip install kaleido')
        raise e

    # ---------- beamer rows ----------
    for i, (alpha, eta_pred) in enumerate(zip(param_values, eta_pred_list)):
        eta_pred = np.asarray(eta_pred, dtype=float)
        fig_b = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=[f'Predicted eta(x,t) - alpha=beta={alpha:.3f}', f'Relative error - alpha=beta={alpha:.3f}'],
            horizontal_spacing=0.05,
        )
        fig_b.add_trace(go.Surface(
            x=t, y=x, z=eta_pred,
            colorscale='Inferno', cmin=z_global_min, cmax=z_global_max,
            showscale=False,
        ), row=1, col=1)
        fig_b.add_trace(go.Scatter(
            x=t, y=time_rel_norms[i],
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ), row=1, col=2)
        fig_b.update_xaxes(title_text='Time (t)', row=1, col=2)
        fig_b.update_yaxes(title_text='Relative error', row=1, col=2)
        fig_b.update_layout(
            title_text=f'alpha=beta={alpha:.3f}', title_x=0.5,
            width=1500, height=650, showlegend=False,
            scene=dict(**scene_kw, camera=camera),
        )
        try:
            fig_b.write_image(os.path.join(dir_beamer, f'{basename}_row_{i}_a{alpha:.3f}{ext}'), scale=2.0)
        except Exception as e:
            print('Erro ao salvar beamer row como PNG. Instale kaleido: pip install kaleido')
            raise e

    fig_box = go.Figure()
    for alpha, rel_norm in zip(param_values, time_rel_norms):
        fig_box.add_trace(go.Box(
            y=rel_norm, name=f'alpha={alpha:.2f}',
            marker_color='#e6550d', fillcolor='#fdd0a2',
            line_color='#e6550d', showlegend=False,
            width=0.2,
        ))
    fig_box.update_layout(
        title_text='Relative error distribution vs. alpha=beta', title_x=0.5,
        xaxis_title='alpha=beta value', yaxis_title='Relative error',
        width=1000, height=450,
    )
    try:
        fig_box.write_image(os.path.join(dir_beamer, f'{basename}_boxplot{ext}'), scale=2.0)
    except Exception as e:
        print('Erro ao salvar boxplot como PNG. Instale kaleido: pip install kaleido')
        raise e
    print(f'alpha/beta beamer rows saved to {dir_beamer}')


def plot_model2_resolution_panel(x_list, t_list, eta_true_list, eta_pred_list, resolutions, outdir, filename, title,
                                 eval_alpha_beta=None, param_label=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dir_text = os.path.abspath(os.path.join(outdir, 'text'))
    dir_beamer = os.path.abspath(os.path.join(outdir, 'beamer'))
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

    time_rel_norms = []
    for eta_true, eta_pred in zip(eta_true_list, eta_pred_list):
        rel_norm = np.linalg.norm(eta_true - eta_pred, axis=0) / (np.linalg.norm(eta_true, axis=0) + 1e-12)
        time_rel_norms.append(rel_norm)

    z_global_min = min(float(np.asarray(e).min()) for e in eta_pred_list)
    z_global_max = max(float(np.asarray(e).max()) for e in eta_pred_list)
    z_margin = (z_global_max - z_global_min) * 0.05
    z_lim = [z_global_min - z_margin, z_global_max + z_margin]

    camera = dict(eye=dict(x=-1.5, y=-1.8, z=0.8), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))
    scene_kw = dict(
        xaxis_title='Time (t)', yaxis_title='Space (x)', zaxis_title='eta',
        zaxis=dict(range=z_lim), aspectmode='manual', aspectratio=dict(x=2.8, y=1.8, z=0.6),
    )

    # ---------- main panel ----------
    specs = [[{'type': 'scene'}, {'type': 'xy'}]] * n_res + [[{'type': 'xy', 'colspan': 2}, None]]
    subplot_titles = []
    for res in resolutions:
        subplot_titles += [f'Predicted eta(x,t) - res={int(res)}', f'Relative error - res={int(res)}']
    subplot_titles.append('Relative error distribution vs. resolution')

    fig = make_subplots(
        rows=n_res + 1, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[0.6, 0.4],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    scene_layout = {}
    for i, (res, x_res, t_res, eta_pred) in enumerate(zip(resolutions, x_list, t_list, eta_pred_list)):
        eta_pred = np.asarray(eta_pred, dtype=float)
        fig.add_trace(go.Surface(
            x=t_res, y=x_res, z=eta_pred,
            colorscale='Inferno', cmin=z_global_min, cmax=z_global_max,
            showscale=False,
        ), row=i + 1, col=1)
        fig.add_trace(go.Scatter(
            x=t_res, y=time_rel_norms[i],
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ), row=i + 1, col=2)
        fig.update_xaxes(title_text='Time (t)', row=i + 1, col=2)
        fig.update_yaxes(title_text='Relative error', row=i + 1, col=2)
        scene_key = 'scene' if i == 0 else f'scene{i + 1}'
        scene_layout[scene_key] = dict(**scene_kw, camera=camera)

    for res, rel_norm in zip(resolutions, time_rel_norms):
        fig.add_trace(go.Box(
            y=rel_norm, name=str(int(res)),
            marker_color='#e6550d', fillcolor='#fdae6b',
            line_color='#e6550d', showlegend=False,
            width=0.2,
        ), row=n_res + 1, col=1)
    fig.update_xaxes(title_text='Resolution', row=n_res + 1, col=1)
    fig.update_yaxes(title_text='Relative error', row=n_res + 1, col=1)

    suptitle = title
    if param_label is not None:
        suptitle = f'{suptitle} (eval alpha=beta = {param_label})'
    elif eval_alpha_beta is not None:
        suptitle = f'{suptitle} (eval alpha=beta = {eval_alpha_beta:.3f})'
    fig.update_layout(
        title_text=suptitle, title_x=0.5,
        width=1900, height=680 * n_res + 480,
        showlegend=False, **scene_layout,
    )
    try:
        fig.write_image(outpath_text, scale=2.0)
        print(f'resolution panel saved to {outpath_text}')
    except Exception as e:
        print('Erro ao salvar panel como PNG. Instale kaleido: pip install kaleido')
        raise e

    # ---------- beamer rows ----------
    for i, (res, x_res, t_res, eta_pred) in enumerate(zip(resolutions, x_list, t_list, eta_pred_list)):
        eta_pred = np.asarray(eta_pred, dtype=float)
        fig_b = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=[f'Predicted eta(x,t) - res={int(res)}', f'Relative error - res={int(res)}'],
            horizontal_spacing=0.05,
        )
        fig_b.add_trace(go.Surface(
            x=t_res, y=x_res, z=eta_pred,
            colorscale='Inferno', cmin=z_global_min, cmax=z_global_max,
            showscale=False,
        ), row=1, col=1)
        fig_b.add_trace(go.Scatter(
            x=t_res, y=time_rel_norms[i],
            mode='lines', line=dict(color='crimson', width=1.8), showlegend=False,
        ), row=1, col=2)
        fig_b.update_xaxes(title_text='Time (t)', row=1, col=2)
        fig_b.update_yaxes(title_text='Relative error', row=1, col=2)
        fig_b.update_layout(
            title_text=f'res={int(res)}', title_x=0.5,
            width=1500, height=650, showlegend=False,
            scene=dict(**scene_kw, camera=camera),
        )
        try:
            fig_b.write_image(os.path.join(dir_beamer, f'{basename}_row_{i}_res{int(res)}{ext}'), scale=2.0)
        except Exception as e:
            print('Erro ao salvar beamer row como PNG. Instale kaleido: pip install kaleido')
            raise e

    fig_box = go.Figure()
    for res, rel_norm in zip(resolutions, time_rel_norms):
        fig_box.add_trace(go.Box(
            y=rel_norm, name=str(int(res)),
            marker_color='#e6550d', fillcolor='#fdae6b',
            line_color='#e6550d', showlegend=False,
            width=0.2,
        ))
    fig_box.update_layout(
        title_text='Relative error distribution vs. resolution', title_x=0.5,
        xaxis_title='Resolution', yaxis_title='Relative error',
        width=1000, height=450,
    )
    try:
        fig_box.write_image(os.path.join(dir_beamer, f'{basename}_boxplot{ext}'), scale=2.0)
    except Exception as e:
        print('Erro ao salvar boxplot como PNG. Instale kaleido: pip install kaleido')
        raise e
    print(f'resolution beamer rows saved to {dir_beamer}')


def plot_model2_spectral_panel(x, t, eta_true, eta_pred, outdir, filename, title,
                               n_times=3, eval_resolution=None, res_label=None, param_label=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dir_text = os.path.abspath(os.path.join(outdir, 'text'))
    dir_beamer = os.path.abspath(os.path.join(outdir, 'beamer'))
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
        indices = list(np.arange(len(t), dtype=int))
        target_times = list(t)
    else:
        target_times_arr = np.linspace(t[0], t[-1], n_times)
        selected = []
        seen = set()
        for target in target_times_arr:
            idx = int(np.argmin(np.abs(t - target)))
            if idx not in seen:
                seen.add(idx)
                selected.append((idx, float(target)))
        indices, target_times = zip(*selected) if selected else ([], [])
        indices = list(indices)
        target_times = list(target_times)

    def _tl(tt):
        return f'{int(round(tt))}' if abs(tt - round(tt)) < 1e-6 else f'{tt:.2f}'

    n_snap = len(indices)
    n_rows = n_snap + 1
    specs = [[{'type': 'xy'}, {'type': 'xy'}]] * n_snap + [[{'type': 'xy', 'colspan': 2}, None]]
    subplot_titles = []
    for idx, tt in zip(indices, target_times):
        tl = _tl(tt)
        subplot_titles += [f'Spectrum at t = {tl}', f'Relative spectral error at t = {tl}']
    subplot_titles += ['Mean relative spectral error over time', '']

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.07,
        horizontal_spacing=0.1,
    )

    for row, (idx, tt) in enumerate(zip(indices, target_times)):
        tl = _tl(tt)
        show_legend = (row == 0)
        fig.add_trace(go.Scatter(
            x=kx.tolist(), y=true_spec[:, idx].tolist(),
            mode='lines', name='True', showlegend=show_legend,
            line=dict(color='#222222', width=2.0),
        ), row=row + 1, col=1)
        fig.add_trace(go.Scatter(
            x=kx.tolist(), y=pred_spec[:, idx].tolist(),
            mode='lines', name='Predicted', showlegend=show_legend,
            line=dict(color='#ff7f0e', width=1.8, dash='dash'),
        ), row=row + 1, col=1)
        fig.update_xaxes(title_text='Spectral index n', row=row + 1, col=1)
        fig.update_yaxes(title_text='Amplitude', row=row + 1, col=1)

        mean_rel = float(mean_rel_err_over_time[idx])
        fig.add_trace(go.Scatter(
            x=kx.tolist(), y=rel_err[:, idx].tolist(),
            mode='lines', showlegend=False,
            line=dict(color='crimson', width=1.8),
        ), row=row + 1, col=2)
        fig.add_annotation(
            text=f'Mean: {mean_rel:.2e}',
            xref=f'x{(row * 2 + 2) if row > 0 else "2"} domain', yref=f'y{(row * 2 + 2) if row > 0 else "2"} domain',
            x=0.97, y=0.97, xanchor='right', yanchor='top',
            showarrow=False, font=dict(size=10),
            bgcolor='white', bordercolor='#cccccc', borderwidth=1, borderpad=4,
        )
        fig.update_xaxes(title_text='Spectral index n', row=row + 1, col=2)
        fig.update_yaxes(title_text='Relative error', row=row + 1, col=2)

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=mean_rel_err_over_time.tolist(),
        mode='lines+markers', showlegend=False,
        line=dict(color='#c28b00', width=2.0),
        marker=dict(size=5),
    ), row=n_rows, col=1)
    fig.update_xaxes(title_text='Time (t)', row=n_rows, col=1)
    fig.update_yaxes(title_text='Mean relative error', row=n_rows, col=1)

    suptitle = title
    if param_label is not None:
        suptitle = f'{suptitle} (alpha=beta={param_label})'
    if res_label is not None:
        suptitle = f'{suptitle} (res={res_label})'
    elif eval_resolution is not None:
        suptitle = f'{suptitle} (res={int(eval_resolution)})'

    fig.update_layout(
        title_text=suptitle, title_x=0.5,
        width=1400, height=420 * n_snap + 350,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
    )
    try:
        fig.write_image(outpath_text, scale=2.0)
        print(f'spectral panel text saved to {outpath_text}')
    except Exception as e:
        print('Erro ao salvar spectral panel como PNG. Instale kaleido: pip install kaleido')
        raise e

    # beamer: one figure per time snapshot
    for row, (idx, tt) in enumerate(zip(indices, target_times)):
        tl = _tl(tt)
        fig_b = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Spectrum at t = {tl}', f'Relative spectral error at t = {tl}'],
            horizontal_spacing=0.1,
        )
        fig_b.add_trace(go.Scatter(
            x=kx.tolist(), y=true_spec[:, idx].tolist(),
            mode='lines', name='True', line=dict(color='#222222', width=2.0),
        ), row=1, col=1)
        fig_b.add_trace(go.Scatter(
            x=kx.tolist(), y=pred_spec[:, idx].tolist(),
            mode='lines', name='Predicted', line=dict(color='#ff7f0e', width=1.8, dash='dash'),
        ), row=1, col=1)
        fig_b.add_trace(go.Scatter(
            x=kx.tolist(), y=rel_err[:, idx].tolist(),
            mode='lines', showlegend=False, line=dict(color='crimson', width=1.8),
        ), row=1, col=2)
        fig_b.update_xaxes(title_text='Spectral index n', row=1, col=1)
        fig_b.update_yaxes(title_text='Amplitude', row=1, col=1)
        fig_b.update_xaxes(title_text='Spectral index n', row=1, col=2)
        fig_b.update_yaxes(title_text='Relative error', row=1, col=2)
        fig_b.update_layout(
            title_text=f't = {tl}', title_x=0.5,
            width=1200, height=450,
        )
        try:
            fig_b.write_image(os.path.join(dir_beamer, f'{basename}_row_{row}_t{tl}{ext}'), scale=2.0)
        except Exception as e:
            print('Erro ao salvar beamer spectral row como PNG. Instale kaleido: pip install kaleido')
            raise e

    # beamer: bottom panel (mean error over time)
    fig_bot = go.Figure()
    fig_bot.add_trace(go.Scatter(
        x=t.tolist(), y=mean_rel_err_over_time.tolist(),
        mode='lines+markers', showlegend=False,
        line=dict(color='#c28b00', width=2.0), marker=dict(size=6),
    ))
    fig_bot.update_layout(
        title_text='Mean relative spectral error over time', title_x=0.5,
        xaxis_title='Time (t)', yaxis_title='Mean relative error',
        width=1000, height=400,
    )
    try:
        fig_bot.write_image(os.path.join(dir_beamer, f'{basename}_bottom{ext}'), scale=2.0)
    except Exception as e:
        print('Erro ao salvar beamer spectral bottom como PNG. Instale kaleido: pip install kaleido')
        raise e
    print(f'spectral panel beamer rows saved to {dir_beamer}')


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
        print('  no spectral history data available - retrain models first')
        return
    plot_spectral_bias_panel(models_data, outdir=outdir, filename=filename)


def plot_spectral_bias_panel(models_data, outdir, filename='spectral_bias_evolution.png'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    band_colors = ['#1f77b4', '#9467bd', '#d62728']
    band_labels = ['Low freq (k < Nx/4)', 'Mid freq (Nx/4 <= k < Nx/2)', 'High freq (k >= Nx/2)']
    band_keys = ['low_band', 'mid_band', 'high_band']

    n = len(models_data)
    n_top = n - 1
    n_pair_rows = (n_top + 1) // 2 if n_top > 0 else 0
    n_rows = n_pair_rows + 1

    specs = []
    for r in range(n_pair_rows):
        has_right = (r * 2 + 1) < n_top
        specs.append([{'type': 'xy'}, {'type': 'xy'} if has_right else None])
    specs.append([{'type': 'xy', 'colspan': 2}, None])

    subplot_titles = []
    for r in range(n_pair_rows):
        li, ri = r * 2, r * 2 + 1
        subplot_titles.append(models_data[li][0] if li < n_top else '')
        subplot_titles.append(models_data[ri][0] if ri < n_top else '')
    subplot_titles += [models_data[-1][0], '']

    fig = make_subplots(
        rows=n_rows, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    for i in range(n_top):
        row = i // 2 + 1
        col = i % 2 + 1
        label, sh = models_data[i]
        epochs = np.asarray(sh['epochs']).tolist()
        for color, blab, key in zip(band_colors, band_labels, band_keys):
            vals = np.asarray(sh[key]).tolist()
            fig.add_trace(go.Scatter(
                x=epochs, y=vals, mode='lines', name=blab,
                line=dict(color=color, width=2.0), opacity=0.9,
                showlegend=(i == 0),
            ), row=row, col=col)
        fig.update_yaxes(type='log', row=row, col=col)
        fig.update_xaxes(title_text='Epoch', row=row, col=col)
        fig.update_yaxes(title_text='Rel. spectral error', row=row, col=col)

    last_label, last_sh = models_data[-1]
    epochs = np.asarray(last_sh['epochs']).tolist()
    for color, blab, key in zip(band_colors, band_labels, band_keys):
        vals = np.asarray(last_sh[key]).tolist()
        fig.add_trace(go.Scatter(
            x=epochs, y=vals, mode='lines', name=blab,
            line=dict(color=color, width=2.0), opacity=0.9,
            showlegend=False,
        ), row=n_rows, col=1)
    fig.update_yaxes(type='log', row=n_rows, col=1)
    fig.update_xaxes(title_text='Epoch', row=n_rows, col=1)
    fig.update_yaxes(title_text='Rel. spectral error', row=n_rows, col=1)

    fig.update_layout(
        title_text='Spectral error evolution by frequency band during training', title_x=0.5,
        width=1400, height=480 * n_rows,
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'spectral bias panel saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar spectral bias panel como PNG. Instale kaleido: pip install kaleido')
        raise e


def save_solution_surface_3d_html(x, t, eta_true, eta_pred, outdir, filename, title):
    # plotly interactive html: side-by-side 3d surface of reference and predicted eta(x,t)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    outpath = os.path.join(outdir, base + '.html')

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    z_min = min(float(eta_true.min()), float(eta_pred.min()))
    z_max = max(float(eta_true.max()), float(eta_pred.max()))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Reference', 'Prediction'],
        horizontal_spacing=0.05,
    )

    surface_kw = dict(x=x.tolist(), y=t.tolist(), colorscale='Viridis', cmin=z_min, cmax=z_max)
    fig.add_trace(go.Surface(z=eta_true.T.tolist(), showscale=False, **surface_kw), row=1, col=1)
    fig.add_trace(go.Surface(z=eta_pred.T.tolist(), colorbar=dict(title='eta(x,t)', x=1.02), **surface_kw), row=1, col=2)

    camera = dict(eye=dict(x=1.8, y=-1.8, z=1.2))
    scene_common = dict(xaxis_title='Space (x)', yaxis_title='Time (t)', zaxis_title='eta(x,t)', camera=camera)
    fig.update_layout(
        title_text=title, title_x=0.5,
        width=1400, height=650,
        scene=scene_common, scene2=scene_common,
    )
    fig.write_html(outpath)
    print(f'solution surface html saved to {outpath}')


def save_solution_plotly_html(x, t, eta_true, eta_pred, outdir, filename, title):
    # plotly interactive html: reference + predicted eta(x,t) heatmaps + relative error
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _ensure_outdir(outdir)
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    outpath = os.path.join(outdir, base + '.html')

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    rel_error = compute_relative_error(eta_true, eta_pred)
    rel_error = np.clip(rel_error, 0.0, 1.0)
    time_rel_norm = (
        np.linalg.norm(np.abs(eta_true - eta_pred), axis=0)
        / (np.linalg.norm(eta_true, axis=0) + 1e-8)
    )

    z_min = min(float(eta_true.min()), float(eta_pred.min()))
    z_max = max(float(eta_true.max()), float(eta_pred.max()))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Reference eta(x,t)', 'Predicted eta(x,t)', 'Relative Error', 'Error Norm Over Time'],
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )

    heatmap_kw = dict(x=x.tolist(), y=t.tolist(), colorscale='Viridis', zmin=z_min, zmax=z_max)
    fig.add_trace(go.Heatmap(z=eta_true.T.tolist(), showscale=False, **heatmap_kw), row=1, col=1)
    fig.add_trace(go.Heatmap(z=eta_pred.T.tolist(), colorbar=dict(title='eta', x=1.02, len=0.45, y=0.78), **heatmap_kw), row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=rel_error.T.tolist(), x=x.tolist(), y=t.tolist(),
        colorscale='Magma', zmin=0.0, zmax=1.0,
        colorbar=dict(title='Rel. err', x=1.02, len=0.45, y=0.22),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=time_rel_norm.tolist(),
        mode='lines', line=dict(color='crimson', width=2), name='error norm',
    ), row=2, col=2)

    fig.update_xaxes(title_text='Space (x)', row=1)
    fig.update_xaxes(title_text='Space (x)', row=2, col=1)
    fig.update_xaxes(title_text='Time (t)', row=2, col=2)
    fig.update_yaxes(title_text='Time (t)', col=1)
    fig.update_yaxes(title_text='Relative error', row=2, col=2)
    fig.update_layout(title_text=title, title_x=0.5, width=1200, height=900, showlegend=False)

    fig.write_html(outpath)
    print(f'solution html saved to {outpath}')


def plot_pinn_ntk_panel(theta_rel_histories, k_rel_histories, log_epochs,
                         eigenvalues_init, eigenvalues_final, widths, outdir, filename):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    _ensure_outdir(outdir)
    outpath = os.path.join(outdir, filename)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'Relative parameter change',
            'Relative NTK change',
            'NTK eigenvalue spectrum',
        ],
        horizontal_spacing=0.1,
    )

    for i, (w, theta_hist, k_hist) in enumerate(zip(widths, theta_rel_histories, k_rel_histories)):
        color = _colors[i % len(_colors)]
        name = f'width={w}'
        fig.add_trace(go.Scatter(
            x=log_epochs, y=theta_hist,
            mode='lines', name=name,
            line=dict(color=color, width=2.0),
            legendgroup=name, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=log_epochs, y=k_hist,
            mode='lines', name=name,
            line=dict(color=color, width=2.0),
            legendgroup=name, showlegend=False,
        ), row=1, col=2)

    for i, (w, ev_init, ev_final) in enumerate(zip(widths, eigenvalues_init, eigenvalues_final)):
        color = _colors[i % len(_colors)]
        name = f'width={w}'
        idx = np.arange(1, len(ev_init) + 1).tolist()
        fig.add_trace(go.Scatter(
            x=idx, y=list(ev_init),
            mode='lines', name=f'{name} init',
            line=dict(color=color, width=2.0, dash='solid'),
            legendgroup=name, showlegend=False,
        ), row=1, col=3)
        idx_f = np.arange(1, len(ev_final) + 1).tolist()
        fig.add_trace(go.Scatter(
            x=idx_f, y=list(ev_final),
            mode='lines', name=f'{name} final',
            line=dict(color=color, width=2.0, dash='dash'),
            legendgroup=name, showlegend=False,
        ), row=1, col=3)

    fig.update_xaxes(title_text='Iteration', row=1, col=1)
    fig.update_yaxes(title_text='Relative parameter change', row=1, col=1)
    fig.update_xaxes(title_text='Iteration', row=1, col=2)
    fig.update_yaxes(title_text='Relative NTK change', row=1, col=2)
    fig.update_xaxes(title_text='Eigenvalue index', type='log', row=1, col=3)
    fig.update_yaxes(title_text='Eigenvalue', type='log', row=1, col=3)

    fig.update_layout(
        title_text='NTK Analysis (Wang et al. 2020)',
        title_x=0.5,
        template='plotly_white',
        width=1800, height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
    )
    try:
        fig.write_image(outpath, scale=2.0)
        print(f'ntk panel saved to {outpath}')
    except Exception as e:
        print('Erro ao salvar ntk panel como PNG. Instale kaleido: pip install kaleido')
        raise e