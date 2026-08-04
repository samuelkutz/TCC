"""Animated solution figures (the .gif companions to the static panels)."""

import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.plots.figures.style import ensure_outdir, mpl_legend_below


def save_solution_gif(x, t, eta_true, eta_pred, outdir, filename, title_prefix, fps=20):
    # animate eta(x, t) over time: reference (gray) vs prediction (orange line),
    # with the evolving time value shown in the title. Fields are (Nx, Nt).
    from matplotlib.animation import FuncAnimation, PillowWriter

    ensure_outdir(outdir)
    base = filename.rsplit('.', 1)[0] if '.' in filename else filename
    outpath = os.path.join(outdir, base + '.gif')

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    eta_true = np.asarray(eta_true, dtype=float)
    eta_pred = np.asarray(eta_pred, dtype=float)

    n_frames = int(min(eta_true.shape[1], eta_pred.shape[1], t.shape[0]))

    y_min = float(min(eta_true.min(), eta_pred.min()))
    y_max = float(max(eta_true.max(), eta_pred.max()))
    pad = 0.08 * (y_max - y_min + 1e-9)
    y_min, y_max = y_min - pad, y_max + pad

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    line_true, = ax.plot([], [], color='0.55', lw=2.0, label='Reference (pseudospectral)')
    line_pred, = ax.plot([], [], color='#E66C11', lw=2.4, label='Prediction')
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\eta(x,t)$')
    mpl_legend_below(ax)
    # reserve room at the top so the per-frame title is never clipped
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.13)
    title = ax.set_title(f'{title_prefix},   t = {t[0]:.2f}', color='black')

    def _init():
        line_true.set_data([], [])
        line_pred.set_data([], [])
        return line_true, line_pred

    def _update(n):
        line_true.set_data(x, eta_true[:, n])
        line_pred.set_data(x, eta_pred[:, n])
        title.set_text(f'{title_prefix},   t = {t[n]:.2f}')
        return line_true, line_pred

    anim = FuncAnimation(fig, _update, frames=n_frames, init_func=_init, blit=False)
    anim.save(outpath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'solution gif saved to {outpath}')
    return outpath
