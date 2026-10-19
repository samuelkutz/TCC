"""Figuras do pôster no tamanho de impressão (A0), com fonte de pôster.

Os PNGs da tese foram desenhados para A4; reduzidos ao espaço do pôster, os
números dos eixos ficavam com metade do tamanho do texto. Aqui cada figura é
desenhada já na largura em que é impressa, então os tamanhos em pt valem no papel.
Mesmos dados e mesmas cores da tese: as curvas de banda vêm do JSON salvo pela
tese; o erro relativo e o espectro do PINO com dados são recalculados dos pesos.

Rodar a partir da raiz do repositório:
    py -3.11 tex/poster/ermac-tcc/figuras_poster.py
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'src'))
os.chdir(REPO)

from config import Config
from experiments.common import resolve_device
from experiments.evaluate import operator_predictor
from experiments.plots.figures import (
    compute_spectral_error, spectral_amplitude, spectral_mode_index,
    time_relative_error_norm,
)
from experiments.plots.figures.style import (
    THESIS_BAND_COLORS, THESIS_ERROR_COLOR, THESIS_PREDICTION_COLOR,
    THESIS_REFERENCE_COLOR,
)
from experiments.plots.pino import _load as load_pino

OUTDIR = os.path.join(REPO, 'tex', 'poster', 'ermac-tcc', 'imgs')
CM = 1 / 2.54

# larguras impressas: painel da linha inteira (Figura 1) e meia coluna (Figuras 2 e 3)
FULL_PANEL_W, FULL_PANEL_H = 24.0 * CM, 13.7 * CM
HALF_PANEL_W, HALF_PANEL_H = 17.8 * CM, 11.0 * CM

# Lato, a mesma fonte do pôster, tirada do cache do Tectonic
for path in glob.glob(os.path.join(os.environ.get('LOCALAPPDATA', ''), 'TectonicProject',
                                   'Tectonic', 'bundles', 'data', '*', 'Lato-Regular.ttf'))[:1]:
    font_manager.fontManager.addfont(path)
    plt.rcParams['font.family'] = 'Lato'

plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.size': 20,
    'axes.labelsize': 23,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'axes.edgecolor': '#999999',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.color': '#e6e6e6',
    'grid.linewidth': 1.0,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 3.0,
    'legend.frameon': False,
})


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print('figura salva em', path)


def band_figures():
    path = os.path.join('results', 'imgs', 'boussinesq_sciml', 'comparison',
                        'spectral_bias_evolution_metrics.json')
    with open(path, encoding='utf-8') as fh:
        metrics = json.load(fh)
    runs = [('fno_pure_data', 'bandas_fno.png'),
            ('pino_data_and_physics', 'bandas_pino_dados.png'),
            ('pino_pure_physics', 'bandas_pino_puro.png')]
    keys = ('low_band', 'mid_band', 'high_band')
    names = ('Baixa', 'Média', 'Alta')
    # uma escala vertical comum aos três painéis, para a comparação ser direta
    values = np.concatenate([np.asarray(metrics[r][k]) for r, _ in runs for k in keys])
    ylim = (values.min() / 1.4, values.max() * 1.4)
    for run, filename in runs:
        data = metrics[run]
        fig, ax = plt.subplots(figsize=(HALF_PANEL_W, HALF_PANEL_H), layout='constrained')
        for color, key, name in zip(THESIS_BAND_COLORS, keys, names):
            ax.plot(data['epochs'], data[key], color=color, label=name)
        ax.set_yscale('log')
        ax.set_ylim(*ylim)
        ax.set_xlabel('Iteração')
        ax.set_ylabel(r'$E_{\mathrm{band}}$')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f'{int(v):,}'.replace(',', '.')))
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                  handlelength=1.6, columnspacing=1.2, borderaxespad=0.2)
        save(fig, filename)


def pino_figures():
    cfg = Config()
    device = resolve_device()
    param = cfg.median_param
    metadata, model = load_pino(cfg.metadata_file('pino', 'pino', 'data_and_physics'), device)
    predict = operator_predictor(model, metadata.get('norm_stats'), device)
    x, t, eta_true, eta_pred = predict(param, cfg.x_limit, cfg.t_limit, int(cfg.spectral_res))
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    eta_true, eta_pred = np.asarray(eta_true, dtype=float), np.asarray(eta_pred, dtype=float)

    # erro relativo L2 em cada instante (eq. m_rel_time da tese)
    rel = np.asarray(time_relative_error_norm(eta_true, eta_pred), dtype=float)
    fig, ax = plt.subplots(figsize=(FULL_PANEL_W, FULL_PANEL_H), layout='constrained')
    ax.plot(t, rel, color=THESIS_ERROR_COLOR)
    ax.set_xlabel('Tempo $t$')
    ax.set_ylabel(r'$E(t_n)$')
    ax.set_xlim(t[0], t[-1])
    save(fig, 'erro_tempo_pino_dados.png')

    # espectro em t = T/2, o instante do meio do painel espectral da tese
    idx = int(np.argmin(np.abs(t - 0.5 * (t[0] + t[-1]))))
    k = np.asarray(spectral_mode_index(x))
    true_spec = spectral_amplitude(eta_true, axis=0)
    pred_spec = spectral_amplitude(eta_pred, axis=0)
    abs_err = compute_spectral_error(true_spec, pred_spec)
    print(f'espectro em t = {t[idx]:.2f}, {len(k)} modos')

    fig, ax = plt.subplots(figsize=(HALF_PANEL_W, HALF_PANEL_H), layout='constrained')
    ax.plot(k, true_spec[:, idx], color=THESIS_REFERENCE_COLOR, lw=2.6, label='Referência')
    ax.plot(k, pred_spec[:, idx], color=THESIS_PREDICTION_COLOR, lw=2.6, ls='--', label='Previsão')
    ax.set_yscale('log')
    ax.set_xlabel('Modo $k$')
    ax.set_ylabel(r'$|\hat{\eta}(k,t)|$')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2)
    save(fig, 'espectro_pino_dados.png')

    fig, ax = plt.subplots(figsize=(HALF_PANEL_W, HALF_PANEL_H), layout='constrained')
    ax.plot(k, abs_err[:, idx], color=THESIS_ERROR_COLOR, lw=2.6)
    ax.set_yscale('log')
    ax.set_xlabel('Modo $k$')
    ax.set_ylabel(r'$E_{\mathrm{spec}}(k,t)$')
    save(fig, 'erro_espectral_pino_dados.png')


if __name__ == '__main__':
    band_figures()
    pino_figures()
