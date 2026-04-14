import os
import numpy as np
import torch

from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
from FNO.FNO import FNO2d
from tools import load_model
from plots import plot_training_statistics, plot_relative_error_panel, save_solution_gif, compute_spectral_relative_error

RESULTS_DIR = 'RESULTS'
EVAL_PARAMS = [0.1, 2.75, 5.75]
RESOLUTIONS = [256, 128, 64]


def eval_fno(model_metadata_file=None):
    if model_metadata_file is None:
        model_metadata_file = os.path.join(RESULTS_DIR, 'fno', 'models', 'fno_model_metadata.pth')

    model_metadata = torch.load(model_metadata_file, map_location='cpu')
    params = model_metadata['params']
    model_file = model_metadata['model_file']

    outdir = os.path.dirname(os.path.dirname(model_metadata_file))
    os.makedirs(outdir, exist_ok=True)

    plot_training_statistics(
        [model_metadata['train_history']],
        ['fno'],
        outdir=outdir,
        filename='fno_training_statistics.png',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO2d(modes1=params['modes1'], modes2=params['modes2'], width=params['width']).to(device)
    load_model(model_file, model, device=device)

    print('start fno evaluation')
    for val in EVAL_PARAMS:
        resolutions = RESOLUTIONS if val == EVAL_PARAMS[1] else [256]
        for res in resolutions:
            bsq = Boussinesq(-30, 30, 0, 15, val, val)
            solver = PseudoSpectralBoussinesq(bsq, Nx=res, Nt=res - 1, device=device)
            x, t, eta_true, u_true = solver.solve()

            eta_true_t = eta_true.T
            eta0 = np.asarray(eta_true[0, :], dtype=np.float32)
            u0 = np.asarray(u_true[0, :], dtype=np.float32)
            ch0 = np.tile(eta0[:, None], (1, res))
            ch1 = np.tile(u0[:, None], (1, res))
            ch2 = np.ones((res, res), dtype=np.float32) * val
            ch3 = np.ones((res, res), dtype=np.float32) * val
            input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1)
            input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                pred = model(input_tensor)

            eta_pred = pred.squeeze().cpu().numpy()[0, :, :]
            rel_error = np.linalg.norm(eta_true_t - eta_pred) / (np.linalg.norm(eta_true_t) + 1e-8)
            spec_error = compute_spectral_relative_error(eta_true_t, eta_pred)
            print(f'val={val:.3f} res={res} rel_error={rel_error:.4e} spec_error={spec_error:.4e}')

            plot_relative_error_panel(
                x,
                t,
                eta_true_t,
                eta_pred,
                times=[0.0, 15.0],
                outdir=outdir,
                filename=f'fno_summary_a{val:.3f}_res{res}.png',
                title=f'fno relative error a={val:.3f} res={res}',
            )
            if res == 256:
                save_solution_gif(
                    x,
                    t,
                    eta_true_t,
                    eta_pred,
                    outdir=outdir,
                    filename=f'fno_animation_a{val:.3f}_res{res}.gif',
                    title=f'fno animation a={val:.3f} res={res}',
                )


if __name__ == '__main__':
    eval_fno()
