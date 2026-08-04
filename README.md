# Scientific Machine Learning Methods for the Nonlinear Boussinesq System of Equations

> Undergraduate thesis (*Trabalho de Conclusão de Curso*) submitted in partial fulfillment of the requirements for the degree of **Bacharel em Matemática Industrial**, Departamento de Matemática, **Universidade Federal do Paraná (UFPR)**.

| | |
|---|---|
| **Author** | Samuel Kutz Paranhos |
| **Advisor** | Prof. Roberto Ribeiro |
| **Institution** | Departamento de Matemática, UFPR — Curitiba, Brazil |
| **Year** | 2026 |
| **Area** | Matemática Industrial / Scientific Machine Learning |
| **License** | [MIT](LICENSE) |

---

## Abstract

This work investigates the effectiveness of Scientific Machine Learning (SciML) methods in solving and approximating the nonlinear Boussinesq system, a system of Partial Differential Equations (PDEs) used to model water-wave dynamics. The Boussinesq system is treated as a PDE dependent on two parameters — the nonlinearity coefficient and the dispersion coefficient. A pseudospectral solver generates a dataset of reference solutions across parameters, which then serves as the training basis for three SciML architectures:

- **Physics-Informed Neural Networks (PINNs)** — trained at a single parameter, both with data constraints and in a purely physics-informed regime;
- **Fourier Neural Operators (FNOs)** — a purely supervised, multi-parameter regime using only data, with no differential-operator information;
- **Physics-Informed Neural Operators (PINOs)** — an FNO that adds the PDE differential operator to the loss function.

Fidelity is measured through the evolution of the **spectral error** and the **relative error** over space-time against the reference solution. The central finding is that traditional PINNs — especially without data — suffer from **spectral bias**, learning low-frequency components first and struggling with high-frequency detail as nonlinearity grows. Operator-learning models (FNO and PINO with data), learning directly in frequency space, substantially attenuate this in the low and mid bands, though **no method removes it entirely**: a residual high-frequency error persists across all architectures. Adding the physics residual further confers **temporal stability**, suppressing the Gibbs artifact that degrades the FNO at the final time. The conclusion is that including data in the loss is the most effective strategy to *mitigate — though not eliminate —* the spectral bias common to these architectures.

**Keywords:** Scientific Machine Learning; Partial Differential Equations; Boussinesq System; Physics-Informed Neural Networks.

---

## Table of contents

- [The mathematical problem](#the-mathematical-problem)
- [Methods](#methods)
- [Repository structure](#repository-structure)
- [Requirements and environment](#requirements-and-environment)
- [Reproducing the results](#reproducing-the-results)
- [Outputs](#outputs)
- [The thesis document](#the-thesis-document)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## The mathematical problem

The one-dimensional Boussinesq system models shallow-water waves through two coupled fields — the free-surface displacement `η(x,t)` and the horizontal velocity `u(x,t)`:

```
η_t + u_x + α (η u)_x = 0
u_t − (β/3) u_{xxt} + η_x + α u u_x = 0
```

The coefficients `α` (nonlinearity) and `β` (dispersion) define the family of problems. Throughout the experiments they are held equal, `α = β`, sweeping a single difficulty axis from the near-linear regime (small `α = β`, energy in a few low-frequency modes) to the strongly nonlinear one (large `α = β`, sharp crests spread over a broad band of wavenumbers). The initial condition is a fixed localized wave:

```
η(x,0) = A · sech²(x),   u(x,0) = 0.
```

Reference solutions are produced by a **pseudospectral solver** (Fourier transforms for the spatial derivatives, RK4 in time), whose spectral spatial accuracy makes it the natural high-fidelity baseline for a dispersive wave problem and the accuracy ceiling against which every SciML method is measured.

## Methods

| Method | File | Regime | What it learns |
|---|---|---|---|
| **FNO** | [`src/FNO/FNO.py`](src/FNO/FNO.py) | supervised | Spectral operator mapping parameters + initial data → full space-time solution. |
| **PINO** | [`src/PINO/PINO.py`](src/PINO/PINO.py) | with / without data | Same operator backbone as the FNO, plus the Boussinesq residual in the loss. |
| **PINN** | [`src/PINN/PINN.py`](src/PINN/PINN.py) | with / without data | Continuous map `(x,t) → (η,u)` trained on the PDE residual via automatic differentiation. |

The FNO and PINO share the same spectral backbone, so any difference in behaviour isolates the effect of the physics term. The PINN is a single-parameter model, whereas the operators cover the whole parameter family from one training. A **Neural Tangent Kernel (NTK)** probe ([`src/experiments/spectral_bias_ntk.py`](src/experiments/spectral_bias_ntk.py)) examines the spectral-bias mechanism on the data-only MLP fitting `η` of the same Boussinesq solution at `α = β = 3.21`: it measures the frozen-kernel predictions (per-eigendirection decay, per-mode spectral decay, closed-form iteration counts) and, across widths `8/128/512`, the parameter and kernel drift together with the NTK eigenvalue spectrum.

## Repository structure

```
.
├── src/
│   ├── main.py                  # single entry point: dataset → train → evaluate
│   ├── tools.py                 # normalization, save/load, L2 / relative-L2 losses, metrics
│   ├── _plots.py                # shared evaluation-figure utilities
│   ├── settings.json            # all experiment hyperparameters and evaluation config
│   ├── BOUSSINESQ/              # PDE definition, pseudospectral solver, dataset generation
│   ├── FNO/                     # Fourier Neural Operator + training + plots
│   ├── PINO/                    # Physics-Informed Neural Operator + training + plots
│   └── PINN/                    # Physics-Informed Neural Network + training + plots
├── tex/                         # LaTeX sources of the thesis (see "The thesis document")
├── results/                     # generated dataset, model weights, metadata, figures
├── beamers/                     # presentation slides
├── requirements.txt
└── LICENSE
```

## Requirements and environment

The experiments were implemented from scratch — the pseudospectral solver, the three architectures and every training loop — with no pre-trained weights, external datasets or high-level operator-learning libraries.

- **Python** 3.11
- **PyTorch** 2.12 with CUDA 13.0 (`torch==2.12.0+cu130`)
- NumPy, Matplotlib, Plotly (full pinned list in [`requirements.txt`](requirements.txt))

```bash
pip install -r requirements.txt
# PyTorch with CUDA 13.0 (as pinned in requirements.txt):
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```

Reference hardware: a single **NVIDIA GeForce GTX 1660 SUPER** (6 GB). A CPU-only install runs but is considerably slower.

## Reproducing the results

Every experiment is driven by [`src/settings.json`](src/settings.json) and a single fixed global seed (`37`), with cuDNN set to deterministic mode. The full pipeline — generate the shared dataset, train all five models (FNO; PINO with/without data; PINN with/without data), run the NTK spectral-bias probe, and render every evaluation figure — is reproduced by one command:

```bash
python src/main.py
```

> **Note.** `src/main.py` is the authoritative entry point: it reads `settings.json` and generates the exact dataset used in the thesis (domain `T = 30`, resolution `128`, 20 parameter values on `linspace(0.1, 4.0, 20)`). The helper `src/BOUSSINESQ/run_dataset.py` exists for internal use and carries different standalone defaults — do not rely on it to reproduce the thesis dataset.

Key configuration (all in `settings.json`): spatial half-length `L = 60`, final time `T = 30`, dataset resolution `128`, training grid `α = β ∈ linspace(0.1, 4.0, 20)`, evaluation parameters `{0.1, 3.21, 4.2}`, evaluation resolutions `{128, 256, 512}`.

## Outputs

Results are written to a standardized `results/` hierarchy:

- `results/models/boussinesq_dataset.pth` — shared reference dataset;
- `results/models/weights/{fno,pino,pinn}/…` — trained weights per model and regime;
- `results/models/metadata/` — training metadata for every experiment;
- `results/imgs/{fno,pino,pinn}/…` — evaluation figures (spectral fidelity, error panels, resolution and nonlinearity sweeps);
- `results/imgs/gifs/` — animations of the wave dynamics.

## The thesis document

The written thesis lives under [`tex/`](tex/) and is built with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd tex
tectonic -X build          # configuration in tex/Tectonic.toml
```

Chapters: introduction, mathematical background, methodology, numerical results, and conclusions
([`tex/src/chapters/`](tex/src/chapters/)). The compiled PDF is produced at `tex/build/TCC/TCC.pdf`.

## Citation

If you use this code or refer to this work, please cite:

```bibtex
@thesis{paranhos2026sciml,
  author      = {Paranhos, Samuel Kutz},
  title       = {Scientific Machine Learning Methods for the Nonlinear Boussinesq System of Equations},
  type        = {Bachelor's thesis (Trabalho de Conclus{\~a}o de Curso)},
  institution = {Universidade Federal do Paran{\'a}},
  address     = {Curitiba, Brazil},
  year        = {2026},
  url         = {https://github.com/samuelkutz/TCC}
}
```

## License

Released under the [MIT License](LICENSE) — © 2026 Samuel Kutz Paranhos.

## Acknowledgements

Developed at the Departamento de Matemática, UFPR, under the guidance of Prof. Roberto Ribeiro, with the support of the Laboratório de Dinâmica de Fluidos (LabFluid) and CNPq.
