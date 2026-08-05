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

This work investigates the effectiveness of Scientific Machine Learning (SciML) methods in solving and approximating the nonlinear Boussinesq system, a system of Partial Differential Equations (PDEs) used to model water-wave dynamics. The Boussinesq system is treated as a PDE dependent on two parameters — the nonlinearity coefficient and the dispersion coefficient. A pseudospectral solver generates a dataset of reference solutions across parameters, which then serves as the training basis for four architectures across six trained models:

- **Multilayer Perceptron (MLP)** — a pointwise, data-only baseline and the object of the spectral-bias probe;
- **Physics-Informed Neural Networks (PINNs)** — pointwise, trained at a single parameter, both with data and in a purely physics-informed regime;
- **Fourier Neural Operators (FNOs)** — a purely supervised, multi-parameter regime using only data, with no differential-operator information;
- **Physics-Informed Neural Operators (PINOs)** — an FNO that adds the PDE differential operator to the loss, with and without data.

Fidelity is measured through the evolution of the **spectral error** and the **relative error** over space-time against the reference. The recurring finding is that every architecture exhibits **spectral bias**, learning low-frequency structure first and leaving high-frequency detail last, an effect that sharpens as nonlinearity grows. A **Neural Tangent Kernel (NTK)** probe traces the effect to its cause — a steep kernel eigenvalue spectrum — directly on the Boussinesq system. Operator-learning models (FNO and PINO with data), learning in frequency space, compress the bias in the low and mid bands, though **no method removes it**: a high-frequency residual persists across all architectures. Adding the physics residual further confers **temporal stability**, suppressing the Gibbs artifact that degrades the FNO at the final time. Including data in the loss is the more effective lever for *mitigating — though not eliminating —* the spectral bias these architectures share; the finest scales remain the pseudospectral solver's.

**Keywords:** Scientific Machine Learning; Partial Differential Equations; Boussinesq System; Physics-Informed Neural Networks.

---

## Table of contents

- [The mathematical problem](#the-mathematical-problem)
- [Methods](#methods)
- [Theory and key findings](#theory-and-key-findings)
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

Four architectures give rise to six trained models, arranged along two axes: the architecture family (pointwise vs. operator) and the training regime (what the loss is allowed to see).

| Method | File | Regime(s) | What it learns |
|---|---|---|---|
| **MLP** | [`src/methods/mlp.py`](src/methods/mlp.py) | pure data | Pointwise map `(x,t) → η`; the plainest instance of spectral bias and the basis of the NTK probe. |
| **PINN** | [`src/methods/pinn.py`](src/methods/pinn.py) | data and physics / pure physics | Pointwise map `(x,t) → (η,u)` trained on data and/or the PDE residual via automatic differentiation. |
| **FNO** | [`src/methods/fno.py`](src/methods/fno.py) | pure data | Spectral operator mapping parameters + initial data → full space-time solution. |
| **PINO** | [`src/methods/pino.py`](src/methods/pino.py) | data and physics / pure physics | Same operator backbone as the FNO, plus the Boussinesq residual in the loss. |

The **pointwise** models (MLP, PINN) represent one solution as a continuous map and are retrained for each parameter; the **operator** models (FNO, PINO) learn the whole parameter family from a single training and answer a new parameter in one forward pass. The FNO and PINO share the same spectral backbone, so any difference in behaviour isolates the effect of the physics term.

A **Neural Tangent Kernel (NTK)** probe ([`src/experiments/train/ntk.py`](src/experiments/train/ntk.py)) examines the spectral-bias mechanism on a minimal `ℝ → ℝ` MLP fitting `η(x, 15)` of the Boussinesq solution at `α = β = 3.21`. Across widths `{8, 64, 512}` and 500,000 full-batch gradient-descent steps, it measures the frozen-kernel predictions (per-eigendirection decay, per-mode spectral decay, closed-form iteration counts) together with the parameter and kernel drift and the NTK eigenvalue spectrum.

## Theory and key findings

### The mechanism: spectral bias and the Neural Tangent Kernel

Gradient descent does not learn all frequencies at once. In the *lazy-training* regime, where a wide network's tangent kernel stays close to its value at initialization, the error along each kernel eigendirection decays at a rate set by that direction's eigenvalue, so smooth, low-frequency structure is resolved long before high-frequency detail. This is the *frequency principle*, and its driver is the shape of the NTK spectrum. For a sharp Boussinesq profile that spectrum is punishing: the eigenvalues fall by many orders of magnitude within about a dozen modes, so under a fixed kernel only a handful of smooth directions are reachable in any practical number of steps — and a soliton pair does not live there.

### What the NTK probe reveals

- **The frozen-kernel predictions hold where they should.** Per-eigendirection error decays geometrically at the predicted rate, in exact order of eigenvalue; the frequency principle is visible mode by mode; and the closed-form iteration counts, which carry no dependence on width, place the reachable modes of all three networks on one diagonal.
- **The trap is real, and severe.** The narrowest network never leaves a nearly flat curve, staying at almost the full relative error of its untrained state — a fixed kernel simply cannot reach the sharp crests.
- **But wider networks escape it — the honest twist.** As training proceeds the wider networks drift *out* of the lazy regime: their kernels reshape, the eigenvalue spectrum flattens, and directions the fixed-kernel theory declares unreachable become learnable, so the crests are recovered. The escape arrives only in the final stretch of a half-million-iteration run, however, so it is a capacity-related effect rather than a usable remedy. The probe *exposes* the mechanism; it does not remove it.

### What the six trained models show

- **The high-frequency band is the wall.** Every architecture resolves low frequencies first and high frequencies last; the high band stays at or above the magnitude of the reference for every model except the plain FNO, which alone pushes it clearly below.
- **Data is the main lever, and physics does not substitute for it.** Adding the supervised term lowers the low- and mid-band error in both the PINN and PINO families; the physics residual alone stays close to the biased baseline.
- **Operators are more accurate at low and mid wavenumbers, and cover the family.** Learning in frequency space, the FNO and PINO track the low/mid spectrum below the pointwise models, and a single training answers the whole parameter family in one forward pass.
- **Physics buys temporal stability.** Treating time as a periodic axis makes the FNO spike at the final time (a Gibbs artifact); the PINO's residual suppresses that spike and keeps the late-time error bounded.
- **Zero-shot superresolution is geometric, not accuracy-preserving.** Queried off its training grid the operator still returns a coherent field, but its fidelity is anchored to the resolution its physics residual actually saw and degrades away from it.

### The bottom line

No method removes spectral bias; the operators compress it and physics stabilizes it in time, but the finest scales remain the pseudospectral solver's alone. For the nonlinear Boussinesq system these SciML methods are best read as **complements** to the classical solver — amortizing the cost of a parameter sweep and, when physics is added, gaining temporal stability — rather than replacements for it.

## Repository structure

```
.
├── src/
│   ├── main.py                 # single entry point: dataset → train → evaluate/plot
│   ├── config.py               # experiment configuration, built from settings.json
│   ├── settings.json           # all experiment hyperparameters and evaluation config
│   ├── tools.py                # normalization, save/load, losses, spectral-error metrics
│   ├── methods/                # architectures + PDE: boussinesq, mlp, pinn, fno, pino
│   └── experiments/
│       ├── dataset.py          # pseudospectral reference-dataset generation
│       ├── common.py, evaluate.py, predict.py
│       ├── train/              # training loops: mlp, pinn, fno, pino, and ntk (the probe)
│       └── plots/              # evaluation figures and animations
├── tex/text/                   # LaTeX sources of the thesis (see "The thesis document")
├── results/                    # generated dataset, model weights, metadata, figures, gifs
├── requirements.txt
└── LICENSE
```

## Requirements and environment

The experiments were implemented from scratch — the pseudospectral solver, the four architectures and every training loop — with no pre-trained weights, external datasets or high-level operator-learning libraries.

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

Every experiment is driven by [`src/settings.json`](src/settings.json) and a single fixed global seed (`37`), with cuDNN set to deterministic mode. The full pipeline — generate the shared dataset, train all six models (MLP; FNO; PINN with/without data; PINO with/without data), run the NTK spectral-bias probe, and render every evaluation figure and animation — is reproduced by one command:

```bash
python src/main.py
```

To re-render every figure and gif from the **saved weights** alone, without retraining:

```bash
python src/main.py --plots-only
```

> **Note.** `src/main.py` is the authoritative entry point: it reads `settings.json` and generates the exact dataset used in the thesis (domain `T = 30`, resolution `128`, 20 parameter values on `linspace(0.1, 4.0, 20)`).

Key configuration (all in `settings.json`): spatial half-length `L = 60`, final time `T = 30`, dataset resolution `128`, training grid `α = β ∈ linspace(0.1, 4.0, 20)`, evaluation parameters `{0.1, 3.21, 4.2}`, evaluation resolutions `{128, 256, 512}`, and the probe widths `{8, 64, 512}` over `500,000` iterations.

## Outputs

Results are written to a standardized `results/` hierarchy:

- `results/models/boussinesq_dataset.pth` — shared reference dataset;
- `results/models/weights/{mlp,fno,pinn,pino}/…` — trained weights per model and regime;
- `results/models/metadata/…` — training metadata for every experiment;
- `results/imgs/boussinesq_sciml/{nn,no,comparison}/…` — the six models' evaluation figures (spectral fidelity, error panels, resolution and nonlinearity sweeps, band-tracking), grouped by pointwise (`nn`) and operator (`no`) families;
- `results/imgs/boussinesq_spectral_bias_mlp/…` — the NTK spectral-bias probe figures;
- `results/imgs/setup/` — the soliton reference profile;
- `results/imgs/gifs/` — animations of the wave dynamics; every model is shown at the median and the largest evaluation parameter.

## The thesis document

The written thesis lives under [`tex/text/`](tex/text/) and is built with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd tex/text
tectonic -X build          # configuration in tex/text/Tectonic.toml
```

Chapters: introduction, mathematical background, methodology, numerical results, and conclusions
([`tex/text/src/chapters/`](tex/text/src/chapters/)). The compiled PDF is produced under `tex/text/build/`.

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
