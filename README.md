# FNO-BOUSSINESQ

A Scientific Machine Learning project that compares Fourier Neural Operators (FNO), Physics-Informed Neural Operators (PINO), and Physics-Informed Neural Networks (PINN) on the nonlinear Boussinesq system.

## Project overview

This repository implements and evaluates three SciML approaches for the two-dimensional Boussinesq PDE system:

- **FNO**: a supervised spectral operator learning model.
- **PINO**: a hybrid model that learns an operator while enforcing the PDE residual.
- **PINN**: a classical physics-informed neural network that minimizes the PDE residual directly.

The reference solutions are produced by a pseudo-spectral solver using Fourier transforms and RK4 time integration. The goal is to compare how each method captures the wave dynamics, parameter dependence, and spectral content of the solution.

## Mathematical problem

The Boussinesq system models shallow-water waves with two fields:

- `η(x,t)`: free-surface displacement
- `u(x,t)`: horizontal velocity

The code uses the version:

- `η_t + u_x + α (η u)_x = 0`
- `u_t - (β / 3) u_{xxt} + η_x + α u u_x = 0`

where `α` and `β` are PDE parameters. The initial condition is a localized wave:

- `η(x,0) = A sech^2(x)`
- `u(x,0) = 0`

This system is solved with a pseudo-spectral reference solver and then approximated by SciML models.

## Models implemented

### FNO (Fourier Neural Operator)

Implemented in `src/FNO/FNO.py`, the FNO model:

- lifts the input to a latent channel space with `Linear` layers,
- applies a stack of spectral convolution layers (`SpectralConv2d`) in Fourier space,
- uses low-frequency modes only, which improves efficiency and stability,
- returns a 2-channel output `(η, u)`.

Input channels are:

1. initial surface `η_0`
2. initial velocity `u_0`
3. PDE parameter `α`
4. PDE parameter `β`
5. grid coordinate `x`
6. grid coordinate `t`

The FNO architecture combines global spectral convolutions and point-wise linear skips, enabling it to learn maps from PDE parameters and initial conditions to full spatio-temporal solutions.

### PINO (Physics-Informed Neural Operator)

Implemented in `src/PINO/PINO.py`, PINO uses the same spectral operator backbone as FNO, but adds PDE constraints to the loss.

The training objective contains three terms:

- PDE residual loss: enforces the Boussinesq equations using spectral derivatives and finite-difference time derivatives,
- initial condition loss: enforces `(η, u)` at `t = 0`,
- data loss: optionally enforces supervised fit to reference solution samples.

This repository supports two PINO variants:

- `with_data`: uses both PDE and data losses,
- `no_data`: uses only PDE residual and initial condition loss.

The physics loss is computed in physical units by un-normalizing the model output before evaluating the residual.

### PINN (Physics-Informed Neural Network)

Implemented in `src/PINN/PINN.py`, the PINN is a fully connected network that directly maps `(x,t)` to `(η, u)`.

Key features:

- uses automatic differentiation to compute derivatives `η_t`, `η_x`, `u_t`, `u_x`, `u_{xx}`, and `u_{xxt}`,
- trains on PDE residuals and initial condition loss,
- supports supervised data via a data loss term when model `data_weight > 0`.

This repository also evaluates PINN in two modes:

- `with_data`: supervised data plus PDE and IC losses,
- `no_data`: purely physics-informed training using only PDE and IC losses.

## Reference solver and dataset

The Boussinesq model and reference solver live in `src/BOUSSINESQ/boussinesq.py`:

- `Boussinesq`: defines domain, parameters, initial condition, and PDE residual,
- `PseudoSpectralBoussinesq`: solves the system in Fourier space with RK4 and returns `(x, t, η, u)`.

Dataset generation is handled by `src/BOUSSINESQ/run_dataset.py` and stores normalized solutions into `results/models/boussinesq_dataset.pth`.

The data pipeline uses `src/tools.py` for:

- normalization/un-normalization,
- dataset save/load,
- model save/load,
- L2 and relative L2 losses.

## Experiments

The main controlled experiments are:

1. **FNO supervised training** on the generated dataset,
2. **PINO with PDE + data** and **PINO without data**,
3. **PINN with data** and **PINN without data**.

The evaluations compare:

- training loss history,
- relative error across space-time,
- spectral error and frequency content,
- performance across multiple resolution settings,
- behavior under varying PDE parameters `α` and `β`.

A main research question is whether PINNs suffer spectral bias while FNO/PINO better capture high-frequency components.

## How to run

The most convenient entry point is:

```bash
python src/main.py
```

This script runs the full pipeline:

- generate the shared dataset,
- train FNO,
- train PINO with and without data,
- train PINN with and without data,
- generate evaluation plots for both evaluation panels.

If you want to run only dataset generation:

```bash
python src/BOUSSINESQ/run_dataset.py
```

The model training functions are exposed in:

- `src/FNO/train_fno.py`
- `src/PINO/train_pino.py`
- `src/PINN/train_pinn.py`

Those scripts can be imported or called from custom wrappers.

## Output structure

The repository saves outputs in a standardized `results/` hierarchy:

- `results/models/boussinesq_dataset.pth` — shared dataset file,
- `results/models/weights/fno/` — FNO weights,
- `results/models/weights/pino/with_data/` — PINO with-data weights,
- `results/models/weights/pino/no_data/` — PINO no-data weights,
- `results/models/weights/pinn/with_data/` — PINN with-data weights,
- `results/models/weights/pinn/no_data/` — PINN no-data weights,
- `results/models/metadata/` — saved training metadata for all experiments,
- `results/imgs/eval1/` — first set of evaluation plots,
- `results/imgs/eval2/` — second set of evaluation plots.

## Code structure

- `src/main.py`: orchestrates the full dataset/train/evaluate workflow,
- `src/BOUSSINESQ/`: PDE solving, dataset generation,
- `src/FNO/`: spectral operator model plus FNO training,
- `src/PINO/`: physics-informed operator model plus PINO training,
- `src/PINN/`: classical physics-informed neural network training,
- `src/tools.py`: utilities for normalization, saving/loading, and losses,
- `src/_plots.py`: shared plotting utilities for evaluation figures.

## Requirements

```bash
pip install torch numpy matplotlib
```

Optional but recommended:

```bash
pip install torch torchvision
```

## Research context

This code is built from the thesis project "Scientific Machine Learning Methods for the Nonlinear Boussinesq System of Equations". It is intended to:

- bridge classical PDE solvers with modern SciML architectures,
- analyze the effect of physics knowledge versus purely data-driven training,
- compare operator learning (FNO/PINO) with pointwise PINN learning.

## Notes

- The code already uses a normalized dataset when training FNO, PINO, and PINN.
- PINO is designed to support both data-constrained and fully physics-constrained training.
- The Boussinesq residual is computed using spectral spatial derivatives and finite-difference time derivatives in `src/PINO/PINO.py`.
- FNO uses a low-mode Fourier representation via `SpectralConv2d`, which restricts the learned operator to dominant frequency components.

## Suggested next steps

- add a dedicated CLI wrapper for each model,
- include a detailed `results/` summary table for accuracy vs resolution,
- add a README section with example evaluation plots and interpretation.
