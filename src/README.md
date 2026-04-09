# FNO-BOUSSINESQ

Fourier Neural Operator (FNO) and related physics-informed models for solving the Boussinesq equation.

## Structure

- **BOUSSINESQ/** - Boussinesq PDE solver, dataset utilities, and dataset generation script
- **FNO/** - FNO implementation and an experiment script for training and evaluation
- **PINO/** - PINO implementation and the corresponding experiment script
- **PINN/** - PINN implementation and the corresponding experiment script

## Requirements

```bash
pip install torch numpy matplotlib
```

## Usage

Run the per-method experiments directly:

```bash
python FNO/run_fno.py
python PINO/run_pino.py
python PINN/run_pinn.py
python BOUSSINESQ/run_dataset.py
```

This will:
1. generate the dataset for FNO/PINO experiments
2. train the chosen model
3. save the model checkpoint and dataset under `RESULTS/`
4. create training loss plots
5. generate evaluation plots and animation outputs

## Output Files

- `RESULTS/fno/` - FNO experiment outputs
- `RESULTS/pino/` - PINO experiment outputs
- `RESULTS/pinn/` - PINN experiment outputs
- `RESULTS/*.pth` - saved dataset and model files

- `error_analysis.png` - Error analysis for multiple parameter values

## Model Details

- **Model**: FNO2d with 4 spectral conv layers
- **Input channels**: 4 (eta_0, u_0, a, b)
- **Output channels**: 2 (eta, u)
- **Resolution**: 64×64
