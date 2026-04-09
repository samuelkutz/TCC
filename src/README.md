# FNO-BOUSSINESQ

Fourier Neural Operator (FNO) for solving the Boussinesq equation.

## Structure

- **boussinesq.py** - Boussinesq PDE solver (pseudospectral method)
- **fno_model.py** - FNO2d model with spectral convolution layers
- **dataset.py** - Dataset generation and I/O utilities
- **main.py** - Training, evaluation, and visualization

## Requirements

```bash
pip install torch numpy matplotlib
```

## Usage

Run the full pipeline:

```bash
python main.py
```

This will:
1. Generate dataset of Boussinesq solutions
2. Train FNO model (5000 epochs)
3. Save model checkpoint (`fno_model.pt`) and dataset (`dataset.pt`)
4. Create training loss plot
5. Generate comparison animation
6. Evaluate prediction errors

## Output Files

- `fno_model.pth` - Trained FNO model
- `dataset.pt` - Training dataset
- `training_loss.png` - Training loss history
- `boussinesq_a=1.00_b=1.00.gif` - Animation comparing FNO vs ground truth
- `error_analysis.png` - Error analysis for multiple parameter values

## Model Details

- **Model**: FNO2d with 4 spectral conv layers
- **Input channels**: 4 (eta_0, u_0, a, b)
- **Output channels**: 2 (eta, u)
- **Resolution**: 64×64
