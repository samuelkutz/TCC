import torch
from timeit import default_timer
from torch.optim import Adam

from FNO.fno_model import FNO2d, L2_loss, RelativeL2_loss, PINO2d, pino_loss


# Train FNO model
def train_model(x_train, y_train, epochs=3000, batch_size=16, lr=1e-3,
                modes1=16, modes2=16, width=32, device='cpu', seed=None, print_interval=500):
    """Train FNO model and return the trained model plus the relative loss history."""
    model = FNO2d(modes1=modes1, modes2=modes2, width=width).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = RelativeL2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss_history = []

    seed_tag = f"seed {seed}: " if seed is not None else ""
    print(f"{seed_tag}Starting FNO training...")
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        train_rel = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
            train_rel += loss.item()

        train_rel /= len(train_loader)
        train_loss_history.append(train_rel)

        if (ep + 1) % print_interval == 0 or ep == epochs - 1:
            elapsed = default_timer() - t0
            epoch_display = ep + 1
            print(f"{seed_tag}Epoch: {epoch_display}, Elapsed: {elapsed:.1f}s, Relative L2 Loss: {train_rel:.4e}")

    return model, train_loss_history


def train_pino_model(x_train, y_train, epochs=3000, batch_size=16, lr=1e-3,
                     modes1=16, modes2=16, width=32, dx=0.46875, dt=0.1171875,
                     phys_weight=1.0, ic_weight=0.1, data_weight=0.01, device='cpu', seed=None, print_interval=500):
    """Train a PINO2d model and return the trained model plus the relative loss history."""
    model = PINO2d(modes1=modes1, modes2=modes2, width=width, out_channels=y_train.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = RelativeL2_loss()

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss_history = []

    seed_tag = f"seed {seed}: " if seed is not None else ""
    print(f"{seed_tag}Starting PINO training...")
    t0 = default_timer()
    for ep in range(epochs):
        model.train()
        train_rel = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_data = pino_loss(model=model,
                                                          batch_x=batch_x,
                                                          batch_y=batch_y,
                                                          dx=dx,
                                                          dt=dt,
                                                          phys_weight=phys_weight,
                                                          ic_weight=ic_weight,
                                                          data_weight=data_weight)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = model(batch_x)
                rel_loss = loss_fn(pred, batch_y).item()
            train_rel += rel_loss

        train_rel /= len(train_loader)
        train_loss_history.append(train_rel)

        if (ep + 1) % print_interval == 0 or ep == epochs - 1:
            elapsed = default_timer() - t0
            epoch_display = ep + 1
            print(f"{seed_tag}Epoch: {epoch_display}, Elapsed: {elapsed:.1f}s, Relative L2 Loss: {train_rel:.4e}, PDE: {loss_pde.item():.6e}, IC: {loss_ic.item():.6e}, Data: {loss_data.item():.6e}")

    return model, train_loss_history


def save_model(model, filepath=None, *, model_name='model', epochs=None, n_samples=None, modes=(16,16), width=None, seed=None, extra=''):
    """Save model into RESULTS with informative filename when filepath is None.
    If filepath is provided, use it directly.
    """
    import os
    if filepath is None:
        os.makedirs('RESULTS', exist_ok=True)
        modes1, modes2 = modes
        parts = [f'{model_name}', f'epochs{epochs}', f'samples{n_samples}', f'modes{modes1}x{modes2}']
        if width is not None:
            parts.append(f'width{width}')
        if seed is not None:
            parts.append(f'seed{seed}')
        if extra:
            parts.append(extra)
        filename = '_'.join([p for p in parts if p]) + '.pth'
        filepath = os.path.join('RESULTS', filename)

    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath


def load_model(filepath, model, device='cpu'):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


# Multi-resolution test (returns relative errors per resolution and predictions)
def test_multi_resolution(model, test_resolutions, val_a=1.0, val_b=1.0, device='cpu'):
    from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq
    import numpy as np

    results = {}
    model.eval()
    for res in test_resolutions:
        bsq_test = Boussinesq(-30, 30, 0, 15, val_a, val_b)
        solver_test = PseudoSpectralBoussinesq(bsq_test, Nx=res, Nt=res-1, device=device)
        x_test, t_test, eta_true, u_true = solver_test.solve()

        ch0 = np.tile(eta_true[0:1, :].T, (1, res))
        ch1 = np.tile(u_true[0:1, :].T, (1, res))
        ch2 = np.ones((res, res)) * val_a
        ch3 = np.ones((res, res)) * val_b

        input_numpy = np.stack([ch0, ch1, ch2, ch3], axis=-1).astype(np.float32)
        input_tensor = torch.from_numpy(input_numpy).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_tensor = model(input_tensor)
            eta_pred = pred_tensor.squeeze().cpu().numpy()[0, :, :]

        eta_true_t = eta_true.T
        diff = np.abs(eta_true_t - eta_pred)
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(eta_true_t) + 1e-8)

        results[res] = {
            'x': x_test,
            't': t_test,
            'eta_true': eta_true_t,
            'eta_pred': eta_pred,
            'rel_error': float(rel_error)
        }
    return results


def train_pinn_model(param_value, epochs=3000, neurons=64, hidden_layers=4,
                     domain_points=3000, ic_points=500, optimizer_name='Adam', lr=1e-3,
                     data_weight=1.0, train_resolution=128, device='cpu', seed=None, print_interval=500):
    """Train a PINN on a single Boussinesq parameter case."""
    from PINN.PINN import PINN
    from BOUSSINESQ.boussinesq import Boussinesq, PseudoSpectralBoussinesq

    bsq = Boussinesq(-30, 30, 0, 15, param_value, param_value)
    solver = PseudoSpectralBoussinesq(bsq, Nx=train_resolution, Nt=train_resolution, device=device)
    x_sol, t_sol, eta_sol, u_sol = solver.solve()

    data = {'x': x_sol, 't': t_sol, 'eta': eta_sol, 'u': u_sol}
    pinn = PINN(input_size=2, output_size=2, neurons=neurons, hidden_layers=hidden_layers,
                Boussinesq=bsq, domain_points=domain_points, ic_points=ic_points,
                optimizer_name=optimizer_name, lr=lr, data=data, data_weight=data_weight,
                device=device)

    history = pinn.run_train_loop(bsq, epochs=epochs, seed=seed, print_interval=print_interval)
    return pinn, history
