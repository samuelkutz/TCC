import torch
import torch.nn as nn


_ACTIVATIONS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}


class MLP(nn.Module):
    """Linear+activation, then (hidden_layers-1) more, then a linear output.

    For input and output size 1 and width W the parameter count is
    3W + 1 + (hidden_layers - 1)(W^2 + W).
    """

    def __init__(self, input_size, output_size, neurons, hidden_layers,
                 activation='tanh', device='cpu'):
        super().__init__()
        self.device = torch.device(device)
        if activation.lower() not in _ACTIVATIONS:
            raise ValueError(f'unsupported activation: {activation}')
        act = _ACTIVATIONS[activation.lower()]

        layers = [nn.Linear(input_size, neurons), act()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), act()])
        layers.append(nn.Linear(neurons, output_size))

        self.net = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        return self.net(x)
