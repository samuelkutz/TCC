import torch


def save_model(model, filepath=None, *, model_name='model', epochs=None, n_samples=None, modes=(16,16), width=None, seed=None, extra=''):
    # save model weights to disk with descriptive filename metadata
    """save model weights to disk using a descriptive filename."""
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
    print(f'model saved to {filepath}')
    return filepath


def load_model(filepath, model, device='cpu'):
    # load model weights and set network to evaluation mode
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


class L2_loss(object):
    # standard l2 loss for model output compared to target data
    def __init__(self, eps=1e-10):
        super(L2_loss, self).__init__()
        self.eps = eps

    def __call__(self, x, y):
        # flattened l2 norm: sqrt(sum((x - y)^2)) per sample
        diff = x - y
        loss = torch.sqrt(torch.sum(diff * diff, dim=[1, 2, 3]) + self.eps)
        return torch.mean(loss)


class RelativeL2_loss(object):
    # relative l2 loss normalized by target norm
    def __init__(self, eps=1e-10):
        super(RelativeL2_loss, self).__init__()
        self.eps = eps

    def __call__(self, x, y):
        diff = x - y
        numerator = torch.sqrt(torch.sum(diff * diff, dim=[1, 2, 3]) + self.eps)
        denominator = torch.sqrt(torch.sum(y * y, dim=[1, 2, 3]) + self.eps)
        return torch.mean(numerator / (denominator + self.eps))
