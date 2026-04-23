import os
import torch


def save_model(model, filepath, *, model_name='model', epochs=None, n_samples=None, modes=(16,16), width=None, seed=None, extra=''):
    # save model weights to disk with descriptive filename metadata
    """save model weights to disk using a descriptive filename."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f'model saved to {filepath}')
    return filepath


def load_model(filepath, model, device):
    # load model weights and set network to evaluation mode
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_dataset(x_train, y_train, filepath):
    # save dataset tensors to a pytorch .pth file
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    torch.save({
        'x_train': x_train,
        'y_train': y_train
    }, filepath)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath):
    # load dataset tensors from a pytorch .pth file
    data = torch.load(filepath)
    return data['x_train'], data['y_train']


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
