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


def save_dataset(x_train, y_train, filepath, norm_stats=None):
    # save dataset tensors and optional normalization statistics to a pytorch .pth file
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    payload = {
        'x_train': x_train,
        'y_train': y_train,
    }
    if norm_stats is not None:
        payload['norm_stats'] = norm_stats
    torch.save(payload, filepath)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath):
    # load dataset tensors and optional normalization statistics from a pytorch .pth file
    data = torch.load(filepath)
    x_train = data['x_train']
    y_train = data['y_train']
    norm_stats = data.get('norm_stats', None)
    return x_train, y_train, norm_stats


def compute_norm_stats(x_train, y_train, eps=1e-12):
    # compute channel-wise min/max ranges for input and output tensors
    input_min = x_train.amin(dim=(0, 2, 3), keepdim=True)
    input_max = x_train.amax(dim=(0, 2, 3), keepdim=True)
    output_min = y_train.amin(dim=(0, 2, 3), keepdim=True)
    output_max = y_train.amax(dim=(0, 2, 3), keepdim=True)
    return {
        'input_min': input_min,
        'input_max': input_max,
        'output_min': output_min,
        'output_max': output_max,
        'eps': eps,
    }


def normalize_tensor(tensor, min_val, max_val, eps=1e-12):
    # map tensor values to [0, 1] range per channel: (x - min) / (max - min)
    return (tensor - min_val) / (max_val - min_val + eps)


def unnormalize_tensor(tensor, min_val, max_val, eps=1e-12):
    # map tensor from [0, 1] back to original scale: x * (max - min) + min
    return tensor * (max_val - min_val + eps) + min_val


def normalize_dataset(x_train, y_train, norm_stats):
    x_norm = normalize_tensor(x_train, norm_stats['input_min'], norm_stats['input_max'], norm_stats['eps'])
    y_norm = normalize_tensor(y_train, norm_stats['output_min'], norm_stats['output_max'], norm_stats['eps'])
    return x_norm, y_norm


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
