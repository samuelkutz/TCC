import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import normalize_tensor, unnormalize_tensor


class SpectralConv2d(nn.Module):
    """2D spectral convolution on the low Fourier modes (Li et al., 2021)."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # two weight tensors, one per sign of the full-FFT space axis: the low modes
        # live at both ends, so |k| up to ~modes1 is kept. A single centred band
        # would retain only half the bandwidth.
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (b, in_ch, kx, kt) x (in_ch, out_ch, kx, kt) -> (b, out_ch, kx, kt)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # keeping the same modes at any grid size is what makes the operator
        # resolution-invariant
        batchsize = x.shape[0]
        size1, size2 = x.size(-2), x.size(-1)
        x_ft = torch.fft.rfft2(x, norm='forward')

        m1 = min(self.modes1, size1 // 2)
        m2 = min(self.modes2, size2 // 2 + 1)

        out_ft = torch.zeros(
            batchsize, self.out_channels, size1, size2 // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(size1, size2), norm='forward')


class FNO2d(nn.Module):
    """Lifting, four Fourier layers each with a local 1x1 skip, projection
    (Li et al., 2021)."""
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.lifting = nn.Sequential(
            nn.Linear(6, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, self.width),
        )

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.projection = nn.Sequential(
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Linear(self.width * 2, 2),
        )

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(-1, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, size_y])
        gridy = torch.linspace(-1, 1, size_y, dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y).repeat([batchsize, size_x, 1])
        return torch.stack((gridx, gridy), dim=3)

    def forward(self, x):
        # the two normalized grid coordinates are appended to the four input channels
        grid = self.get_grid(x.shape, x.device)
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, grid), dim=-1)

        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # no activation after the last Fourier layer (canonical FNO)

        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def compute_loss(self, pred, target):
        diff = pred - target
        return torch.mean(diff * diff)

    def predict_physical(self, input_raw, norm_stats):
        """Normalize, forward, unnormalize; returns a prediction in physical units.

        Normalization stays on the norm_stats device and the forward pass on the
        model's, so the result does not depend on where either lives.
        """
        model_dev = next(self.parameters()).device
        stats_dev = norm_stats['input_min'].device
        x = normalize_tensor(input_raw, norm_stats['input_min'], norm_stats['input_max'], norm_stats['eps'])
        self.eval()
        with torch.no_grad():
            pred = self(x.to(model_dev))
        pred = pred.to(stats_dev)
        return unnormalize_tensor(pred, norm_stats['output_min'], norm_stats['output_max'], norm_stats['eps'])