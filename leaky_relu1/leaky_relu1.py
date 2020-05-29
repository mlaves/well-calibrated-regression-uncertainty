import torch


def leaky_relu1(x, slope=0.1, cutoff=1.0):
    x = torch.nn.functional.leaky_relu(x, negative_slope=slope)
    x = -torch.nn.functional.leaky_relu(-x+cutoff, negative_slope=slope)+cutoff
    return x


class LeakyReLU1(torch.nn.Module):
    def __init__(self, slope=0.1, cutoff=1.0):
        super().__init__()
        self._slope = slope
        self._cutoff = cutoff

    def forward(self, x):
        return leaky_relu1(x, slope=self._slope, cutoff=self._cutoff)
