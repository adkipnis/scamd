"""Gaussian-process-inspired nonlinear activation modules.

Each GP instance samples its own kernel, length-scale, and random Fourier
features from an explicit ``rng``, avoiding shared global random state.
"""

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch import distributions as D

from scamd.utils import logUniform
from scamd.meta import Standardizer


class MaternKernel:
    """Random Fourier features for a Matérn kernel with sampled smoothness."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.df = rng.choice([1, 3, 5]) / 2

    def __repr__(self) -> str:
        return f'Matern-{self.df}'

    def __call__(self, k: int, ell: float):
        scale = self.df**0.5 / ell
        freqs = D.StudentT(df=self.df).sample((k,)) * scale
        factor = (2 / k) ** 0.5
        return freqs, factor


class SEKernel:
    """Random Fourier features for the squared-exponential (RBF) kernel."""

    def __init__(self, rng: np.random.Generator) -> None:
        pass  # no rng needed at construction

    def __repr__(self) -> str:
        return 'SE'

    def __call__(self, k: int, ell: float):
        freqs = torch.randn(k) / ell
        factor = (2 / k) ** 0.5
        return freqs, factor


class FractionalKernel:
    """Scale-free fractional kernel with a sampled power-law decay."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def __repr__(self) -> str:
        return 'Fractional'

    def __call__(self, k: int, ell: float = 0.0):
        freqs = (k * torch.rand(k)).clamp_min(1e-6)
        decay_exponent = -float(logUniform(self.rng, 0.7, 3.0))
        factor = freqs**decay_exponent
        factor = factor / (factor**2).sum().sqrt()
        return freqs, factor


KERNELS = {
    'matern': MaternKernel,
    'se': SEKernel,
    'fractional': FractionalKernel,
}


class GP(nn.Module):
    """Random-feature approximation to a GP with a sampled kernel.

    On each instantiation, a kernel family is chosen (or passed explicitly),
    and random Fourier features are drawn.  The result is a single-input,
    single-output smooth nonlinearity.
    """

    def __init__(
        self,
        k: int = 512,
        gp_type: Literal['se', 'matern', 'fractional'] | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        if rng is None:
            rng = np.random.default_rng()
        self.standardizer = Standardizer()

        # choose kernel
        if gp_type is None:
            gp_type = rng.choice(
                list(KERNELS.keys()),
                p=[0.5, 0.2, 0.3],
            )
        elif gp_type not in KERNELS:
            raise ValueError(f'Unknown kernel; choose from {list(KERNELS.keys())}')
        self.kernel = KERNELS[gp_type](rng)

        # sample random Fourier feature parameters
        ell = logUniform(rng, 0.1, 16.0)
        freqs, factor = self.kernel(k, ell)
        bias = 2 * torch.pi * torch.rand(k)
        weight = factor * torch.randn(k)
        self.register_buffer('freqs', freqs)
        self.register_buffer('bias', bias)
        self.register_buffer('weight', weight)

    def __repr__(self) -> str:
        return f'GP-{self.kernel}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardizer(x)
        phi = torch.cos(self.freqs * x.unsqueeze(-1) + self.bias)
        return phi @ self.weight
