"""Gaussian-process-inspired nonlinear activation modules."""

from typing import Literal
import torch
from torch import nn
from torch import distributions as D
from .utils import logUniform, getRng
from .meta import Standardizer


# --- GP based activations
class MaternKernel:
    def __init__(self) -> None:
        self.df = getRng().choice([1, 3, 5]) / 2

    def __repr__(self) -> str:
        return f'Matern-{self.df}'

    def __call__(self, k: int, ell: float):
        scale = self.df**0.5 / ell
        freqs = D.StudentT(df=self.df).sample((k,)) * scale
        factor = (2 / k) ** 0.5
        return freqs, factor


class SEKernel:  # squared exponential / RBF
    def __repr__(self) -> str:
        return 'SE'

    def __call__(self, k: int, ell: float):
        freqs = torch.randn(k) / ell
        factor = (2 / k) ** 0.5
        return freqs, factor


class FractionalKernel:  # scale-free fractional kernel
    def __repr__(self) -> str:
        return 'Fractional'

    def __call__(self, k: int, ell: float = 0.0):
        freqs = (k * torch.rand(k)).clamp_min(1e-6)
        decay_exponent = -float(logUniform(getRng(), 0.7, 3.0))
        factor = freqs**decay_exponent
        factor = factor / (factor**2).sum().sqrt()
        return freqs, factor


KERNELS = {
    'matern': MaternKernel,
    'se': SEKernel,
    'fractional': FractionalKernel,
}


class GP(nn.Module):

    # sample from a GP with a random kernel [SE, Matern, Fractal]
    def __init__(
        self,
        k: int = 512,
        gp_type: Literal['se', 'matern', 'fractional'] | None = None,
    ):
        super().__init__()
        self.standardizer = Standardizer()
        # choose kernel
        if gp_type is None:
            gp_type = getRng().choice(
                list(KERNELS.keys()),
                p=[0.5, 0.2, 0.3],
            )
        elif gp_type not in KERNELS:
            raise ValueError(f'Kernel not found in {list(KERNELS.keys())}')
        self.kernel = KERNELS[gp_type]()   # type: ignore

        # setup parameters
        ell = logUniform(getRng(), 0.1, 16.0)
        self.freqs, factor = self.kernel(k, ell)
        self.bias = 2 * torch.pi * torch.rand(k)
        self.weight = factor * torch.randn(k)

    def __repr__(self) -> str:
        return f'GP-{self.kernel}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardizer(x)
        phi = torch.cos(self.freqs * x.unsqueeze(-1) + self.bias)
        return phi @ self.weight
