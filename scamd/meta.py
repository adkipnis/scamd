"""Meta layers for preprocessing, scaling, and random activation choice.

These helpers are used to inject lightweight stochasticity around activations:
- ``Standardizer`` normalizes inputs using statistics from its first forward pass.
- ``RandomScale`` applies a random affine scaling.
- ``RandomChoice`` mixes outputs from randomly sampled activation modules.
"""

import torch
from torch import nn
from torch.nn import functional as F


# --- preprocessing layers
class Standardizer(nn.Module):
    """Standardize feature-wise using mean/std estimated on first call.

    The running statistics are lazily initialized and then reused for all future
    forwards, so this behaves like a one-shot fit + transform layer.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cache first-batch statistics and reuse them afterwards.
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True, unbiased=False)
            good = torch.isfinite(std) & (std > 1e-6)
            self.std = torch.where(good, std, torch.ones_like(std))
        return (x - self.mean) / self.std


class RandomScale(nn.Module):
    """Apply a random affine transform with positive scale."""

    def __init__(self):
        super().__init__()
        # Log-normal draw keeps scale strictly positive.
        self.scale = (2 * torch.randn(1)).exp()
        self.bias = torch.randn(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x + self.bias)


class RandomScaleFactory:
    """Factory producing `Standardizer -> RandomScale -> activation` blocks."""

    def __init__(self, act: nn.Module):
        self.act = act

    def __call__(self) -> nn.Module:
        return nn.Sequential(Standardizer(), RandomScale(), self.act())


# --- random choice layers
class RandomChoice(nn.Module):
    """Route each feature through one of `n_choice` randomly sampled activations."""

    def __init__(self, acts: list[nn.Module], n_choice: int = 1):
        super().__init__()
        assert len(acts), 'provided empty list of activations'
        assert n_choice > 0, 'number of choices must be positive'
        self.acts = acts
        self.n = len(acts)
        self.k = min(n_choice, self.n)

    def __repr__(self) -> str:
        return f'Random-{self.k}'

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        # Build a per-feature one-hot assignment across k sampled activations,
        # then shuffle so adjacent features do not always use the same branch.
        n = x.shape[-1]
        u = torch.arange(0, n) % self.k
        mask = F.one_hot(u, num_classes=self.k).float()
        perm = torch.randperm(n)
        return mask[perm]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample k activation classes (with replacement) on every forward call.
        indices = torch.randint(0, self.n, (self.k,))
        acts = [self.acts[idx]() for idx in indices]
        out = torch.stack([act(x) for act in acts], dim=-1)
        mask = self.mask(x)
        return (out * mask).sum(-1)


class RandomChoiceFactory:
    """Factory creating `RandomChoice` modules with fixed candidate pool."""

    def __init__(self, acts: list[nn.Module], n_choice: int = 1):
        self.acts = acts
        self.n_choice = n_choice

    def __call__(self) -> nn.Module:
        return RandomChoice(self.acts, self.n_choice)
